// stream_codec_bench2.rs — nested-radix, frame-parallel, and stream-ILP rows
//
// Uniform ternary weights w in {-1, 0, +1} map to radix digits t = w + 1.
// Every timed decode consumes activations immediately and checks its i64 dot
// product against the same unpacked reference over the same 2^24-trit corpus.
//
// Rows:
//   B   rung (5,8): one 256-entry LUT lookup per five trits.
//   SB  packed rung (306,485): fixed 485-bit blocks concatenated without
//       per-block alignment, extracted by byte+shift arithmetic into u64x8,
//       then decoded as 61 base-243 digits plus one final trit.
//   FP  486-frame: 7 independent (41,65) u128 blocks plus one (19,31)
//       block.  The seven u128 chains and dot accumulators are unrolled.
//   I8  eight round-robin carry-counting range streams, advanced together on
//       one core or one stream per thread.
//
// Build:
//   ~/.cargo/bin/rustc -O -C target-cpu=native stream_codec_bench2.rs

// The SB byte count is ceil(total_blocks * 485 / 8), rounded once at the end
// of the entire stream.  The allocation has nine zero guard bytes solely so
// the final unaligned extraction can use the same fast path; those bytes are
// not part of the payload or the reported bpw.

use std::process::Command;
use std::thread;
use std::time::Instant;

const NT: usize = 1 << 24;
const SUPER_TRITS: usize = 306;
const SUPER_BITS: usize = 485;
const FRAME_BITS: usize = 486;
const STREAMS: usize = 8;

const STREAM_FULL: u64 = 1u64 << 32;
const STREAM_MASK: u64 = STREAM_FULL - 1;
const STREAM_TOP: u64 = 1u64 << 24;
const STREAM_FLUSH_SHIFTS: usize = 5;
const STREAM_INIT_BYTES: usize = 5;

fn lut243() -> [[i8; 5]; 256] {
    let mut lut = [[0i8; 5]; 256];
    for v in 0..243usize {
        let mut x = v;
        for k in 0..5 {
            lut[v][k] = (x % 3) as i8 - 1;
            x /= 3;
        }
    }
    lut
}

#[inline(always)]
fn dot5(ws: &[i8; 5], acts: &[i8], offset: usize) -> i64 {
    // All callers establish that offset..offset+5 is in bounds.  Removing
    // five repeated bounds checks keeps the benchmark focused on decode cost.
    unsafe {
        ws[0] as i64 * *acts.get_unchecked(offset) as i64
            + ws[1] as i64 * *acts.get_unchecked(offset + 1) as i64
            + ws[2] as i64 * *acts.get_unchecked(offset + 2) as i64
            + ws[3] as i64 * *acts.get_unchecked(offset + 3) as i64
            + ws[4] as i64 * *acts.get_unchecked(offset + 4) as i64
    }
}

fn report_bytes(name: &str, nbytes: usize, secs: f64) {
    let bpw = 8.0 * nbytes as f64 / NT as f64;
    let gtps = NT as f64 / secs / 1e9;
    let gbs = gtps * bpw / 8.0;
    println!("{name:<30} {gtps:7.3} Gtrit/s   {gbs:7.3} GB/s   {bpw:.5} bpw");
}

fn report_bits(name: &str, nbits: usize, secs: f64) {
    let bpw = nbits as f64 / NT as f64;
    let gtps = NT as f64 / secs / 1e9;
    let gbs = gtps * bpw / 8.0;
    println!("{name:<30} {gtps:7.3} Gtrit/s   {gbs:7.3} GB/s   {bpw:.5} bpw");
}

#[inline]
fn reference_dot(trits: &[i8], acts: &[i8]) -> i64 {
    trits
        .iter()
        .zip(acts)
        .map(|(&w, &a)| w as i64 * a as i64)
        .sum()
}

// -------------------------------------------------------------------------
// SB: back-to-back fixed-width (306,485) radix blocks.

struct SuperblockStream {
    data: Vec<u8>,
    payload_bytes: usize,
    blocks: usize,
}

#[inline]
fn radix306(block: &[i8]) -> [u64; 8] {
    let mut limbs = [0u64; 8];
    for &w in block.iter().rev() {
        let mut carry = (w + 1) as u64 as u128;
        for limb in &mut limbs {
            let value = *limb as u128 * 3 + carry;
            *limb = value as u64;
            carry = value >> 64;
        }
        assert_eq!(carry, 0, "306-trit radix value exceeded 512 bits");
    }
    assert_eq!(limbs[7] >> 37, 0, "306-trit radix value exceeded 485 bits");
    limbs
}

fn encode_superblocks(trits: &[i8]) -> SuperblockStream {
    let blocks = trits.len().div_ceil(SUPER_TRITS);
    let payload_bits = blocks * SUPER_BITS;
    let payload_bytes = payload_bits.div_ceil(8);
    let mut data = vec![0u8; payload_bytes + 9];

    for (block_index, block) in trits.chunks(SUPER_TRITS).enumerate() {
        let limbs = radix306(block);
        let stream_bit = block_index * SUPER_BITS;

        // Packing is outside the decode timer.  Writing explicit bits makes
        // the fixed 485-bit contract transparent, including shared bytes at
        // every non-byte-aligned block boundary.
        for bit in 0..SUPER_BITS {
            let value = (limbs[bit / 64] >> (bit % 64)) & 1;
            if value != 0 {
                let dst = stream_bit + bit;
                data[dst / 8] |= 1u8 << (dst % 8);
            }
        }
    }

    SuperblockStream {
        data,
        payload_bytes,
        blocks,
    }
}

#[inline(always)]
fn load_u64_le(data: &[u8], byte: usize) -> u64 {
    // SuperblockStream supplies enough zero guard bytes for the last load.
    unsafe { u64::from_le((data.as_ptr().add(byte) as *const u64).read_unaligned()) }
}

#[inline(always)]
fn extract_485(data: &[u8], bit_offset: usize) -> [u64; 8] {
    let mut limbs = [0u64; 8];
    for (limb_index, limb) in limbs.iter_mut().enumerate() {
        let source_bit = bit_offset + limb_index * 64;
        let byte = source_bit >> 3;
        let shift = source_bit & 7;
        let low = load_u64_le(data, byte);
        *limb = if shift == 0 {
            low
        } else {
            (low >> shift) | ((data[byte + 8] as u64) << (64 - shift))
        };
    }
    // The last limb contains only bits 448..484.  Bits above them belong to
    // the next packed block (or the guard) and must not enter radix division.
    limbs[7] &= (1u64 << 37) - 1;
    limbs
}

#[inline(always)]
fn divmod_limbs(limbs: &mut [u64; 8], divisor: u64) -> u64 {
    let mut remainder = 0u64;
    for limb in limbs.iter_mut().rev() {
        let value = ((remainder as u128) << 64) | *limb as u128;
        *limb = (value / divisor as u128) as u64;
        remainder = (value % divisor as u128) as u64;
    }
    remainder
}

fn decode_superblocks(
    packed: &SuperblockStream,
    acts: &[i8],
    lut: &[[i8; 5]; 256],
) -> i64 {
    let mut dot = 0i64;
    for block_index in 0..packed.blocks {
        let mut limbs = extract_485(&packed.data, block_index * SUPER_BITS);
        let base = block_index * SUPER_TRITS;
        let count = (NT - base).min(SUPER_TRITS);

        if count == SUPER_TRITS {
            for digit in 0..61 {
                let radix_digit = divmod_limbs(&mut limbs, 243) as usize;
                dot += dot5(&lut[radix_digit], acts, base + digit * 5);
            }
            dot += (limbs[0] as i64 - 1) * acts[base + 305] as i64;
        } else {
            // Only the corpus tail can take this path.  It remains inside the
            // timed fixed-width extraction and verifies every real trit.
            for k in 0..count {
                let trit = divmod_limbs(&mut limbs, 3) as i64 - 1;
                dot += trit * acts[base + k] as i64;
            }
        }
    }
    dot
}

// -------------------------------------------------------------------------
// FP: 486-bit frames = 7x(41,65) + (19,31).

#[derive(Clone, Copy)]
struct Frame486 {
    blocks: [u128; 7],
    tail: u32,
}

fn encode_frames(trits: &[i8]) -> Vec<Frame486> {
    let frame_count = trits.len().div_ceil(SUPER_TRITS);
    let mut frames = Vec::with_capacity(frame_count);

    for frame_index in 0..frame_count {
        let frame_base = frame_index * SUPER_TRITS;
        let mut blocks = [0u128; 7];
        for (j, block) in blocks.iter_mut().enumerate() {
            let base = frame_base + j * 41;
            for k in (0..41).rev() {
                let w = trits.get(base + k).copied().unwrap_or(-1);
                *block = *block * 3 + (w + 1) as u128;
            }
        }

        let mut tail = 0u32;
        for k in (0..19).rev() {
            let w = trits.get(frame_base + 287 + k).copied().unwrap_or(-1);
            tail = tail * 3 + (w + 1) as u32;
        }
        frames.push(Frame486 { blocks, tail });
    }
    frames
}

#[inline(always)]
fn decode_frame_ilp(frame: &Frame486, acts: &[i8], lut: &[[i8; 5]; 256]) -> i64 {
    debug_assert!(acts.len() >= SUPER_TRITS);

    // These are deliberately separate dependency chains and accumulators.
    // Each digit round exposes seven independent u128 /243 operations.
    let mut x0 = frame.blocks[0];
    let mut x1 = frame.blocks[1];
    let mut x2 = frame.blocks[2];
    let mut x3 = frame.blocks[3];
    let mut x4 = frame.blocks[4];
    let mut x5 = frame.blocks[5];
    let mut x6 = frame.blocks[6];
    let mut d0 = 0i64;
    let mut d1 = 0i64;
    let mut d2 = 0i64;
    let mut d3 = 0i64;
    let mut d4 = 0i64;
    let mut d5 = 0i64;
    let mut d6 = 0i64;

    macro_rules! digit_step {
        ($x:ident, $dot:ident, $block:expr, $digit:expr) => {{
            let r = ($x % 243) as usize;
            $x /= 243;
            $dot += dot5(&lut[r], acts, $block * 41 + $digit * 5);
        }};
    }

    for digit in 0..8 {
        digit_step!(x0, d0, 0, digit);
        digit_step!(x1, d1, 1, digit);
        digit_step!(x2, d2, 2, digit);
        digit_step!(x3, d3, 3, digit);
        digit_step!(x4, d4, 4, digit);
        digit_step!(x5, d5, 5, digit);
        digit_step!(x6, d6, 6, digit);
    }

    d0 += (x0 as i64 - 1) * acts[40] as i64;
    d1 += (x1 as i64 - 1) * acts[81] as i64;
    d2 += (x2 as i64 - 1) * acts[122] as i64;
    d3 += (x3 as i64 - 1) * acts[163] as i64;
    d4 += (x4 as i64 - 1) * acts[204] as i64;
    d5 += (x5 as i64 - 1) * acts[245] as i64;
    d6 += (x6 as i64 - 1) * acts[286] as i64;

    let mut tail = frame.tail;
    let mut dt = 0i64;
    for digit in 0..3 {
        let r = (tail % 243) as usize;
        tail /= 243;
        dt += dot5(&lut[r], acts, 287 + digit * 5);
    }
    for k in 15..19 {
        dt += ((tail % 3) as i64 - 1) * acts[287 + k] as i64;
        tail /= 3;
    }

    d0 + d1 + d2 + d3 + d4 + d5 + d6 + dt
}

fn decode_frame_range(
    frames: &[Frame486],
    acts: &[i8],
    first_frame: usize,
    lut: &[[i8; 5]; 256],
) -> i64 {
    let mut dot = 0i64;
    for (local, frame) in frames.iter().enumerate() {
        let base = (first_frame + local) * SUPER_TRITS;
        dot += decode_frame_ilp(frame, &acts[base..base + SUPER_TRITS], lut);
    }
    dot
}

fn physical_core_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = Command::new("sysctl").args(["-n", "hw.physicalcpu"]).output() {
            if output.status.success() {
                if let Ok(text) = std::str::from_utf8(&output.stdout) {
                    if let Ok(cores) = text.trim().parse::<usize>() {
                        if cores != 0 {
                            return cores;
                        }
                    }
                }
            }
        }
    }

    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn decode_frames_threaded(
    frames: &[Frame486],
    acts: &[i8],
    lut: &[[i8; 5]; 256],
    thread_count: usize,
) -> i64 {
    let chunk = frames.len().div_ceil(thread_count);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(thread_count);
        for worker in 0..thread_count {
            let start = worker * chunk;
            if start >= frames.len() {
                break;
            }
            let end = (start + chunk).min(frames.len());
            handles.push(scope.spawn(move || {
                decode_frame_range(&frames[start..end], acts, start, lut)
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("FRAME-PAR worker panicked"))
            .sum()
    })
}

// -------------------------------------------------------------------------
// I8: independent carry-counting range streams.

struct StreamEncoder {
    low: u64,
    range: u64,
    cache: u8,
    pending_ff: usize,
    out: Vec<u8>,
}

impl StreamEncoder {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            low: 0,
            range: STREAM_FULL,
            cache: 0,
            pending_ff: 0,
            out: Vec::with_capacity(capacity),
        }
    }

    #[inline(always)]
    fn shift_low(&mut self) {
        let top = self.low >> 24;
        if top != 0xff {
            let carry = (self.low >> 32) as u8;
            self.out.push(self.cache.wrapping_add(carry));
            if self.pending_ff != 0 {
                let fill = 0xffu8.wrapping_add(carry);
                self.out.resize(self.out.len() + self.pending_ff, fill);
                self.pending_ff = 0;
            }
            self.cache = (top & 0xff) as u8;
        } else {
            self.pending_ff += 1;
        }
        self.low = (self.low << 8) & STREAM_MASK;
        self.range = self.range.wrapping_shl(8);
    }

    #[inline(always)]
    fn encode(&mut self, trit: u8) {
        let third = self.range / 3;
        let two_thirds = third + third;
        match trit {
            0 => self.range = third,
            1 => {
                self.low += third;
                self.range = third;
            }
            2 => {
                self.low += two_thirds;
                self.range -= two_thirds;
            }
            _ => unreachable!("ternary symbol outside 0..=2"),
        }
        while self.range < STREAM_TOP {
            self.shift_low();
        }
    }

    fn finish(mut self) -> Vec<u8> {
        for _ in 0..STREAM_FLUSH_SHIFTS {
            self.shift_low();
        }
        self.out
    }
}

fn encode_stream(weights: &[i8]) -> Vec<u8> {
    let mut encoder = StreamEncoder::with_capacity(weights.len() / 5 + 8);
    for &w in weights {
        encoder.encode((w + 1) as u8);
    }
    encoder.finish()
}

struct StreamDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    low: u32,
    range: u64,
    code: u32,
}

impl<'a> StreamDecoder<'a> {
    #[inline]
    fn new(data: &'a [u8]) -> Self {
        let mut decoder = Self {
            data,
            pos: 0,
            low: 0,
            range: STREAM_FULL,
            code: 0,
        };
        for _ in 0..STREAM_INIT_BYTES {
            decoder.code = decoder.code.wrapping_shl(8) | decoder.read_byte() as u32;
        }
        decoder
    }

    #[inline(always)]
    fn read_byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let byte = self.data[self.pos];
            self.pos += 1;
            byte
        } else {
            0
        }
    }

    #[inline(always)]
    fn decode(&mut self) -> u8 {
        let third = self.range / 3;
        let two_thirds = third + third;
        let value = self.code.wrapping_sub(self.low) as u64;

        let trit;
        if value < third {
            trit = 0;
            self.range = third;
        } else if value < two_thirds {
            trit = 1;
            self.low = self.low.wrapping_add(third as u32);
            self.range = third;
        } else {
            trit = 2;
            self.low = self.low.wrapping_add(two_thirds as u32);
            self.range -= two_thirds;
        }

        while self.range < STREAM_TOP {
            self.low = self.low.wrapping_shl(8);
            self.code = self.code.wrapping_shl(8) | self.read_byte() as u32;
            self.range <<= 8;
        }
        trit
    }
}

fn verify_range_known_answer() {
    let weights = [
        -1, 0, 1, 1, 0, -1, 1, -1, 0, -1, 0, 1, 1, 0, -1, 1, -1, 0, -1, 0, 1, 1, 0,
        -1, 1, -1, 0, -1, 0, 1, 1,
    ];
    let expected = [
        0x00, 0x37, 0x08, 0x24, 0x69, 0x16, 0xa0, 0xed, 0xa3, 0x1c, 0x00,
    ];
    let encoded = encode_stream(&weights);
    assert_eq!(encoded, expected, "carry-counting known-answer mismatch");
    let mut decoder = StreamDecoder::new(&encoded);
    for &w in &weights {
        assert_eq!(decoder.decode() as i8 - 1, w, "known-answer decode mismatch");
    }
}

fn decode_interleave8(streams: &[Vec<u8>], acts: &[i8]) -> i64 {
    assert_eq!(streams.len(), STREAMS);
    let rounds = NT / STREAMS;
    assert!(streams.iter().all(|stream| !stream.is_empty()));

    let mut s0 = StreamDecoder::new(&streams[0]);
    let mut s1 = StreamDecoder::new(&streams[1]);
    let mut s2 = StreamDecoder::new(&streams[2]);
    let mut s3 = StreamDecoder::new(&streams[3]);
    let mut s4 = StreamDecoder::new(&streams[4]);
    let mut s5 = StreamDecoder::new(&streams[5]);
    let mut s6 = StreamDecoder::new(&streams[6]);
    let mut s7 = StreamDecoder::new(&streams[7]);
    let mut d0 = 0i64;
    let mut d1 = 0i64;
    let mut d2 = 0i64;
    let mut d3 = 0i64;
    let mut d4 = 0i64;
    let mut d5 = 0i64;
    let mut d6 = 0i64;
    let mut d7 = 0i64;

    for round in 0..rounds {
        let base = round * STREAMS;
        d0 += (s0.decode() as i64 - 1) * acts[base] as i64;
        d1 += (s1.decode() as i64 - 1) * acts[base + 1] as i64;
        d2 += (s2.decode() as i64 - 1) * acts[base + 2] as i64;
        d3 += (s3.decode() as i64 - 1) * acts[base + 3] as i64;
        d4 += (s4.decode() as i64 - 1) * acts[base + 4] as i64;
        d5 += (s5.decode() as i64 - 1) * acts[base + 5] as i64;
        d6 += (s6.decode() as i64 - 1) * acts[base + 6] as i64;
        d7 += (s7.decode() as i64 - 1) * acts[base + 7] as i64;
    }
    d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7
}

fn decode_interleave8_threaded(streams: &[Vec<u8>], acts: &[i8]) -> i64 {
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(STREAMS);
        for stream_index in 0..STREAMS {
            handles.push(scope.spawn(move || {
                let mut decoder = StreamDecoder::new(&streams[stream_index]);
                let mut dot = 0i64;
                for index in (stream_index..NT).step_by(STREAMS) {
                    dot += (decoder.decode() as i64 - 1) * acts[index] as i64;
                }
                dot
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("INTERLEAVE-8 worker panicked"))
            .sum()
    })
}

fn main() {
    verify_range_known_answer();

    // Same seed, xorshift update, symbol mapping, and activation generation as
    // ternary_pack_bench.rs and stream_codec_bench.rs.
    let mut state = 0x243F6A8885A308D3u64;
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let trits: Vec<i8> = (0..NT).map(|_| (rnd() % 3) as i8 - 1).collect();
    let acts: Vec<i8> = (0..NT).map(|_| (rnd() % 255) as i8).collect();
    let reference = reference_dot(&trits, &acts);
    let lut = lut243();

    // B: the original (5,8) lookup baseline.
    let packed5: Vec<u8> = trits
        .chunks(5)
        .map(|chunk| {
            chunk
                .iter()
                .rev()
                .fold(0u16, |x, &w| x * 3 + (w + 1) as u16) as u8
        })
        .collect();

    // SB: the only allocation size counted is payload_bytes.  Guard bytes
    // support the last unaligned read and remain zero.
    let superblocks = encode_superblocks(&trits);

    // FP: zero-weight activation padding lets the last physical frame use the
    // exact same ILP decode path without changing the corpus dot product.
    let frames = encode_frames(&trits);
    let mut frame_acts = acts.clone();
    frame_acts.resize(frames.len() * SUPER_TRITS, 0);

    // I8: trit i belongs to stream i mod 8, preserving each stream's order.
    let mut stream_weights: [Vec<i8>; STREAMS] =
        std::array::from_fn(|_| Vec::with_capacity(NT / STREAMS));
    for (index, &w) in trits.iter().enumerate() {
        stream_weights[index % STREAMS].push(w);
    }
    assert!(stream_weights.iter().all(|stream| stream.len() == NT / STREAMS));
    let streams: Vec<Vec<u8>> = stream_weights
        .iter()
        .map(|weights| encode_stream(weights))
        .collect();
    let stream_bytes: usize = streams.iter().map(Vec::len).sum();

    println!("corpus: {NT} trits; reference dot: {reference}");
    println!(
        "SB payload: {} blocks, {} bits -> {} bytes (single end rounding)",
        superblocks.blocks,
        superblocks.blocks * SUPER_BITS,
        superblocks.payload_bytes
    );
    let physical_cores = physical_core_count();
    println!("FRAME-PAR workers: {physical_cores} physical cores");
    println!("{:<30} {:>14}   {:>13}   {}", "row", "decode", "stream", "density");

    let t0 = Instant::now();
    let mut dot = 0i64;
    for (chunk_index, &byte) in packed5.iter().enumerate() {
        let base = chunk_index * 5;
        let count = (NT - base).min(5);
        let weights = &lut[byte as usize];
        for k in 0..count {
            dot += weights[k] as i64 * acts[base + k] as i64;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(dot, reference, "(5,8) LUT dot mismatch");
    report_bytes("B  LUT (5,8)", packed5.len(), secs);

    let t0 = Instant::now();
    let dot = decode_superblocks(&superblocks, &acts, &lut);
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(dot, reference, "SUPERBLOCK dot mismatch");
    report_bytes("SB SUPERBLOCK (306,485)", superblocks.payload_bytes, secs);

    let frame_payload_bits = frames.len() * FRAME_BITS;
    let t0 = Instant::now();
    let dot = decode_frame_range(&frames, &frame_acts, 0, &lut);
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(dot, reference, "FRAME-PAR single-thread dot mismatch");
    report_bits("FP FRAME-PAR 1-thread", frame_payload_bits, secs);

    let t0 = Instant::now();
    let dot = decode_frames_threaded(&frames, &frame_acts, &lut, physical_cores);
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(dot, reference, "FRAME-PAR threaded dot mismatch");
    report_bits(
        &format!("FP FRAME-PAR {physical_cores}-thread"),
        frame_payload_bits,
        secs,
    );

    let t0 = Instant::now();
    let dot = decode_interleave8(&streams, &acts);
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(dot, reference, "INTERLEAVE-8 single-thread dot mismatch");
    report_bytes("I8 INTERLEAVE-8 1-thread", stream_bytes, secs);

    let t0 = Instant::now();
    let dot = decode_interleave8_threaded(&streams, &acts);
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(dot, reference, "INTERLEAVE-8 threaded dot mismatch");
    report_bytes("I8 INTERLEAVE-8 8-thread", stream_bytes, secs);
}
