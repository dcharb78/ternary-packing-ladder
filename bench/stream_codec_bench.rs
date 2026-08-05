// stream_codec_bench.rs — carry-counting range decode vs block baselines
//
// Uniform ternary weights w in {-1, 0, +1} map to trits t = w + 1.
// The stream codec mirrors fmt_stream in ternary_train/pack_ladder.py:
//   * a logical 32-bit range register, initially 2^32;
//   * bounds [0, range/3, 2*(range/3), range], with the last interval
//     absorbing the remainder;
//   * renormalization below 2^24, emitting the high byte of low through the
//     classic delayed-cache + pending-0xFF carry pipeline;
//   * five forced shifts at finish.  Those shifts drain four bytes of low
//     plus the one-byte delayed cache, and the decoder correspondingly
//     preloads five bytes.  There is no separate trailer or fixed header.
//
// Build:
//   ~/.cargo/bin/rustc -O -C target-cpu=native stream_codec_bench.rs

// Reported bpw counts encoded payload bytes only.  The hybrid keeps block
// offsets separately so every independently coded 306-trit block is directly
// addressable; the offset vector is benchmark metadata, not codec payload.
// Its compact terminator chooses a 2^24-aligned point inside the coder's final
// interval, performs the same five carry-propagating shifts, and elides the
// guaranteed trailing zero bytes (EOF reads are zero).  This removes the
// literal fmt_stream flush's fixed small-stream tax without changing decode.

use std::time::Instant;

const NT: usize = 1 << 24;
const HYBRID_BLOCK: usize = 306;

const STREAM_FULL: u64 = 1u64 << 32;
const STREAM_MASK: u64 = STREAM_FULL - 1;
const STREAM_TOP: u64 = 1u64 << 24;
const STREAM_FLUSH_SHIFTS: usize = 5;
const STREAM_INIT_BYTES: usize = 5;

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
        // low is intentionally wider than 32 bits until this shift: bit 32 is
        // the carry into the delayed cache byte.
        let top = self.low >> 24;
        if top != 0xff {
            let carry = (self.low >> 32) as u8;
            self.out.push(self.cache.wrapping_add(carry));
            if self.pending_ff != 0 {
                let fill = 0xffu8.wrapping_add(carry);
                let new_len = self.out.len() + self.pending_ff;
                self.out.resize(new_len, fill);
                self.pending_ff = 0;
            }
            self.cache = (top & 0xff) as u8;
        } else {
            self.pending_ff += 1;
        }

        self.low = (self.low << 8) & STREAM_MASK;
        // During symbol coding this cannot overflow.  The value is immaterial
        // during the five forced flush shifts, so wrapping keeps the operation
        // defined even though the logical Python integer would grow past u64.
        self.range = self.range.wrapping_shl(8);
    }

    #[inline(always)]
    fn encode(&mut self, t: u8) {
        let range3 = self.range / 3;
        let b2 = range3 + range3;
        match t {
            0 => self.range = range3,
            1 => {
                self.low += range3;
                self.range = range3;
            }
            2 => {
                self.low += b2;
                self.range -= b2;
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

    fn finish_compact(mut self) -> Vec<u8> {
        // Since range >= 2^24 here, [low, low + range) always contains the
        // next multiple of 2^24.  Selecting that point makes three suffix
        // bytes zero while preserving the encoded symbol path.
        let low_remainder = self.low & (STREAM_TOP - 1);
        if low_remainder != 0 {
            let delta = STREAM_TOP - low_remainder;
            assert!(delta < self.range);
            self.low += delta;
        }

        for _ in 0..STREAM_FLUSH_SHIFTS {
            self.shift_low();
        }

        let n = self.out.len();
        assert!(
            n >= 3 && self.out[n - 3..].iter().all(|&b| b == 0),
            "aligned range terminator must end in three zero bytes"
        );
        self.out.truncate(n - 3);
        // A fourth zero can also be implicit.  Trim at most this one extra
        // byte so 306-trit payloads retain their expected 61-62-byte span.
        if self.out.last() == Some(&0) {
            self.out.pop();
        }
        self.out
    }
}

#[inline]
fn encode_stream(weights: &[i8]) -> Vec<u8> {
    let mut enc = StreamEncoder::with_capacity(weights.len() / 5 + 8);
    for &w in weights {
        enc.encode((w + 1) as u8);
    }
    enc.finish()
}

#[inline]
fn encode_stream_compact(weights: &[i8]) -> Vec<u8> {
    let mut enc = StreamEncoder::with_capacity(weights.len() / 5 + 8);
    for &w in weights {
        enc.encode((w + 1) as u8);
    }
    enc.finish_compact()
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
        let mut dec = Self {
            data,
            pos: 0,
            low: 0,
            range: STREAM_FULL,
            code: 0,
        };
        for _ in 0..STREAM_INIT_BYTES {
            dec.code = dec.code.wrapping_shl(8) | dec.read_byte() as u32;
        }
        dec
    }

    #[inline(always)]
    fn read_byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }

    #[inline(always)]
    fn decode(&mut self) -> u8 {
        let range3 = self.range / 3;
        let b2 = range3 + range3;
        // wrapping_sub is exactly (code - low) mod 2^32.
        let v = self.code.wrapping_sub(self.low) as u64;

        let t;
        if v < range3 {
            t = 0;
            self.range = range3;
        } else if v < b2 {
            t = 1;
            self.low = self.low.wrapping_add(range3 as u32);
            self.range = range3;
        } else {
            t = 2;
            self.low = self.low.wrapping_add(b2 as u32);
            self.range -= b2;
        }

        while self.range < STREAM_TOP {
            self.low = self.low.wrapping_shl(8);
            self.code = self.code.wrapping_shl(8) | self.read_byte() as u32;
            self.range <<= 8;
        }
        t
    }
}

struct HybridStream {
    data: Vec<u8>,
    offsets: Vec<usize>,
    full_block_min_bytes: usize,
    full_block_max_bytes: usize,
}

fn encode_hybrid(weights: &[i8]) -> HybridStream {
    let nblocks = weights.len().div_ceil(HYBRID_BLOCK);
    let mut data = Vec::with_capacity(weights.len() / 5 + nblocks * 2);
    let mut offsets = Vec::with_capacity(nblocks + 1);
    let mut full_block_min_bytes = usize::MAX;
    let mut full_block_max_bytes = 0usize;
    offsets.push(0);

    for block in weights.chunks(HYBRID_BLOCK) {
        let encoded = encode_stream_compact(block);
        if block.len() == HYBRID_BLOCK {
            full_block_min_bytes = full_block_min_bytes.min(encoded.len());
            full_block_max_bytes = full_block_max_bytes.max(encoded.len());
        }
        data.extend_from_slice(&encoded);
        offsets.push(data.len());
    }

    assert!(
        (61..=62).contains(&full_block_min_bytes)
            && (61..=62).contains(&full_block_max_bytes),
        "306-trit stream blocks should occupy 61-62 bytes; observed {}-{}",
        full_block_min_bytes,
        full_block_max_bytes
    );

    HybridStream {
        data,
        offsets,
        full_block_min_bytes,
        full_block_max_bytes,
    }
}

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

fn report(name: &str, nbytes: usize, secs: f64) {
    let bpw = 8.0 * nbytes as f64 / NT as f64;
    let gtps = NT as f64 / secs / 1e9;
    let gbs = gtps * bpw / 8.0;
    println!("{name}: {gtps:6.2} Gtrit/s   stream {gbs:6.2} GB/s   bpw {bpw:.4}");
}

fn report_encode(name: &str, secs: f64) {
    let gtps = NT as f64 / secs / 1e9;
    println!("{name}: {gtps:6.2} Gtrit/s encode");
}

fn verify_python_known_answer() {
    let weights = [
        -1, 0, 1, 1, 0, -1, 1, -1, 0, -1, 0, 1, 1, 0, -1, 1, -1, 0, -1, 0, 1, 1, 0,
        -1, 1, -1, 0, -1, 0, 1, 1,
    ];
    // Generated by fmt_stream in exploration/ternary_train/pack_ladder.py.
    let expected = [
        0x00, 0x37, 0x08, 0x24, 0x69, 0x16, 0xa0, 0xed, 0xa3, 0x1c, 0x00,
    ];
    let encoded = encode_stream(&weights);
    assert_eq!(encoded, expected, "Python fmt_stream known-answer mismatch");

    let mut dec = StreamDecoder::new(&encoded);
    for &w in &weights {
        assert_eq!(dec.decode() as i8 - 1, w, "known-answer decode mismatch");
    }
}

fn main() {
    verify_python_known_answer();

    // Same xorshift seed, update, symbol mapping, and activation generation as
    // ternary_pack_bench.rs.
    let mut state = 0x243F6A8885A308D3u64;
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let trits: Vec<i8> = (0..NT).map(|_| (rnd() % 3) as i8 - 1).collect();
    let acts: Vec<i8> = (0..NT).map(|_| (rnd() % 255) as i8).collect();
    let reference: i64 = trits
        .iter()
        .zip(&acts)
        .map(|(&w, &a)| w as i64 * a as i64)
        .sum();
    let lut = lut243();

    // Encode each range-coded representation once.  Encoding is reported but
    // is outside every decode timer.
    let t0 = Instant::now();
    let stream = encode_stream(&trits);
    let stream_encode_secs = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let hybrid = encode_hybrid(&trits);
    let hybrid_encode_secs = t0.elapsed().as_secs_f64();

    report_encode("S  range stream      ", stream_encode_secs);
    report_encode("H  range block(306)  ", hybrid_encode_secs);
    println!(
        "H  full-block payload: {}-{} bytes across {} full blocks",
        hybrid.full_block_min_bytes,
        hybrid.full_block_max_bytes,
        NT / HYBRID_BLOCK
    );

    // A: 2-bit baseline, copied from ternary_pack_bench.rs.
    let pa: Vec<u8> = trits
        .chunks(4)
        .map(|c| {
            c.iter().enumerate().fold(0u8, |b, (k, &t)| {
                b | (((t + 1) as u8) << (2 * k))
            })
        })
        .collect();
    let t0 = Instant::now();
    let mut dot = 0i64;
    for (i, &b) in pa.iter().enumerate() {
        let base = i * 4;
        for k in 0..4 {
            let idx = base + k;
            if idx >= NT {
                break;
            }
            dot += (((b >> (2 * k)) & 3) as i64 - 1) * acts[idx] as i64;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(dot, reference, "2-bit dot mismatch");
    report("A  2-bit shift      ", pa.len(), secs);

    // B: (5,8) LUT baseline, copied from ternary_pack_bench.rs.
    let pb: Vec<u8> = trits
        .chunks(5)
        .map(|c| {
            c.iter()
                .rev()
                .fold(0u16, |x, &t| x * 3 + (t + 1) as u16) as u8
        })
        .collect();
    let t0 = Instant::now();
    let mut dot = 0i64;
    for (i, &b) in pb.iter().enumerate() {
        let ws = &lut[b as usize];
        let base = i * 5;
        for k in 0..5 {
            let idx = base + k;
            if idx >= NT {
                break;
            }
            dot += ws[k] as i64 * acts[idx] as i64;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(dot, reference, "(5,8) LUT dot mismatch");
    report("B  rung(5,8) LUT    ", pb.len(), secs);

    // S: sequential range decode with the activation consumed immediately.
    let t0 = Instant::now();
    let mut dot = 0i64;
    let mut dec = StreamDecoder::new(&stream);
    for &a in &acts {
        dot += (dec.decode() as i64 - 1) * a as i64;
    }
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(dot, reference, "range-stream dot mismatch");
    report("S  range stream      ", stream.len(), secs);

    // H: restart the decoder for every independent 306-trit block, retaining
    // the same sequential decode-and-dot access pattern within each block.
    let t0 = Instant::now();
    let mut dot = 0i64;
    for block_idx in 0..hybrid.offsets.len() - 1 {
        let start = hybrid.offsets[block_idx];
        let end = hybrid.offsets[block_idx + 1];
        let base = block_idx * HYBRID_BLOCK;
        let block_end = (base + HYBRID_BLOCK).min(NT);
        let mut dec = StreamDecoder::new(&hybrid.data[start..end]);
        for &a in &acts[base..block_end] {
            dot += (dec.decode() as i64 - 1) * a as i64;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(dot, reference, "hybrid range-stream dot mismatch");
    report("H  range block(306)  ", hybrid.data.len(), secs);
}
