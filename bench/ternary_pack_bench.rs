// ternary_pack_bench.rs — the packing ladder vs decode cost, v2 (nested-alphabet decode)
// Formats (ternary weight w in {-1,0,+1}, activations i8, dot must match reference exactly):
//   A  2-bit           4 trits/byte    2.0000 bpw  shift/mask
//   B  rung (5,8)      5 trits/byte    1.6000 bpw  one 243-LUT per byte (prior art class)
//   C  rung (41,65)    trit-serial     1.5854 bpw  41 div-3 per block   (naive)
//   C2 rung (41,65)    digit decode    1.5854 bpw  8 div-243 + LUT      (nested alphabet)
//   D  rung (306,485)  trit-serial     1.5850 bpw  306 multi-limb div-3 (naive)
//   D2 rung (306,485)  digit decode    1.5850 bpw  61 multi-limb div-243 + LUT
//   E  486-frame       7x(41,65)+(19,31) 1.5882 bpw all-u128 digit decode (1-bit split tax)
use std::time::Instant;

const NT: usize = 1 << 24;

fn lut243() -> [[i8; 5]; 256] {
    let mut lut = [[0i8; 5]; 256];
    for v in 0..243usize { let mut x = v; for k in 0..5 { lut[v][k] = (x % 3) as i8 - 1; x /= 3; } }
    lut
}

fn report(name: &str, bits_per_trit_num: usize, bits_per_trit_den: usize, secs: f64) {
    let bpw = bits_per_trit_num as f64 / bits_per_trit_den as f64;
    let gtps = NT as f64 / secs / 1e9;
    let gbs = gtps * bpw / 8.0; // packed-stream GB/s at this bpw
    println!("{name}: {gtps:6.2} Gtrit/s   stream {gbs:6.2} GB/s   bpw {bpw:.4}");
}

fn main() {
    let mut state = 0x243F6A8885A308D3u64;
    let mut rnd = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
    let trits: Vec<i8> = (0..NT).map(|_| (rnd() % 3) as i8 - 1).collect();
    let acts: Vec<i8> = (0..NT).map(|_| (rnd() % 255) as i8).collect();
    let reference: i64 = trits.iter().zip(&acts).map(|(&w, &a)| (w as i64) * (a as i64)).sum();
    let lut = lut243();

    // A: 2-bit
    let pa: Vec<u8> = trits.chunks(4).map(|c| c.iter().enumerate().fold(0u8, |b,(k,&t)| b | (((t+1) as u8) << (2*k)))).collect();
    let t0 = Instant::now(); let mut dot = 0i64;
    for (i,&b) in pa.iter().enumerate() { let base = i*4;
        for k in 0..4 { let idx = base+k; if idx >= NT { break; }
            dot += (((b >> (2*k)) & 3) as i64 - 1) * acts[idx] as i64; } }
    let d = t0.elapsed().as_secs_f64(); assert_eq!(dot, reference); report("A  2-bit shift     ", 2, 1, d);

    // B: (5,8) LUT
    let pb: Vec<u8> = trits.chunks(5).map(|c| c.iter().rev().fold(0u16, |x,&t| x*3 + (t+1) as u16) as u8).collect();
    let t0 = Instant::now(); let mut dot = 0i64;
    for (i,&b) in pb.iter().enumerate() { let ws = &lut[b as usize]; let base = i*5;
        for k in 0..5 { let idx = base+k; if idx >= NT { break; }
            dot += (ws[k] as i64) * (acts[idx] as i64); } }
    let d = t0.elapsed().as_secs_f64(); assert_eq!(dot, reference); report("B  rung(5,8) LUT   ", 8, 5, d);

    // C blocks
    let pc: Vec<u128> = trits.chunks(41).map(|c| c.iter().rev().fold(0u128, |x,&t| x*3 + (t+1) as u128)).collect();
    // C: trit-serial
    let t0 = Instant::now(); let mut dot = 0i64;
    for (i,&blk) in pc.iter().enumerate() { let mut x = blk; let base = i*41;
        for k in 0..41 { let idx = base+k; if idx >= NT { break; }
            dot += ((x % 3) as i64 - 1) * acts[idx] as i64; x /= 3; } }
    let d = t0.elapsed().as_secs_f64(); assert_eq!(dot, reference); report("C  rung(41,65) serial", 65, 41, d);
    // C2: digit decode — 8 base-243 digits + 1 final trit
    let t0 = Instant::now(); let mut dot = 0i64;
    for (i,&blk) in pc.iter().enumerate() { let mut x = blk; let base = i*41;
        if base + 41 <= NT {
            for dg in 0..8 { let ws = &lut[(x % 243) as usize]; x /= 243; let o = base + dg*5;
                for k in 0..5 { dot += (ws[k] as i64) * (acts[o+k] as i64); } }
            dot += (x as i64 - 1) * acts[base+40] as i64;
        } else { for k in 0..41 { let idx = base+k; if idx >= NT { break; }
            dot += ((x % 3) as i64 - 1) * acts[idx] as i64; x /= 3; } } }
    let d = t0.elapsed().as_secs_f64(); assert_eq!(dot, reference); report("C2 rung(41,65) digit ", 65, 41, d);

    // D blocks: 306 trits in 8x64 limbs
    let pd: Vec<[u64; 8]> = trits.chunks(306).map(|c| { let mut l = [0u64; 8];
        for &t in c.iter().rev() { let mut carry = (t+1) as u64 as u128;
            for x in l.iter_mut() { let v = (*x as u128)*3 + carry; *x = v as u64; carry = v >> 64; } } l }).collect();
    let divmod = |l: &mut [u64; 8], m: u64| -> u64 { let mut rem = 0u64;
        for x in l.iter_mut().rev() { let v = ((rem as u128) << 64) | (*x as u128); *x = (v / m as u128) as u64; rem = (v % m as u128) as u64; } rem };
    // D: trit-serial
    let t0 = Instant::now(); let mut dot = 0i64;
    for (i,blk) in pd.iter().enumerate() { let mut l = *blk; let base = i*306;
        for k in 0..306 { let idx = base+k; if idx >= NT { break; }
            let r = divmod(&mut l, 3); dot += (r as i64 - 1) * acts[idx] as i64; } }
    let d = t0.elapsed().as_secs_f64(); assert_eq!(dot, reference); report("D  rung(306,485) serial", 485, 306, d);
    // D2: digit decode — 61 base-243 digits + 1 final trit
    let t0 = Instant::now(); let mut dot = 0i64;
    for (i,blk) in pd.iter().enumerate() { let mut l = *blk; let base = i*306;
        if base + 306 <= NT {
            for dg in 0..61 { let r = divmod(&mut l, 243); let ws = &lut[r as usize]; let o = base + dg*5;
                for k in 0..5 { dot += (ws[k] as i64) * (acts[o+k] as i64); } }
            dot += (l[0] as i64 - 1) * acts[base+305] as i64;
        } else { for k in 0..306 { let idx = base+k; if idx >= NT { break; }
            let r = divmod(&mut l, 3); dot += (r as i64 - 1) * acts[idx] as i64; } } }
    let d = t0.elapsed().as_secs_f64(); assert_eq!(dot, reference); report("D2 rung(306,485) digit ", 485, 306, d);

    // E: 486-frame = 7x(41,65) + (19,31); all-u128 digit decode; 1-bit split tax over 306 trits
    let nfr = NT / 306; // full frames only; tail handled scalar from trits directly in timing-neutral way
    let mut pe: Vec<([u128; 7], u32)> = Vec::with_capacity(nfr);
    for f in 0..nfr { let c = &trits[f*306..(f+1)*306];
        let mut blocks = [0u128; 7];
        for j in 0..7 { blocks[j] = c[j*41..(j+1)*41].iter().rev().fold(0u128, |x,&t| x*3 + (t+1) as u128); }
        let tail = c[287..306].iter().rev().fold(0u32, |x,&t| x*3 + (t+1) as u32);
        pe.push((blocks, tail)); }
    let t0 = Instant::now(); let mut dot = 0i64;
    for (f,(blocks,tail)) in pe.iter().enumerate() { let fb = f*306;
        for j in 0..7 { let mut x = blocks[j]; let o0 = fb + j*41;
            for dg in 0..8 { let ws = &lut[(x % 243) as usize]; x /= 243; let o = o0 + dg*5;
                for k in 0..5 { dot += (ws[k] as i64) * (acts[o+k] as i64); } }
            dot += (x as i64 - 1) * acts[o0+40] as i64; }
        let mut x = *tail; let o0 = fb + 287;
        for dg in 0..3 { let ws = &lut[(x % 243) as usize]; x /= 243; let o = o0 + dg*5;
            for k in 0..5 { dot += (ws[k] as i64) * (acts[o+k] as i64); } }
        for k in 15..19 { dot += ((x % 3) as i64 - 1) * acts[o0+k] as i64; x /= 3; } }
    for idx in nfr*306..NT { dot += (trits[idx] as i64) * acts[idx] as i64; }
    let d = t0.elapsed().as_secs_f64(); assert_eq!(dot, reference); report("E  486-frame digit  ", 486, 306, d);
}
