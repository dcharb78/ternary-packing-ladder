#!/usr/bin/env python3
"""Repack the public BitNet b1.58 2B-4T checkpoint through our container
formats. Exact-integer verdicts; per-tensor round-trip asserts.

Their format: U8 tensors [out/4, in], 4 trits/byte 2-bit packed, per-tensor
scalar scale. We unpack to the trit sequence (alphabet validated: every
2-bit field must be in {0,1,2}; a 3 anywhere aborts), then measure:
  their-2bit | (5,8) 5-per-byte | uniform-ternary ideal (bit_length) |
  stream (real carry-counting coder, exact round-trip)
Sizes for our formats are measured on real packed bytes, round-tripped
exactly. Their bit order within a byte does not affect any size or the
round-trip claim (which is about our containers on the trit content).
"""
import json, math, struct, sys, time
import numpy as np

sys.path.insert(0, ".")
from pack_ladder import pack_5_8, unpack_5_8, stream_pack, stream_unpack
import adaptive_entropy as ae

PATH = "data/bitnet/model.safetensors"

def load_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode())
    return hdr, 8 + n

def main():
    hdr, data0 = load_header(PATH)
    names = sorted(k for k in hdr if k != "__metadata__"
                   and hdr[k]["dtype"] == "U8" and k.endswith(".weight"))
    print(f"packed ternary tensors: {len(names)}")
    mm = np.memmap(PATH, dtype=np.uint8, mode="r")
    tot = {"trits": 0, "their2bit": 0, "b58": 0, "ideal": 0, "stream": 0, "adaptive": 0}
    counts_tot = np.zeros(3, dtype=np.int64)
    t0 = time.time()
    lut = np.zeros((256, 4), dtype=np.int8)
    for b in range(256):
        for k in range(4):
            lut[b, k] = (b >> (2 * k)) & 3
    for idx, name in enumerate(names):
        beg, end = hdr[name]["data_offsets"]
        raw = np.asarray(mm[data0 + beg : data0 + end])
        fields = lut[raw].reshape(-1)          # 4 fields per byte, LSB-first
        assert fields.max() <= 2, f"non-ternary field in {name}: their packing convention differs"
        trits = (fields.astype(np.int8) - 1)   # {-1,0,1}
        n = trits.size
        c = np.array([np.count_nonzero(trits == v) for v in (-1, 0, 1)], dtype=np.int64)
        counts_tot += c
        # (5,8)
        b58 = pack_5_8(trits)
        assert np.array_equal(unpack_5_8(b58, n), trits), name
        # stream (real coder, exact round-trip)
        st = stream_pack(trits)
        assert np.array_equal(stream_unpack(st, n), trits), name
        ad, _ = ae.static_roundtrip((trits + 1).astype(np.int64))
        ad = ad + b"\x00" * 12  # honest overhead: 3x u32 counts header per tensor
        ideal = -(-(3 ** n).bit_length() // 8)
        tot["trits"] += n
        tot["their2bit"] += raw.size
        tot["b58"] += len(b58)
        tot["ideal"] += ideal
        tot["stream"] += len(st)
        tot["adaptive"] += len(ad)
        if idx % 21 == 0:
            print(f"  [{idx+1}/{len(names)}] {name.split('model.layers.')[-1]:<40}"
                  f" n={n:>10,}  ({time.time()-t0:.0f}s)", flush=True)
    n = tot["trits"]
    p = counts_tot / n
    H = -sum(x * math.log2(x) for x in p if x > 0)
    print(f"\n=== BitNet b1.58 2B-4T: {n:,} ternary weights ===")
    print(f"alphabet: p(-1)={p[0]:.4f}  p(0)={p[1]:.4f}  p(+1)={p[2]:.4f}   H={H:.4f} bits/w (log2 3 = {math.log2(3):.4f})")
    rows = [("their 2-bit (as shipped)", tot["their2bit"], 4 * tot["their2bit"] * 2 / (8 * 0.5)),
            ]
    print(f"{'format':<28}{'bytes':>16}{'bpw':>9}{'vs shipped':>12}")
    base = tot["their2bit"]
    for label, sz in [("their 2-bit (as shipped)", tot["their2bit"]),
                      ("rung (5,8) 5-per-byte", tot["b58"]),
                      ("uniform-ternary ideal", tot["ideal"]),
                      ("stream (real, round-tripped)", tot["stream"]),
                      ("adaptive (real, round-tripped)", tot["adaptive"])]:
        print(f"{label:<28}{sz:>16,}{sz*8/n:>9.4f}{sz/base:>12.4f}")
    print(f"\nstream slack vs ideal: {tot['stream'] - tot['ideal']:,} bytes over {len(names)} tensors")
    print(f"savings vs shipped 2-bit: {(base - tot['stream'])/1e6:.1f} MB")
    print("REPACK COMPLETE — all tensors round-tripped exactly")

if __name__ == "__main__":
    main()
