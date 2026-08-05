#!/usr/bin/env python3
"""Verdicts for the ternary iso-bits test — thresholds fixed in prereg
(docs/parallel/PREREG_clock_binding_rows.md, 2026-08-06 entry) BEFORE any run.

P1: D = bpc(ter384) - bpc(fp120); D<=+0.03 SUPPORTED, D>=+0.15 REFUTED, else
    INCONCLUSIVE. Same rule for bin384 vs fp96 (exact iso-bits pair).
P2: which accounting (ideal log2(3) vs realized entropy) places ter384 closer
    to the fp bpc-vs-log2(bits) piecewise-linear interpolation. No threshold.
Runs with steps < --min-steps are PILOT and never verdict-eligible.
"""
import json, math, argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--min-steps", type=int, default=3000, help="verdict eligibility floor (prereg: 3000)")
ap.add_argument("--runs", default="runs.jsonl")
a = ap.parse_args()

rows = []
with open(a.runs) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

# latest run per (arm, dim) at the highest step count
best = {}
for r in rows:
    k = (r["arm"], r["dim"])
    if k not in best or (r["steps"], rows.index(r)) >= (best[k]["steps"], rows.index(best[k])):
        best[k] = r

print(f"{'arm':8} {'dim':>4} {'steps':>6} {'params':>10} {'Mbits_ideal':>12} {'Mbits_real':>11} {'val_bpc':>8} {'tok/s':>8}")
for (arm, dim), r in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    print(f"{arm:8} {dim:>4} {r['steps']:>6} {r['params_total']:>10} "
          f"{r['bits_ideal']/1e6:>12.3f} {r['bits_realized']/1e6:>11.3f} "
          f"{r['val_bpc']:>8.4f} {r.get('tok_per_s', float('nan')):>8.0f}")

def get(arm, dim):
    r = best.get((arm, dim))
    if r is None:
        return None
    return r

def verdict(d):
    if d <= 0.03: return "SUPPORTED"
    if d >= 0.15: return "REFUTED"
    return "INCONCLUSIVE"

print()
eligible = all(
    (r := get(arm, dim)) is not None and r["steps"] >= a.min_steps
    for arm, dim in [("ternary", 384), ("fp", 120), ("binary", 384), ("fp", 96)]
)
tag = "" if eligible else f"  [PILOT ONLY — runs below {a.min_steps} steps; NOT verdict-eligible]"

t, f120 = get("ternary", 384), get("fp", 120)
if t and f120:
    D = t["val_bpc"] - f120["val_bpc"]
    print(f"P1 ternary: bpc(ter384)={t['val_bpc']:.4f}  bpc(fp120)={f120['val_bpc']:.4f}  "
          f"D={D:+.4f}  -> {verdict(D)}{tag}")
b, f96 = get("binary", 384), get("fp", 96)
if b and f96:
    Db = b["val_bpc"] - f96["val_bpc"]
    print(f"P1 binary : bpc(bin384)={b['val_bpc']:.4f}  bpc(fp96) ={f96['val_bpc']:.4f}  "
          f"D={Db:+.4f}  -> {verdict(Db)}{tag}")

# P2: fp interpolation curve in (log2 bits, bpc)
fps = sorted((r for (arm, _), r in best.items() if arm == "fp"), key=lambda r: r["bits_ideal"])
if len(fps) >= 2 and t:
    xs = [math.log2(r["bits_ideal"]) for r in fps]
    ys = [r["val_bpc"] for r in fps]
    def interp(x):
        if x <= xs[0]: i = 0
        elif x >= xs[-1]: i = len(xs) - 2
        else: i = max(j for j in range(len(xs) - 1) if xs[j] <= x)
        w = (x - xs[i]) / (xs[i + 1] - xs[i])
        return ys[i] + w * (ys[i + 1] - ys[i])
    for label, bits in [("ideal   log2(3)/w", t["bits_ideal"]), ("realized entropy ", t["bits_realized"])]:
        pred = interp(math.log2(bits))
        print(f"P2 {label}: bits={bits/1e6:.3f}M  fp-curve predicts bpc={pred:.4f}  "
              f"actual={t['val_bpc']:.4f}  gap={t['val_bpc']-pred:+.4f}")
    if t["params_quantized"]:
        H = sum(w["n"] * w["entropy_bits"] for w in t["weight_stats"]) / t["params_quantized"]
        p0 = sum(w["n"] * w["p_0"] for w in t["weight_stats"]) / t["params_quantized"]
        print(f"P2 realized mean entropy H={H:.4f} bits/weight (log2 3={math.log2(3):.4f});  mean p_0={p0:.4f}")

print()
print("P3: run  python3 pack_ladder.py pack ckpt_ternary_384.npz packed/ter384  (packer asserts exactness itself)"
      if os.path.exists("ckpt_ternary_384.npz") else "P3: no ckpt_ternary_384.npz yet")
