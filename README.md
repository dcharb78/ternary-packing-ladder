# The Ternary Packing Ladder

**Container mathematics and packing formats for ternary ({−1, 0, +1}) weights:
certified constants, exact laws, and a measured demonstration on a public 2B
model.**

Ternary weights carry log₂3 ≈ 1.58496 bits each. This repository shows that
the only block sizes approaching that floor are the continued-fraction
convergent denominators of log₂3 (the *rungs*: 5→8, 41→65, 306→485,
15601→24727 bits), proves the constants in the Lean 4 kernel, derives three
exact structural laws (rung atomicity, chiral split tax, digit-alphabet
nesting), and demonstrates **20.8–22.2% size reduction on the actual
BitNet b1.58 2B ternary payload** with byte-exact reversibility and 840 bytes
of total streaming overhead across 210 tensors.

**Scope, stated up front:** this work addresses packing density and container
mathematics only; it does not claim new end-to-end model quality or
fused-kernel performance.

## The two notes

- [`TERNARY_PACKING_LADDER.md`](TERNARY_PACKING_LADDER.md) — the main note:
  the ladder, the three laws, measured decode Paretos, the training
  experiment, the streaming coder, and the BitNet demonstration.
- [`METHOD_COMPARISON.md`](METHOD_COMPARISON.md) — companion: an honest
  side-by-side with the vendor-class approach (engines, ~1.6–1.7 bpw block
  formats), framing the two as complements.

## What is certified, what is measured, what is not claimed

**Kernel-certified** (Lean 4, `by decide`, no axioms beyond the kernel;
[`lean/Certificates.lean`](lean/Certificates.lean) is standalone — no
imports, no Mathlib):
- The rung inequalities and their tightness (3^Q < 2^P and 2^(P−1) < 3^Q for
  every ladder row), and the zero-tax assembly constants of Law B.

**Exact-verified** (byte-exact round-trip asserted before any number is
reported):
- Every packing format on every tensor of the BitNet payload
  (2,084,044,800 weights), and every benchmark's decoded output against a
  reference dot product.

**Measured** (with harness stated in context):
- Decode throughputs (single-file Rust, Apple M5 Max, core counts quoted
  per row); the training experiment (MLX, enwik8, preregistered thresholds);
  the BitNet alphabet census and per-module decomposition.

**Not claimed:**
- Engine-level speed, fused-GPU-kernel performance, product-scale capability
  retention, or superiority over any shipping system. Vendor numbers cited
  in the comparison are treated as unverified claims, never as false.

## Contents

```
lean/Certificates.lean      standalone kernel certificates (Lean 4, no deps)
bench/ternary_pack_bench.rs block-format decode Pareto (std-only Rust)
bench/stream_codec_bench2.rs concurrent-dimension benchmarks (frame/interleave)
repack/pack_ladder.py       all container formats + exact round-trip selftests
repack/adaptive_entropy.py  per-tensor static range coder (exact integers)
repack/repack_bitnet.py     the BitNet b1.58 2B repack (downloads not included)
repack/train_arms.py        the preregistered training experiment (MLX)
repack/analyze_verdicts.py  verdict computation for the training experiment
```

## Reproducing

- **Certificates**: `lean Certificates.lean` under any recent Lean 4
  toolchain (seconds; no packages).
- **Benchmarks**: `rustc -O -C target-cpu=native bench/<file>.rs && ./<bin>`
  (std only).
- **Formats**: `python3 repack/pack_ladder.py selftest` (numpy only; 187
  exact round-trip cases + the streaming coder's preregistered property
  checks).
- **BitNet repack**: download `microsoft/bitnet-b1.58-2B-4T`'s
  `model.safetensors` (~1.18 GB) into `repack/data/bitnet/`, then
  `python3 repack/repack_bitnet.py`. Every tensor is round-tripped exactly
  through every format before its size counts.
- **Training experiment**: enwik8 into `repack/data/`, then
  `python3 repack/train_arms.py --arm ternary --dim 384 --steps 3000`
  (MLX, Apple Silicon).

## Provenance

The constants come from a larger project on the 3n+1 map, where the same
2-vs-3 phase governs orbit descent; the rung inequalities double as that
project's kernel-certified descent certificates. Development followed a
preregistration-and-public-retraction discipline; the notes retain grade
labels throughout.
