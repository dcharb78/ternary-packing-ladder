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

## The notes

- [`TERNARY_PACKING_LADDER.md`](TERNARY_PACKING_LADDER.md) — the main note:
  the ladder, the three laws, measured decode Paretos, the training
  experiment, the streaming coder, and the BitNet demonstration.
- [`METHOD_COMPARISON.md`](METHOD_COMPARISON.md) — companion: an honest
  side-by-side with the vendor-class approach (engines, ~1.6–1.7 bpw block
  formats), framing the two as complements.
- [`EXTENSIONS.md`](EXTENSIONS.md) — Kronecker / tax-graph / multi-linear
  extensions (Phase 1–2): hierarchical D3 decode, Law B frame catalog,
  tensor-axis tax forms, measured verdicts.
- [`SCALE_PROBE.md`](SCALE_PROBE.md) — targeted LLM-shape probes (axis
  choice, Kronecker factors, tax-0 tiling grammar) and creative connections.
  Not a blind larger flat stream.
- [`CHECKPOINT_PROBE.md`](CHECKPOINT_PROBE.md) — same levers on the real
  BitNet b1.58 2B-4T payload (axis, pad-to-tax0, structure scan).
- [`HARMONIC.md`](HARMONIC.md) — log-domain / fractional-part geometry of the
  tax (`{Q log₂ 3}`); additive ledgers as the flat chart.

## Extensions at a glance

| Phase | Claim tested | Headline result |
|-------|--------------|-----------------|
| 1-D Law C hierarchical (D3) | ÷`3^41`→C2 vs flat ÷243 | ~1.5× D2 (0.36 vs 0.24 Gtrit/s); 486-frame still Pareto |
| 1-D Law B tax graph | Enumerate tax 0/1 assemblies | 665 + 65 novel tax-0 frames; catalog JSON |
| Multi-linear tax form | `tax_rows` / `tax_cols` on modes | 1-D tax graph = rank-1 slice; axis choice is free |
| Scale probe (LLM shapes) | Axis choice on 7B–34B rectangles | e.g. 11008×4096 saves 7936 bits by packing cols |
| Kronecker-factor pack | Pack `A`,`B` vs flatten `A⊗B` | ~100× when structure exists (toy→scale cases) |

Hard rule unchanged: size/tax/pack/unpack verdicts are **exact integers**.

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
- Extension selftests (tax graph, frames, hierarchical digits, mode pack,
  scale-probe unit checks).

**Measured** (with harness stated in context):
- Decode throughputs (single-file Rust, Apple Silicon, core counts quoted
  per row); the training experiment (MLX, enwik8, preregistered thresholds);
  the BitNet alphabet census and per-module decomposition; extension and
  scale-probe ledgers in `EXTENSIONS.md` / `SCALE_PROBE.md`.

**Not claimed:**
- Engine-level speed, fused-GPU-kernel performance, product-scale capability
  retention, or superiority over any shipping system. Vendor numbers cited
  in the comparison are treated as unverified claims, never as false.

## Contents

```
lean/Certificates.lean         standalone kernel certificates (Lean 4, no deps)
bench/ternary_pack_bench.rs    block-format decode Pareto (+ D3 hierarchical)
bench/stream_codec_bench*.rs   stream / frame-parallel benches
repack/pack_ladder.py          container formats + exact round-trip selftests
repack/adaptive_entropy.py     per-tensor static range coder
repack/repack_bitnet.py        BitNet b1.58 2B repack (checkpoint not included)
repack/train_arms.py           preregistered training experiment (MLX)
repack/tax_graph.py            Law B tax-graph enumerator
repack/frame_formats.py        486- and 665-frame pack/unpack
repack/hierarchical_digits.py  Law C nested / Kronecker digit unpack
repack/recursive_pack.py       recursive P_n blocked composition
repack/collatz_schedule.py     Collatz-adaptive frame schedule
repack/pre_transform_probe.py  structured Hadamard→ternary probe
repack/tax_tensor.py           multi-linear tax form on mode sizes
repack/mode_pack.py            mode-wise pack + phase offsets
repack/kronecker_tensor_pack.py Kronecker-factor pack ledgers
repack/tensor_hierarchical.py  per-row hierarchical decode
repack/mode_schedule.py        per-mode phase + axis choice
repack/scale_probe.py          targeted LLM-shape axis/Kronecker/tax0 probe
repack/pad_to_tax0.py          pad mode length to tax-0 frame multiples
repack/checkpoint_axis_probe.py real BitNet axis/pad/structure probe
repack/harmonic_tax.py         fractional-part / harmonic tax dictionary
HARMONIC.md                    harmonic geometry note
CHECKPOINT_PROBE.md            real-checkpoint evaluation
EXTENSIONS.md / SCALE_PROBE.md evaluation notes
```

## Reproducing

- **Certificates**: `lean Certificates.lean` under any recent Lean 4
  toolchain (seconds; no packages).
- **Benchmarks**: `rustc -O -C target-cpu=native bench/<file>.rs && ./<bin>`
  (std only). Look for the **D3** row in `ternary_pack_bench`.
- **Formats**: `python3 repack/pack_ladder.py selftest` (numpy only).
- **Extensions**: from `repack/`,
  `python3 tax_graph.py catalog && python3 tax_graph.py selftest` and
  likewise `hierarchical_digits.py`, `frame_formats.py`, `tax_tensor.py`,
  `mode_pack.py`, `kronecker_tensor_pack.py`, `tensor_hierarchical.py`,
  `mode_schedule.py`, `recursive_pack.py`, `collatz_schedule.py`,
  `pre_transform_probe.py`.
- **Scale probe**: `python3 repack/scale_probe.py run`
- **Pad-to-tax0**: `python3 repack/pad_to_tax0.py run`
- **Checkpoint probe**: download BitNet `model.safetensors` into
  `repack/data/bitnet/`, then
  `python3 repack/checkpoint_axis_probe.py --rt-sample 5`
- **Harmonic dictionary**: `python3 repack/harmonic_tax.py selftest` /
  `run`
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

## License

Dedicated to the public domain under [CC0 1.0 Universal](LICENSE). Use
freely, no attribution required.
