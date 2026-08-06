# Architecture-level phase design (Phase 5)

*Exploratory. Uses settled circle identities. Does not claim to beat flat
`(5,8)` on dense BitNet. Process language is a design prior to test.*

Code: [`repack/architecture_phase_design.py`](repack/architecture_phase_design.py)  
Results: [`repack/architecture_phase_results.json`](repack/architecture_phase_results.json)  
Prior: [`COLLATZ_BRIDGE.md`](COLLATZ_BRIDGE.md), [`HARMONIC.md`](HARMONIC.md)

## Plan (reviewed)

Phase 4 taught: seed→amplify→harmonize is a faithful rename; drop Collatz
dynamical predictors; packing *does* get algebraic companion compensation.
Next packing move = **choose sizes as seeds**, not more Collatz metaphor.

Three tests:

| ID | Question |
|----|----------|
| T1 | Do surplus / harmonize pads lower multi-linear tax on stock LLM shapes? |
| T2 | Can a whole mode inventory be pushed into surplus cheaply (global budget)? |
| T3 | For squares near ½, is self-complement better than surplus pad? |

## Results (`max_pad=64`)

### T1 — per-shape design (11 LLM / BitNet-like rectangles)

| Strategy | Shapes with tax↓ | Mean Δ best_tax | Mean pad (both modes) |
|----------|-----------------:|----------------:|----------------------:|
| Independent surplus pad | **11 / 11** | **−3041** | ~78 |
| Amplify-surplus joint | **11 / 11** | −3041 | (same family) |
| Harmonize (φᵢ+φⱼ→1) | **9 / 11** | −2527 | ~60 |

Stock widths sit mostly in **deficit / mid / near-half**. Small pads into
surplus crush `best_tax` on every shape tested.

**Caveat:** `llama7b_mlp_down` (4096×11008) — harmonize *raised* tax (+11)
while surplus still won. Harmonize ≠ surplus; axis/asymmetry matters.

**BitNet 2560×2560:** harmonize pad = 0 (already self-complement);
surplus pad (−1262 tax) still wins. See T3.

### T2 — global phase budget (10 unique modes in the suite)

| Metric | Value |
|--------|------:|
| Pad fraction of mode inventory | **0.49%** |
| Modes in surplus band (φ≥0.9) | 0 → **10** |
| Suite Σ best_tax | −33450 |
| Verdict | **`cheap_simultaneous_surplus`** |

Within `max_pad=64`, every unique mode length in the suite can be moved into
the surplus band at once for &lt;0.5% of the mode-length inventory. There is
**no** hard simultaneous-surplus barrier at this scale — the budget tension
is soft (pad trit cost), not a conservation law forbidding full-net surplus.

### T3 — half vs surplus on squares

| Q | side | stay tax | surplus Δ tax | note |
|--:|------|---------:|--------------:|------|
| 640 | mid | 399 | −387 | leave wins |
| **2560** | **near_half** | **1269** | **−1262** | **self-complement, high tax** |
| 4096 | deficit | 4069 | −4028 | leave wins |
| 5120 | deficit | 5079 | −5035 | leave wins |
| 8192 | deficit | 8087 | −8057 | leave wins |

**Critical clarification:** pairwise self-complement (`φ+φ≈1`) is **not**
zero-tax. Zero-tax rows need `{n·α} ≥ 1 − 1/m` (surplus). BitNet’s
`{2560α}≈½` makes squares *complementary under squaring* but leaves
`tax_rows(2560,2560)` large (~1269). The design prior should prefer
**surplus seeds**, not mid-phase self-complements, when the objective is tax.

## What we learned

1. Architecture phase design **works as a prior** on stock widths: pad toward
   surplus is the dominant lever; harmonize is secondary and can fail on
   asymmetric rectangles.
2. Global surplus across a small mode inventory is **cheap** at pad≤64 —
   first quantitative answer to “global phase budget” for this suite.
3. BitNet ½ is an **accidental geometric curiosity**, not a packing optimum.
   Do not design hidden sizes to sit at ½ for packing reasons.

## What this does *not* settle

- Beating `(5,8)` byte density on dense unstructured BitNet (axis/pad wins
  are tax-chart wins; payload format still dominates).
- Characters / cohomology obstruction.
- Dynamic MoE / sparsity phases.
- Coupled hierarchical decode orders.

## How to run

```bash
python3 repack/architecture_phase_design.py selftest
python3 repack/architecture_phase_design.py run --max-pad 64
```
