# Harmonic geometry of the packing tax

*Additive block taxes are a flat chart of a log-domain / circle phenomenon.
Primary code path: `{Q·α}` via `repack/harmonic_tax.py` (wired into
`tax_tensor`, `scale_probe.axis_bits`, pad design).*

## Review in context (what we learned)

| Layer | Finding |
|-------|---------|
| 1-D ladder | Only CF rungs of α=log₂3 approach the trit floor; Laws A/B/C exact |
| Tax graph | Zero-tax assemblies are systematic; 665 + novel deficit cancellations |
| D3 / frames | Hierarchical nesting helps modestly; 486-frame remains speed knee |
| Multi-linear | `tax_rows`/`tax_cols` = torus holonomy; axis choice is free |
| BitNet-2B | Axis ~169 KB; pad-to-tax0 ~26 MB vs fiber-41; still loses to `(5,8)`; no Kronecker tiles |
| **Harmonic** | Additive tax = flat chart of `{Q·α}` on the circle |

## Exact dictionary (α = log₂ 3)

| Additive chart | Harmonic chart (primary) |
|----------------|--------------------------|
| `bits(Q)` | `⌊Q·α⌋ + 1` |
| `tax_rows(m,n)` | `m − 1 − ⌊m·{n·α}⌋` |
| `tax_cols(m,n)` | `n − 1 − ⌊n·{m·α}⌋` |
| zero-tax rows | `{n·α} ≥ 1 − 1/m` |
| axis holonomy | `m·bits(n) − n·bits(m)` (= tax_rows − tax_cols) |

**Practical consequence:** layout/tax passes use `{Q·α}` (Decimal) and never
form `3^{MN}`. Same integers; theory stays on the circle. Additive
`(3^Q).bit_length()` remains the ledger cross-check for modest Q.

## Law B / energy / MRA (verified)

- **Surplus rungs** `{5,41,306}·α → 0.925, 0.983, 0.999` — phase maxima near 1;
  distance to 1 shrinks up the ladder (MRA scales).
- **Energy** `E_rows = tax_rows`: minima are exactly `m = 1..max_m` with tax 0;
  phase-peak scan recovers rungs 5, 41, 306.
- **Law C nesting** `41=8·5+1`, `306=7·41+19` = parent-alphabet expansion at
  successive surplus scales (harmonic multiresolution sketch).

## Precise decimals as circle coordinates

The values are not merely “near 0/1.” **Distance and side** control
multi-dimensional interaction under `m · {n·α}`:

| Q | `{Qα}` | Side | Orbit behaviour |
|---|--------|------|-----------------|
| 5 | 0.9248 | →1 from below | tax_rows=0 for m=1..13; residual drifts down |
| 41 | 0.9835 | →1 from below | tax_rows=0 for m=1..60; slow drift from 1 |
| 306 | 0.9985 | →1 from below | tax_rows=0 for hundreds of m |
| 19 | 0.1143 | mid | tax grows ~m−1 until overflow at m=9 |
| 53 | 0.0030 | →0 | floor stays 0; tax = m−1 (hungry) |
| 665 | 6.3e-5 | →0 | extreme deficit |

**Complementary phases** (φ+ψ≈1) explain Law B cancellations:

- `53+306` → sum ≈ **1.0015** (665 = 2×306+53 sits on this pair)
- `41+19` → sum ≈ 1.098 (486-frame grammar)
- `306+665` → sum ≈ 0.999

Axis choice: one orientation multiplies the large φ, the other the small —
`⌊m·φ_n⌋` vs `⌊n·φ_m⌋` differ; that is the holonomy.

```bash
python3 repack/harmonic_orbit.py run   # orbits + complements + BitNet
```

## Multidimensional regime (phase cloud, triples, pad tracker)

```bash
python3 repack/harmonic_multi.py   # needs BitNet ckpt for cloud; else --skip-ckpt
```

### 1. BitNet-2B phase cloud

Only **three** unique mode lengths: `640, 2560, 6912` (210 tensors).

| shape | count | `{mα}+{nα}` dist to 1 | notes |
|-------|------:|----------------------:|-------|
| 2560×2560 | 60 | **≈0.008** | accidental self-complement (`{2560α}≈0.504`) |
| 640×2560 | 60 | ≈0.120 | mid + half |
| 6912×2560 | 90 | ≈0.235 | both mid |

So **60/210 tensors are already near-complementary** without padding — the
square projections sit at φ≈½. That is an architectural accident with
harmonic consequences, not a packing format.

Accidental mode-pair complements among unique lengths are weak except
self-pairs at 2560.

### 2. Triples and 3-mode fiber tax

3-mode tax (circle, no `3^{mnp}`):

`tax_orient_i = tax_rows(d_i, ∏_{j≠i} d_j) = d_i − 1 − ⌊d_i · {(∏_{j≠i} d_j)·α}⌋`

Best of three orientations. Triple search finds low pairwise-sum scores
(e.g. `[306,665,1024]`); closed chains with all three pairs ≈1 force
phases ≈½ (consistent with the 2560 self-complement story).

### 3. Multi-mode pad tracker

Greedy pads toward pairwise φᵢ+φⱼ≈1:

| dims | padded | total pad | score→ |
|------|--------|----------:|--------|
| 2560×6912 | 2598×6912 | 38 | 0.235→0.007 |
| 640×2560 | 659×2560 | 19 | 0.120→0.006 |
| 306×53×41 | 312×59×47 | 18 | ~1.00→0.03 |

Small pads can move whole rectangles onto near-complement configurations —
the multi-D analogue of surplus-phase pad.

### Still open

Multi-track map (keep dimensions separate; static frontier = 665-flat):
[`DIMENSIONS.md`](DIMENSIONS.md). Circle/phase here is the **coordinate
system**; dynamical / multi-scale / training questions live on other tracks.

- Global phase budget on a *real* full net (suite-level answer in
  [`ARCHITECTURE_PHASE.md`](ARCHITECTURE_PHASE.md): cheap simultaneous
  surplus at pad≤64 for LLM_SHAPES modes — not yet BitNet-full inventory)
- Dynamic phases under MoE / sparsity
- Multi-D Beatty partitions as tilings
- Character/cohomology obstruction for impossible zero-tax tuples
  (**exploratory** algebraic probe — not a certified next step)
- Coupled hierarchical decode contraction orders
- Architecture-level phase design — **first tests done**:
  [`ARCHITECTURE_PHASE.md`](ARCHITECTURE_PHASE.md);
  `python3 repack/architecture_phase_design.py run`
  (surplus pad dominates; BitNet ½ ≠ low tax)
- Geometry lab (reshape / fiber-pad / associator) —
  [`GEOMETRY_LAB.md`](GEOMETRY_LAB.md);
  `python3 repack/geometry_lab.py run`
- Packing stack measurement + failure analysis —
  [`PACKING_STACK.md`](PACKING_STACK.md);
  `python3 repack/packing_stack.py all`
- Change-at-41 probe (CF rung vs twin midpoint) —
  [`CHANGE_AT_41.md`](CHANGE_AT_41.md);
  `python3 repack/change_at_41.py run`
- Organizing principles (operators validated on bytes/tax) —
  [`ORGANIZING_PRINCIPLES.md`](ORGANIZING_PRINCIPLES.md);
  `python3 repack/organizing_principles.py run`
- Practical three-ledger packer + BitNet measurement —
  [`PACKING_PIPELINE.md`](PACKING_PIPELINE.md);
  `python3 repack/ledger_packer.py all`
  (`--hybrid` / `--hybrid665` for prefix density without exact ÷306/÷665)
- Packet/seam framing applicability (pattern/recursion/nesting/hierarchy) —
  [`PACKET_SEAM.md`](PACKET_SEAM.md);
  `python3 repack/packet_seam_probe.py run`
- Better density / hybrid-306 vs flat —
  [`BETTER_DENSITY.md`](BETTER_DENSITY.md);
  `python3 repack/better_density.py run`
- Pattern of patterns / 665-flat frontier —
  [`PATTERN_OF_PATTERNS.md`](PATTERN_OF_PATTERNS.md);
  `PYTHONPATH=repack python3 repack/pattern_of_patterns.py run`
- Large-scale synthetic 7B–70B —
  [`LARGE_SCALE.md`](LARGE_SCALE.md);
  `PYTHONPATH=repack python3 repack/large_scale_probe.py run`

## Unseen connections (checklist)

1. **Collatz / stream schedule** `C_i = bits(i) = ⌊i·α⌋+1` — same circle object;
   “descending” tracks how `{i·α}` moves, not a separate gadget.
2. **Tax graph search** = hunting Q with high `{Q·α}` and compositions whose
   floored masses cancel (energy minima).
3. **Pad design** = move L toward surplus phase (`pad_toward_surplus_phase`);
   e.g. 2560→2583 (+23) lifts `{·α}` 0.50→0.96 and `max_m` tax0 2→23.
4. **Stream vs block tension** unchanged: blocks follow holonomy; streams hate
   per-fiber slack — orthogonal to the chart vs geometry question.
5. **Beatty sequence** `⌊(n+1)α⌋−⌊n·α⌋` is the per-trit bit increment; rungs
   are where the Beatty discrepancy is minimal (phase near 1).
6. **Characters (open, exploratory):** tax as nontriviality of a map
   `⟨2,3⟩→S¹` — not a theorem; energy/MRA are the concrete stand-ins shipped
   here. Process-language probes
   (`repack/process_language_probes.py`) test whether seed/amplify/harmonize
   is faithful renaming of the circle dictionary (hypothesis, not claim).
7. **Factor tree = decode tree** still the structure win when Kronecker exists;
   harmonic chart does not create factors in dense BitNet.

## What is not claimed

- Beating `(5,8)` on unstructured BitNet via harmonic language alone.
- That raw `τ=m{nα}−n{mα}` replaces floored tax (floors are the quantization).

```bash
python3 repack/harmonic_tax.py selftest   # dictionary
python3 repack/harmonic_energy.py run     # E minima + phase peaks
python3 repack/harmonic_mra.py run        # Law C scales
python3 repack/tax_tensor.py selftest     # wired to circle
python3 repack/pad_to_tax0.py selftest    # surplus-phase pad
```
