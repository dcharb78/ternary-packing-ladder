# Practical packing pipeline — measured

*Use the geometry. Measure bytes. Diagnose non-wins.*

Code: [`repack/ledger_packer.py`](repack/ledger_packer.py)  
Results: [`repack/ledger_packer_results.json`](repack/ledger_packer_results.json)

## Decision tree (deterministic)

```
if some layout length ∈ {flat MN, row n, col m} is a multiple of 306 or 665
   and the best rung/frame ledger beats flat fmt_5_8:
    use that ledger
      • 306 → prefer fmt_306_485 (rung block density)
      • 486-frame = chiral assembly (usually denser decode story, not bytes)
      • 665 → fmt_665_frame (tax 0; ties 5_8 bytes here)
else if reshape (aspect ≤ 16, align) makes fiber-41 < flat 5_8:
    reshape → axis → fiber-41
else:
    flat fmt_5_8
```

### Odd / even split (codified in packer)

| Family | Rule |
|--------|------|
| **Odd rungs** 5,19,41,53,665,15601 | Sheet/complement tools OK at **design time** |
| **Even 306** | **Frame/rung-block only** — never typed sheets |
| Design-time surplus×align64 `576,1600,2624,3648` | Publish as prior; **never** post-hoc pad |

## BitNet-2B measurement (210 tensors)

| Ledger | Bytes |
|--------|------:|
| Selected (pipeline) | **416,808,960** |
| Flat `fmt_5_8` | 416,808,960 |
| Fiber-41 (original shapes) | 457,344,000 |
| Shipped U8 2-bit | 521,011,200 |

| Path counts | |
|-------------|--:|
| `flat58` | **210 / 210** |
| `frame` | 0 |
| `fiber41` | 0 |

**Verdict: `matches_flat58_knee`** — Δ vs flat 5_8 = **0 B**.

**Why (honest):**
1. No BitNet mode length is a multiple of 306 or 665 → frame branch never fires.
2. Reshape can improve fiber-41 (e.g. 2560²) but reshaped fiber-41 still loses to flat 5_8 on every unique shape.
3. Pipeline correctly selects flat 5_8; vs U8 still **−104.2 MB** (−20%).

No larger near-ternary checkpoint was available locally to re-measure.

## Controlled nesting test

| n | 5_8 B | 306_485 B | 486-frame B | nested decode vs 5_8 |
|--:|------:|----------:|------------:|----------------------|
| 306 | 62 | **61** | 67 | ~54 µs vs ~6 µs |
| 612 | 123 | **122** | 134 | ~62 vs ~7 |
| 1224 | 245 | **244** | 268 | ~85 vs ~8 |
| 665 | 133 | — | 133 (tie) | frame ~60 vs ~7 |

**Recommendation: `keep_never_nested_rule`** for decode-critical paths.

- `fmt_306_485` **wins ~1 byte per 306 trits** vs 5_8 (rung_block_byte_wins=True).
- Law C nesting is ≈ flat-digit speed and **~8× slower** than 5_8 unpack.
- Chiral 486-frame **loses** bytes to both 5_8 and 306_485 on these lengths.

**Split use:**
- Size-critical offline / archive → allow `fmt_306_485` when length ∣ 306.
- Decode-critical inference → keep `fmt_5_8` unless a fused kernel closes the speed gap.

## Synthetic checks (packer sanity)

| Shape | Selected |
|-------|----------|
| 1×306 | `fmt_306_485` (61 B) |
| 2×306 | `fmt_306_485` (122 B) |
| 1224×1 | `fmt_306_485` (244 B) |
| 41×306 | `fmt_306_485` on flat stream (2501 vs 2510) |
| 2560×2560 | `fmt_5_8` |
| 665×1 | `fmt_5_8` (tie with 665-frame) |

## Correction vs earlier O5 claim

[`ORGANIZING_PRINCIPLES.md`](ORGANIZING_PRINCIPLES.md) O5 compared **486-frame @ 306 trits** to **flat 5_8 @ 306×306 trits** (apples/oranges). Same-length truth: 306_485 (61) ≲ 5_8 (62) < 486-frame (67).

## What this means

The geometric work produced a **correct packer and a clear measurement**:

- On unstructured BitNet-scale tensors → **no extra win beyond flat 5_8** (already −20% vs U8).
- On frame-aligned lengths → **rung block `fmt_306_485` is the real density lever** (~1 B / 306 trits), not chiral nesting for speed.
- Odd/even + three-ledger rules prevent searching the wrong family.

That is practical clarification, not a new BitNet compression headline.

## Updated stance (after packet/seam + density + pattern-of-patterns)

See [`PACKET_SEAM.md`](PACKET_SEAM.md) and the multi-track map
[`DIMENSIONS.md`](DIMENSIONS.md). Short form:

- Three-ledger pipeline remains the main practical tool; optional hybrid
  prefixes extend it without changing the default.
- **Shipped static density frontier:** flat Law-B-sum `fmt_665_1055`
  (**132/665**) via `--hybrid665` — **not** “freeze at 0.33%/306.”
- **Probe geometric candidate:** flat **1277** (665+2×306, **253/1277**)
  beats 665 by ~**−0.66 MB** on BitNet ([`FARTHER.md`](FARTHER.md));
  not wired (`theory_bytes_flat_Q` lives in `farther_probes.py` only).
- Hybrid 306 (`61/306` ≈ 0.33% vs 0.2 baseline; **−1 byte per 306 trits**)
  remains a valid prior lever (`--hybrid`) but is below 665 (and 1277).
- The 61/306 gap is the classic rung-block effect — **not** a new seam/twin
  discovery. Chiral frames / fiber-41 remain inferior for density.
- Close the **static** dimension after docs + one larger absolute measure;
  do **not** collapse the other five dimensions into “just use 665/1277.”
- No live-model padding, no theory re-opening.

**Measured follow-up** ([`BETTER_DENSITY.md`](BETTER_DENSITY.md)): on BitNet,
lifting the exact-÷306 gate (`--hybrid`) saves **−1.36 MB** (= 99.99% of the
pure 61/306 gap). Cascade / 41 / rem tricks do not beat that *among 306-family
strategies*. Default packer unchanged.

**Large scale** ([`LARGE_SCALE.md`](LARGE_SCALE.md)): synthetic 7B/13B/70B
suites amplify the 61/306 law (≈ −4 / −8 / −51 MB). No new 306-phenomenon.
Design-time ×306 widths make the default packer take `frame` without `--hybrid`.

**Pattern of patterns** ([`PATTERN_OF_PATTERNS.md`](PATTERN_OF_PATTERNS.md)):
Law B sum as **flat** `fmt_665_1055` (132 B/665) beats hybrid 306
(BitNet **−3.26 MB** vs flat, **−1.89 MB** vs hybrid 306). Chiral 486/665
frames lose bytes. Optional `--hybrid665`.

**Farther** ([`FARTHER.md`](FARTHER.md)): flat 1277 continues the same
Law-B-sum-as-flat pattern (−0.66 MB vs 665). Leave probe-only unless promoting
with a clean optional flag + RT tests.

## Suggestions (next, still low-cost)

1. **`--hybrid665`** — shipped static density frontier (~−3.26 MB on BitNet).
2. **`--hybrid`** — 306-prefix + rem (~1.36 MB; still valid, below 665).
3. **Optional `--hybrid1277`** — only if promoting the farther probe (tiny wire + selftest).
4. **Length-align at train time** — multiples of 665 (or 306) so flat rung-blocks fire without flags.
5. **GEMM-tile ∩ 306/665** — only if designing new widths anyway.
6. **Real >2B ternary ckpt** — re-run `ledger_packer.py bitnet --ckpt … --hybrid665` when available (closes static absolute measure).
7. **Dynamical / multi-scale probes** — see cadence in [`DIMENSIONS.md`](DIMENSIONS.md); do not reopen characters / Stokes / packet theory unless a length-aligned model or fused decode changes the Pareto story.

## Run

```bash
PYTHONPATH=repack python3 repack/ledger_packer.py selftest
PYTHONPATH=repack python3 repack/ledger_packer.py all --nest-repeats 40
PYTHONPATH=repack python3 repack/ledger_packer.py pack --m 1 --n 306
PYTHONPATH=repack python3 repack/ledger_packer.py bitnet --hybrid665
python3 repack/architecture_prior.py run --align 64   # design-time list only
```
