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

## Updated stance (after packet/seam + density dissection)

See [`PACKET_SEAM.md`](PACKET_SEAM.md). Short form:

- Three-ledger pipeline remains the main practical tool.
- The ~0.33% (61/306) is the **classic rung-block** option — design-time ×306
  or optional hybrid prefix — **not** a new seam/twin discovery.
- No live-model padding, no theory re-opening.

**Measured follow-up** ([`BETTER_DENSITY.md`](BETTER_DENSITY.md)): on BitNet,
lifting the exact-÷306 gate (`--hybrid`) saves **−1.36 MB** (= 99.99% of the
pure 61/306 gap). Cascade / 41 / rem tricks do not beat that. Default packer
unchanged; use `--hybrid` when you want the known density without redesign.

**Large scale** ([`LARGE_SCALE.md`](LARGE_SCALE.md)): synthetic 7B/13B/70B
suites amplify the same ~0.33% (≈ −4 / −8 / −51 MB). No new phenomenon.
Design-time ×306 widths make the default packer take `frame` without `--hybrid`.

## Suggestions (next, still low-cost)

1. **Length-align at train time** — multiples of 306 so `fmt_306_485` fires (0.33% density).
2. **Optional `--hybrid`** — 306-prefix + rem if widths cannot change (~1.36 MB on BitNet; scales at larger models).
3. **GEMM-tile ∩ 306** — only if designing new widths anyway.
4. **Real >2B ternary ckpt** — re-run `ledger_packer.py bitnet --ckpt … --hybrid` when available.
5. **Do not** reopen characters / Stokes / packet theory unless a length-aligned model or fused decode changes the Pareto story.

## Run

```bash
python3 repack/ledger_packer.py selftest
python3 repack/ledger_packer.py all --nest-repeats 40
python3 repack/ledger_packer.py pack --m 1 --n 306
python3 repack/architecture_prior.py run --align 64   # design-time list only
```
