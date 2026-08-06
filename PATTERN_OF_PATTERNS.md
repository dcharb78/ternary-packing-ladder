# Pattern of patterns — composition vs hybrid 306

*Primary lens: CF ladder / Law B assemblies as ledgers. Fiber-41 is control only.*

**Multi-track context:** this result updates the **static packing** frontier
only. See [`DIMENSIONS.md`](DIMENSIONS.md) for circle / triskelion /
sphere-and-cube / multi-scale / training tracks — do not collapse them into
“just use 665.” Farther geometric candidate past 665: flat **1277**
([`FARTHER.md`](FARTHER.md); probe-only).

Code: [`repack/pattern_of_patterns.py`](repack/pattern_of_patterns.py)  
Results: [`repack/pattern_of_patterns_results.json`](repack/pattern_of_patterns_results.json)

## Question

Can multi-rung composition beat flat `fmt_5_8` and/or plain hybrid
`theory_bytes_306_485` (61/306 ≈ 0.33%) with honest byte metrics?

## Candidates

| ID | Ledger |
|----|--------|
| A | flat `fmt_5_8` |
| B | hybrid `theory_bytes_306_485` (known control) |
| C | fiber-41 (control) |
| D | Law B **486-frame** (7×41+19 chiral parts) |
| E | Law B **665-frame** (2×306+53 chiral parts) |
| **F** | **flat 665-prefix** `theory_bytes_665_1055` (Law B *sum* as rung-block) |
| G | greedy 665-flat + 306 rem |
| H | DP over atoms `{5,19,41,53,306,665}` |
| I | cascade 306→41→5_8 |

## Verdict

**Yes — F beats hybrid 306.** The win is the flat container of the Law B sum
(`665 → 132 B`), **not** chiral multi-part frames.

| Ledger @ quantum | Bytes |
|------------------|------:|
| `fmt_306_485` @ 306 | **61** |
| 486-frame @ 306 | 67 (loses) |
| **`fmt_665_1055` @ 665** | **132** |
| 665-frame @ 665 | 133 (loses to flat sum) |
| flat `5_8` @ 665 | 133 |

### BitNet-2B (210 tensors)

| Strategy | Δ vs flat `5_8` | Δ vs hybrid 306 |
|----------|----------------:|----------------:|
| B hybrid 306 | −1.36 MB | 0 |
| **F hybrid 665-flat** | **−3.26 MB** | **−1.89 MB** |
| G 665+306 rem | −3.14 MB | −1.77 MB |
| E 665-frame | −0.09 MB | +1.27 MB |
| D 486-frame | +39 MB | +40 MB |
| C fiber-41 | +40.5 MB | +42 MB |

Oracle path: **F on all 210 tensors.**

### Synthetic 7B / 70B (`large_scale_probe` suites)

| Suite | B vs flat | F vs flat | F vs B |
|-------|----------:|----------:|-------:|
| 7B | −4.2 MB | **−9.8 MB** | **−5.5 MB** |
| 13B | −8.3 MB | **−19.1 MB** | **−10.8 MB** |
| 70B | −51.3 MB | **−117 MB** | **−65.7 MB** |
| 7B ×306-aligned | −4.1 MB | **−9.4 MB** | **−5.3 MB** |

Fiber-41 still loses badly at every scale.

## Claim tags

| Claim | Tag |
|-------|-----|
| Flat 665 Law-B-sum blocks beat hybrid 306 | **`applies_operationally`** |
| Hybrid 306 ≈ 0.33% vs flat | `applies_operationally` |
| Law B language names the larger container | `applies_as_language` |
| Chiral 486 / 665 frames denser in bytes | **`does_not_apply`** |
| DP atom mosaic beats F | `does_not_apply` |
| Fiber-41 as primary lever | `does_not_apply` |
| Digit nesting / preserve 0→1 density path | **`false_identification`** |
| Kronecker densifies unstructured BitNet | `does_not_apply` |

## What this is *not*

- Not digit nesting (already false-ID; keep `never_nested` for decode-critical).
- Not “assemble parts for density” — part byte-rounding **costs** bytes vs the flat sum.
- Not a fiber-41 story.

## Packer

- Format: `fmt_665_1055` / `theory_bytes_665_1055` in [`pack_ladder.py`](repack/pack_ladder.py)
  (132 B/block; opt-in, not in default `FIXED_FORMATS`).
- Flag: `ledger_packer.py … --hybrid665` (optional; default unchanged).
- Prefer F over chiral `fmt_665_frame` for size.
- Next Law-B-sum flat **1277** (253 B/block) is measured in
  [`farther_probes.py`](repack/farther_probes.py) only — **no** `--hybrid1277`
  unless promoted.

```bash
PYTHONPATH=repack python3 repack/pattern_of_patterns.py selftest
PYTHONPATH=repack python3 repack/pattern_of_patterns.py run
PYTHONPATH=repack python3 repack/ledger_packer.py selftest
PYTHONPATH=repack python3 repack/ledger_packer.py bitnet --hybrid665
```
