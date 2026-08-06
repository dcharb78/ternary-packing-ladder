# Harmonic geometry of the packing tax

*Additive block taxes are a flat chart of a log-domain / circle phenomenon.
Verified identities below; code: `repack/harmonic_tax.py`.*

## The discomfort

Law B and the multi-linear tax form are **correct** and give exact integer
wins — but they count in additive trits/bits while the obstruction is a
**ratio** (powers of 3 vs 2). Continued fractions already live in the log
domain; tensor packing then fell back to additive row/column counts and
padding. That chart mismatch is what felt geometrically forced.

## Exact dictionary (α = log₂ 3)

Verified against `(3^Q).bit_length()` and `tax_rows = m·bits(n)−bits(m·n)`
on dense grids and large mode lengths:

| Additive chart | Harmonic chart |
|----------------|----------------|
| `bits(Q)` | `⌊Q·α⌋ + 1` |
| `tax_rows(m,n)` | `m − 1 − ⌊m·{n·α}⌋` |
| `tax_cols(m,n)` | `n − 1 − ⌊n·{m·α}⌋` |
| zero-tax rows | `{n·α} ≥ 1 − 1/m` |
| axis holonomy | `tax_rows − tax_cols = m·bits(n) − n·bits(m)` |

Raw torus 2-form before flooring: `τ(m,n) = m{n·α} − n{m·α}` (related; the
**tax** itself is the floored form above).

## Law B, reread

| Q | `{Q·α}` (approx) | class | max m with `tax_rows=0` |
|---|------------------|-------|-------------------------|
| 5 | 0.925 | surplus near 1 | 13 |
| 41 | 0.983 | surplus near 1 | 49+ |
| 306 | 0.999 | surplus near 1 | 49+ |
| 19 | 0.114 | mid | 1 |
| 53 | 0.003 | deficit near 0 | 1 |
| 665 | ~0 | deficit near 0 | 1 |

**Surplus rungs** are points whose fractional part sits just below 1 — almost
spilled into the next power of two. **Strong deficits** (53, 665) sit near 0;
19 is a mid-phase remainder that still participates in Law B assemblies.
Chiral tax and zero-tax assemblies are statements about how these circle
points add when you take `m` copies (`⌊m·{n·α}⌋`).

## Axis choice, reread

Prefer the axis with smaller floored mass of the other mode’s phase:

- `tax_rows` small when `{n·α}` is large (column length is “surplus-like”)
- On BitNet `6912×2560`, harmonic holonomy matches the integer probe
  (`prefer cols` when packing the short fiber), without ever forming `3^{MN}`.

## What this does *not* claim

- A new packing format that beats `(5,8)` on unstructured BitNet (still no).
- That `τ` alone replaces the floored tax (floors are essential; τ is the
  pre-quantum geometry).
- Full 2-adic/3-adic character theory or a Dirichlet energy whose critical
  points are all zero-tax frames — those are the **next** harmonic probes.

## Possible next probes

1. **Energy:** define `E({φ_i}) = Σ ⌊m_i φ_j⌋`-style Dirichlet form on the
   torus whose minima recover the tax-0 catalog.
2. **Wavelet / CF recurrence:** write hierarchical digit decode as
   multiresolution on the 2–3 scale (Law C as harmonic MRA).
3. **Characters:** tax as failure of a homomorphism `⟨2,3⟩ → S¹` to be trivial.
4. Keep additive ledgers for engineering; use `{Q·α}` for design (choose mode
   lengths with surplus phase, pad toward high `{L·α}`).

```bash
python3 repack/harmonic_tax.py selftest
python3 repack/harmonic_tax.py run
```
