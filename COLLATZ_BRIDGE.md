# Collatz ↔ packing bridge (exploratory)

*Phase 4 hypothesis map. Nothing here is a packing theorem until a probe
passes or fails against the settled circle dictionary. Seed → amplify →
harmonize is a framing to test, not a claim to certify.*

Primary code: [`repack/process_language_probes.py`](repack/process_language_probes.py).
Settled math: [`HARMONIC.md`](HARMONIC.md), [`repack/harmonic_tax.py`](repack/harmonic_tax.py).

## Stance

Both projects study competition of ×2 and ×3 via α = log₂ 3. Packing reads it
as irrational rotation `{Qα}` on the circle; Collatz reads it as
expansion-vs-contraction drift. Shared **geometry** is transferable as probes;
Collatz **dynamical predictors** that Q4–Q7 falsified are not.

## Map (geometry that survives)

| Packing | Collatz / UFRF | Shared object |
|---------|----------------|---------------|
| Rotation by α | 2-adic valuation vs ×3 growth | ×2 vs ×3 |
| `{Qα}` near 0/1 | Critical flips / phase halves | Preferred loci on circle |
| Self-complement φ≈½ (BitNet `{2560α}≈0.504`) | `flip_at_half` (6.5/13 = ½) | Special role of ½ |
| Law C MRA / hierarchical digits | Recursive towers, base-13 digits | Nested scales |
| Multi-mode phase cloud | Concurrent scales, breathing score | Simultaneous phases |
| Complements φ+ψ≈1 | Phase alignment / complement | Cancellation |

## Do not import as predictors (Collatz Q4–Q7 dead ends)

From Collatz `exploration/FINDINGS.md` (explore-v2 tree):

- Breathing score / coarse-scale compensation as predictors of length or streaks
- Trinity mod-3 class as convergence-speed predictor
- Cover time as convergence bound; Pythagorean comma vs log(n)
- Base-13 digit pairs as streak predictors
- 3-adic modulus enrichment; rotating coset culprits
- Period-13 in W(k) as useful structure (k=13 is a **break**, not a gift)

**Packing implication:** breathing score may still be a cloud *summary*
statistic; do not treat it as a packing certificate. Process language must
reduce to `tax_rows` / complements or fail clearly.

## Process language (hypothesis)

| Word | Proposed packing meaning | Exact object already in code |
|------|--------------------------|------------------------------|
| **Seed** | Small surplus/deficit generators; `{Qα}` | `frac_Q_alpha`, rung catalogs |
| **Amplify** | Scale phase by mode m → `⌊m·{nα}⌋` | Floor term in `tax_rows` |
| **Harmonize** | Bring phases so φ+ψ≈1 | Complements; pad tracker |
| **Tax** | Dissonance after failed harmonization | `tax_rows = m−1−⌊m·φ_n⌋` |
| **Zero-tax** | Successful harmonization | `{nα} ≥ 1−1/m` |
| **Padding** | Retune seed before amplify | `pad_toward_surplus_phase`, `multi_mode_pad` |
| **Return map** | Feed residual as new seed | Exploratory probe only |

Validation rule: probes either rename settled identities faithfully, or report
null/fail. No new theorems by renaming.

## Provenance (beyond README)

| Source | What it contributes |
|--------|---------------------|
| [`TERNARY_PACKING_LADDER.md`](TERNARY_PACKING_LADDER.md) | Prepared from UFRF Collatz parallax; Law B integers match Collatz descent constants; waste `2−α` |
| [`HARMONIC.md`](HARMONIC.md) | Circle dictionary; open characters / global budget |
| [`EXTENSIONS.md`](EXTENSIONS.md) | Collatz schedule plumbing works; on uniform trits signal is tax form |
| [`repack/collatz_schedule.py`](repack/collatz_schedule.py) | Orbit shapes → frame schedule; surplus/deficit catalogs |
| [`repack/harmonic_*.py`](repack/) | Operational amplify/harmonize math without those names |
| [`METHOD_COMPARISON.md`](METHOD_COMPARISON.md) | Empirical phase language in BitNet weights (not Collatz) |
| Collatz `UFRF/BreathingCycle.lean`, `CollatzSolenoid.lean`, `q4*`–`q7` | Seed wrap, millibit α, breathing formula, W(k) break — definitions only |
| Collatz `FINDINGS.md` | Authoritative **negative** results — not packing todos |

**Gap:** packing note cites `UFRF/ParallaxRungCertificates.lean`; that path was
not found under explore-v2. Shared cert content lives as packing
[`lean/Certificates.lean`](lean/Certificates.lean) plus Collatz millibit /
`3^L≠2^S` spine. Do not invent the missing file.

UFRF “sphere packing” / kissing-number metaphors are **not** ternary weight
packing — different object.

## Anti-tunnel caveats

- Do not overweight Collatz Q4–Q7 scripts over packing’s own open questions
  (global phase budget, architecture phase design, characters, dynamic MoE,
  Beatty tilings).
- Do not rediscover settled facts: circle dictionary, BitNet self-complement,
  flat `(5,8)` still wins density, 0 Kronecker tiles on dense BitNet.
- Practical packing wins already known: axis choice, pad-to-tax0 grammar,
  factor pack when structure exists — keep those in the foreground.
- Do not chase deeper unstructured 1-D rates.
- Deferred Collatz-shaped rabbit holes (not Phase-4 blockers): CF-depth MRA
  discontinuity hunting; modular-vs-exact comma correction bands.

## How to run

```bash
python3 repack/process_language_probes.py selftest
python3 repack/process_language_probes.py run          # uses BitNet ckpt if present
python3 repack/process_language_probes.py run --skip-ckpt
```
