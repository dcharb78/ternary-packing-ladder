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
6. **Characters (open):** tax as nontriviality of a map `⟨2,3⟩→S¹` — next
   algebraic probe; energy/MRA are the concrete stand-ins shipped here.
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
