# Change at 41 — exploratory probe

*Hypothesis, not a theorem. Twin-prime midpoint numerology vs CF/α geometry.*

Code: [`repack/change_at_41.py`](repack/change_at_41.py)  
Results: [`repack/change_at_41_results.json`](repack/change_at_41_results.json)

## Claim under test

User: expect a structural **change at 41**, because 41 is the midpoint of
another system — twin-prime pattern \(p(p-1)\pm 1\):

| p | mid | twins |
|--:|----:|-------|
| 3 | 6 | 5, 7 |
| 5 | 30 | 29, 31 |
| 7 | **42** | **41**, 43 |
| 13 | 312 | 311, 313 |

Packing already knows 41 as the second CF rung of \(\alpha=\log_2 3\)
(5→8, **41→65**, 306→485). Question: is there a packing discontinuity at 41
*beyond* that rung identity?

## Verdict: `pass_as_CF_rung_only`

Within Q∈[38,45], \(\{41\alpha\}\approx 0.983\) is the **local surplus maximum** —
exactly the certified CF-rung signature (`bits(41)=65`). Neighbor jumps in
`max_m_zero_tax` / `tax_rows(8,Q)` / square tax are smooth; nothing unique
fires only when crossing 41 beyond that surplus peak.

**Twin midpoints:** p=3→**5** and p=7→**41** do land on packing rungs (suggestive).
p=5→29/31 and p=13→311/313 do **not**. Prefer the α/CF explanation; do not
elevate twin midpoints to a packing law without a byte-winning rule.

**306 vs 312:** 306 is the next surplus rung (\(\mathrm{dist}(1)\approx 0.0015\));
312 is the p=13 twin midpoint and sits near mid-circle
(\(\mathrm{dist}(1)\approx 0.49\)). Near integers, different roles — not the same object.

## Typed twin-center (update)

UFRF correction: the rule is \(\mathrm{center}(c,m)=2mc\), sheets \(2mc\pm1\),
with typed contexts \(C_0,C_1,C_{\ge3}\) — not \(p(p-1)\). Census starting at
value 3 was a **typing filter** (classical-prime on both sheets), not omission
of layers 0 and 1.

Under that operator, **odd** packing rungs (5, 19, 41, …) appear as sheets;
**even** rung **306 cannot** (sheets are always odd). See
[`ORGANIZING_PRINCIPLES.md`](ORGANIZING_PRINCIPLES.md). The change at 41 is
still the CF surplus peak *and* a typed sheet at \((c,m)=(7,3)\) or \((3,7)\);
those are compatible, not competing, explanations for the odd rung.

## What this does *not* change

- Keep using 41 as a Law-C / fiber block size (already settled).
- Do not redesign the stack around 42 or 312.
- Byte-aware reshape + gated pad remain the operational pipeline
  ([`PACKING_STACK.md`](PACKING_STACK.md)).

## Run

```bash
python3 repack/change_at_41.py selftest
python3 repack/change_at_41.py run
```
