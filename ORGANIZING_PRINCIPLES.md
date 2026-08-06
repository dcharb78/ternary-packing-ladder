# Organizing principles — tested operators

*Exploratory. Symbols and operators are hypotheses until a byte/tax probe
passes or fails. Nulls are progress.*

Code: [`repack/organizing_principles.py`](repack/organizing_principles.py)  
Results: [`repack/organizing_principles_results.json`](repack/organizing_principles_results.json)

Related: [`CHANGE_AT_41.md`](CHANGE_AT_41.md), [`PACKING_STACK.md`](PACKING_STACK.md)

## Operators tested

| ID | Operator | Verdict |
|----|----------|---------|
| **O1/O2** | Typed center `2mc`, sheets `2mc±1` | **Keep — covers odd rungs; misses even 306** |
| **O3** | Complement `φ+ψ≈1` links sheets ↔ rungs | **Keep** |
| **O4** | Snap BitNet dims to nearest surplus (|Δ|≤128) | **Kill** (hurts bytes) |
| **O5** | Rung-block `fmt_306_485` at length ∣ 306 | **Keep for archive density** |
| **O6** | Chiral rectangle tax on twin pairs | **Partial** |

## O1/O2 — Typed twin-center (corrected)

We initially used `M(p)=p(p−1)` — that **misreads** the screenshot
(p=5 → mid 20, not 30). The UFRF typed rule is:

\[
\operatorname{center}(c,m)=2mc,\qquad
\operatorname{sheets}(c,m)=2mc-1,\;2mc+1.
\]

**Typing (contexts), not a missing start at 4:**

| Context | Center | Sheets | Meaning |
|---------|--------|--------|---------|
| \(C_0=0\) | 0 | −1, +1 | signed source / mirror |
| \(C_1=1\) | 2 | 1, 3 | contextual seed + classical 3 |
| \(C\ge3\) | \(2mc\) | \(2mc\pm1\) | classical-prime incidence |

Screenshot exemplars under this rule:

| (c,m) | center | sheets | packing hit? |
|-------|-------:|--------|--------------|
| (3,1) | 6 | 5, 7 | **5** surplus rung |
| (5,3) | 30 | 29, 31 | no rung |
| (7,3) | 42 | 41, 43 | **41** surplus rung |
| (13,12) | 312 | 311, 313 | no rung (≠306) |

Same \(c=5\), other \(m\): `(5,2)` → center 20 → sheets **19**,21 — **deficit rung**.
So the operator is richer than one \(m\) per prime.

### Structural fact (validated)

`2mc` is always even ⇒ sheets are always **odd**.

| Packing rung | parity | Typed sheet? |
|-------------:|--------|--------------|
| 5, 19, 41, 53, 665, 15601 | odd | **Yes** (multiple (c,m)) |
| **306** | **even** | **Impossible** |

**306 is not a failed twin — it is a different operator** (CF surplus / Law B
composition: 7×41+19). The twin-center language and the packing ladder
**share the odd rungs** and **split at the even rung**.

Verdict: `typed_center_covers_odd_rungs_misses_even_306`.

## O3 — Complement operator

Best typed-sheet × rung crosses with **zero row tax** still hold
(5×7, 41×53, …). Same chiral grammar as Law B.

## O5 — Rung-block operator (corrected)

Same-length measurement ([`PACKING_PIPELINE.md`](PACKING_PIPELINE.md)):

| n | fmt_5_8 | fmt_306_485 | 486-frame |
|--:|--------:|------------:|----------:|
| 306 | 62 | **61** | 67 |
| 612 | 123 | **122** | 134 |

Density lever at even quantum 306 is **`fmt_306_485`**, not chiral 486-frame.
(An earlier draft wrongly compared frame@306 trits to flat@306² trits.)

Nested Law C decode ≫ 5_8 decode time → keep never-nested for inference;
allow 306_485 for size-critical archive when length ∣ 306.

> **Ledger choice is an operator.** Flat 5_8, fiber-41, rung-blocks (306),
> and Law-B-sum flats (665; probe 1277) are different symbols. Even 306 is
> where the classic **rung-block** lives — consistent with it not being a
> typed sheet. Fiber-41 is a control, not the density winner.

## O6 — Twin pairs as rectangles

| mid | pair | best_tax | note |
|----:|------|--------:|------|
| 42 | 41×43 | **0** | surplus-driven (not complement) |
| 6 | 5×7 | **0** | true complement |
| 312 | 311×313 | 23 | weak |

## O4 — Design snap (null)

Snapping BitNet widths to surplus grid **hurts** bytes. Architecture prior
is for greenfield widths, not retrofit.

## Revised stack (operators that survived)

```text
1. Pick ledger: flat 5_8 | fiber-41 (control) | fmt_306_485 | fmt_665_1055
   (optional hybrids: --hybrid / --hybrid665; 1277 = probe-only)
2. Reshape (aspect≤16), score by bytes of chosen ledger
3. Axis → flat pack (never nested for decode; never always-pad)
4. Odd rungs ↔ typed sheets (design-time catalog)
5. Even 306 ↔ rung-block only (not twin sheets)
```

See practical measurement: [`PACKING_PIPELINE.md`](PACKING_PIPELINE.md) ·
map: [`DIMENSIONS.md`](DIMENSIONS.md).

## Run

```bash
python3 repack/organizing_principles.py selftest
python3 repack/organizing_principles.py run
```
