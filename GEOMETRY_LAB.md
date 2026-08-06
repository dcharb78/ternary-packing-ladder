# Geometry lab — keep only cheap, testable wins

*Deeper geometry continues only while it produces low-cost improvements.
Nulls are progress. No Collatz predictors. No claim vs flat `(5,8)`.*

Code: [`repack/geometry_lab.py`](repack/geometry_lab.py)  
Results: [`repack/geometry_lab_results.json`](repack/geometry_lab_results.json)

## Probes

| ID | Idea | Symbolic reading |
|----|------|------------------|
| A | Associator `nested_tax − flat_tax` | Tax under composition `(m,n)·p` vs `m·(np)` |
| B | Stokes `hol(m,n)+hol(n,p)+hol(p,m)` | Is holonomy a closed 2-form on size triangles? |
| C | Volume-preserving reshape `(m/d)×(nd)` | Lattice move on the torus of shapes |
| D | Fiber-only surplus pad | Pad the amplified mode; leave multiplicity |
| E | Break-a-square `Q×Q → (Q/d)×(Qd)` | Square = degenerate rectangle; open it |
| F | Character `cos/sin(2πφ)` vs tax | Fourier of phase as packing signal |

Aspect cap **≤16** separates math (extreme ribbons) from deployable reshapes.

## Survivors (actionable)

### A — Prefer flat over nested

Nested packing tax is almost always worse than flat `(m)×(n·p)`:
`fraction_nested_worse ≈ 0.92`, associator ≥ 0 on all samples, often huge.

**Move:** when you have three modes, pack the product fiber flat; do not
accumulate Law-B tax twice through a hierarchy unless decode structure pays.

### C — Aspect-bounded reshape (≤16)

**9 / 11** LLM shapes improve under `max/min ≤ 16` (mean Δ best_tax ≈ −1804).
Examples:
- `4096×4096 → 1024×16384` (Δ −3072)
- `2560×2560 → 640×10240` (Δ −640)
- `6912×2560 → 13824×1280` (Δ −1280)

Unconstrained reshape always “wins” by becoming a ribbon (aspect ≫ 16) —
that is a geometric truth, not a systems win. Gate on aspect.

**Move:** before padding, search divisor reshapes inside the aspect budget.

### D — Pad fiber only

**10 / 11** shapes: fiber-only surplus pad has best tax-drop per pad trit.
Padding multiplicity is usually wasted; pad-both doubles cost for the same tax.

**Move:** choose winning orientation → pad only the fiber toward surplus.

### E — Break squares (aspect ≤16)

All five tested squares improve by becoming rectangles, e.g.
`2560×2560 → 640×10240` (tax 1269 → 629). Fits the square-vs-rectangle
story: a square freezes two copies of one seed; a mild reshape gives
holonomy + a chance at a better fiber phase.

**Move:** never pack a large square as a square if a mild reshape is legal.

## Kills / defer

| ID | Verdict |
|----|---------|
| **B Stokes** | Flux nonzero (not a closed form), but **no cheap rule yet** — symbolic lead only |
| **F Character** | `r(tax, \|1−φ\|)≈0.999` dominates; `cos` useless (`r≈0.02`). `sin` has partial correlation (~0.78) but adds nothing over surplus distance |

## Pipeline that survives

```text
reshape (aspect≤16) → pick axis → pad fiber toward surplus → pack flat (not nested)
```

That is the low-cost geometry stack — **revised after byte measurement**
([`PACKING_STACK.md`](PACKING_STACK.md)):

```text
reshape (aspect≤16, align) → pick axis → pack flat
# fiber surplus pad ONLY if a byte gate passes
```

Always-pad collapses tax but inflates bytes. Deeper symbolic work continues
only if it invents a move that beats this stack on **bytes**, not tax alone.

## Run

```bash
python3 repack/geometry_lab.py selftest
python3 repack/geometry_lab.py run --max-pad 64
```
