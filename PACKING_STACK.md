# Packing stack — measured, diagnosed

*Hardened pipeline from the geometry lab. Bad byte results were not accepted
blindly — ablations show **why**.*

Code: [`repack/packing_stack.py`](repack/packing_stack.py)  
Results: [`repack/packing_stack_results.json`](repack/packing_stack_results.json)

## Pipeline (revised after measurement)

```text
reshape (aspect ≤ 16, align, min_dim) → pick axis → pack flat
         ↑
fiber surplus pad ONLY if a byte gate passes (default: fiber41 improves)
```

Naïve “always pad fiber” **looks great on tax and is harmful on bytes.**

## Deployability filter

`DeployConstraints(max_aspect=16, align=64, min_dim=32)`:
- rejects extreme ribbons
- prefers multiples of `align` (soft-relax if nothing legal)
- objective: minimize `best_tax`, then aspect

## Associator (250 triples)

| Metric | Value |
|--------|------:|
| Fraction nested tax > flat | **0.98** |
| Nesting “acceptable” bins | only tiny `*surplus|surplus*` ties (associator = 0) |
| Rule | **prefer flat almost always** |

Phase-bin means are large and positive whenever deficit/mid modes appear in
the product. Nesting is not a packing-tax strategy.

## Byte measurements

Constraints: `max_aspect=16`, `align=64`, `pad_gate=fiber41` unless noted.

### LLM suite (11 shapes)

| Variant | Δ flat 5_8 | Δ fiber-41 | Δ Σ tax | trit Δ | pads kept |
|---------|-----------:|-----------:|--------:|-------:|----------:|
| **gated (default)** | 0 | **−54,272** | −19,712 | 0 | 0 / 11 |
| no pad | 0 | −54,272 | −19,712 | 0 | — |
| **always pad** | **+105,191** | **+59,944** | −33,615 | +525,952 | 11 / 11 |

### BitNet-2B (210 tensors, weighted)

| Variant | Δ flat 5_8 | Δ fiber-41 | Δ Σ tax | trit Δ | pads kept |
|---------|-----------:|-----------:|--------:|-------:|----------:|
| **gated** | 0 | **−215,040** | −168,960 | 0 | 0 / 210 |
| no pad | 0 | −215,040 | −168,960 | 0 | — |
| always pad (prior run) | +1.24 MB | +1.14 MB | huge ↓ | +6.2M | all |

Per unique shape (reshape only / gated = same here):

| Shape | → reshaped | Δ fiber-41 / tensor | note |
|-------|------------|--------------------:|------|
| 2560×2560 (×60) | 640×10240 | **−5,120** | square break helps |
| 640×2560 (×60) | 3200×512 | 0 | tax↓, bytes flat |
| 2560×6912 (×30) | 1280×13824 | **+1,024** | tax↓, fiber-41 **worse** |
| 6912×2560 (×60) | 13824×1280 | **+1,024** | same |

Net BitNet fiber-41 win ≈ **210 KB** — real, free (no pad), deployable aspect.

## Why results looked “bad” (and what we missed)

1. **Tax ≠ bytes.** Surplus fiber pad collapses tax but **adds trits**. On
   `fmt_5_8` / fiber-41 ledgers, pad growth dominated (~1.6 bit/trit). We
   almost shipped a tax-optimized stack that **inflates** payloads.

2. **Always-pad is the bug, not reshape.** Ablation: reshape-only matches
   gated stack; always-pad destroys both flat 5_8 and fiber-41.

3. **Byte gate kills pad on these models.** After reshape, fiber surplus pad
   never improved fiber-41 bytes (0 kept / 210). Pad remains useful as a
   *design-time* prior for choosing sizes, not as a post-hoc inflate.

4. **Still does not beat flat `fmt_5_8`.** Post-stack fiber-41 on BitNet is
   still ~457 MB − 210 KB ≫ 417 MB flat 5_8. Same Pareto knee as
   [`CHECKPOINT_PROBE.md`](CHECKPOINT_PROBE.md). Geometry improves the
   **fiber-41 family**, not the density champion.

5. **Reshape is not uniformly helpful.** MLP rectangles (2560×6912) lose
   +1 KB fiber-41 when reshaped under the tax-minimizing objective. A
   byte-aware reshape objective would reject those moves (next hardening).

## Settled operational stack

| Step | Keep? | Role |
|------|-------|------|
| Mild reshape (aspect≤16, align) | **Yes**, with byte check | Free on squares; verify on rectangles |
| Axis choice | **Yes** | Free holonomy |
| Fiber surplus pad | **Only if byte gate passes** | Design-time; rarely post-hoc |
| Flat > nested | **Yes** | Associator ~98% |
| Prefer surplus over ½ | **Yes** (design) | Not via payload-growing pad |
| Reshape objective = **fiber41 bytes** | **Yes** | Tax-min reshape can worsen bytes |

## Byte-aware reshape (push)

Default `reshape_objective=fiber41` (was tax). Ablation on BitNet (no pad):

| Objective | Aggregate Δ fiber-41 | What changes |
|-----------|---------------------:|--------------|
| **tax** (legacy) | −215,040 | MLP 2560×6912 reshaped → **+1024**/tensor |
| **fiber41** (default) | **−307,200** | MLP stays identity; squares still break |

Per unique shape:

| Shape | tax objective | fiber41 objective |
|-------|---------------|-------------------|
| 2560² | →640×10240 (−5120) | same |
| 2560×6912 | →1280×13824 (**+1024**) | **identity (0)** |
| 6912×2560 | →13824×1280 (**+1024**) | **identity (0)** |
| 640×2560 | reshape tax-neutral | identity |

LLM suite: tax Δ41=−54,272 → fiber41 Δ41=**−59,392**.

**Understanding:** tax and fiber-41 bytes are correlated but not identical.
Minimizing tax can select reshapes that move mass onto worse fiber lengths.
Score reshape by the ledger you ship.

## Architecture prior (design-time)

`python3 repack/architecture_prior.py run --align 64`

On the 64-grid up to 4096: **4** surplus lengths (`φ≥0.9`):
**576, 1600, 2624, 3648**. Use when *choosing* widths — still does not beat
flat `(5,8)` as a post-hoc pack on dense BitNet.

## Change at 41 (hypothesis probe)

Twin-prime midpoint story (42±1→41,43; 312±1; …) vs CF rung 41.

Verdict: **`pass_as_CF_rung_only`** — see [`CHANGE_AT_41.md`](CHANGE_AT_41.md).
`{41α}` is the surplus local max in 38..45 (the known rung). No extra
discontinuity vs 40/42/43. 306≪312 on distance-to-1 (rung vs twin midpoint).
Twin midpoints that hit rungs (p=3→5, p=7→41) are suggestive coincidences;
prefer the α/CF explanation.

## What not to do next

- Do not open Stokes / character fronts for packing bytes on dense BitNet.
- Do not always-pad production tensors.
- Do not claim the stack beats `(5,8)` on unstructured ternary.
- Do not treat twin-prime midpoints as a packing law without a byte-winning move.

## Run

```bash
python3 repack/packing_stack.py selftest
python3 repack/packing_stack.py all --reshape-objective fiber41 --pad-gate fiber41
python3 repack/architecture_prior.py run --align 64
python3 repack/change_at_41.py run
```
