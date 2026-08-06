# Farther — underexplored angles past the 665 frontier

*Grounded brainstorm + cheap probes. Keep dimensions separate
([`DIMENSIONS.md`](DIMENSIONS.md)). Do not reopen fiber-41 / chiral frames /
digit nesting / pad-to-align / seam-twin predictors as density levers.*

Code: [`repack/farther_probes.py`](repack/farther_probes.py)  
Results: [`repack/farther_probes_results.json`](repack/farther_probes_results.json)

```bash
PYTHONPATH=repack python3 repack/farther_probes.py selftest
PYTHONPATH=repack python3 repack/farther_probes.py run
```

---

## What we already knew

| Worked | Failed / null |
|--------|----------------|
| Flat Law-B **containers** (665 > 306 > 5_8) | Fiber-41 as density |
| α-circle as **language** | Chiral multi-part frames for density |
| Base-p same machine (three loci) | Digit nesting / pad-to-align |
| Hybrid prefixes without exact ÷Q | Seam/twin as predictors |
| | Dynamical nulls on BitNet (next_probes) |

---

## Idea board (all angles)

Tags after probes: `measured` / `null` / `speculative` / `false_id` /
`applies_as_language` / `micro-win`.

### A. Mixed-ledger portfolio

Per-tensor best of {5_8, hybrid306, hybrid665} was already true; try a
**global** budget that minimizes total bytes + λ·decode-complexity.

**Probe:** λ-sweep on BitNet shapes.  
**Result:** On pure bytes, 665 wins **all 210** tensors → portfolio = all-665.
Large λ only trades bytes for simpler decode.  
**Tag:** `null` (density) / decode tradeoff only. **Beats 665?** No.

### B. Activation / KV / runtime tensors

If activations or KV buffers were ternary (synthetic shapes), does 665 still win?

**Probe:** 8 synthetic act/KV/runtime shapes.  
**Result:** hybrid665 wins all 8 on theory bytes.  
**Tag:** `measured`. **Beats 665?** N/A (665 still best).

### C. Non-uniform trit entropy

BitNet real histograms: does adaptive/ANS beat geometric 665?

**Probe:** 12 sampled U8 weight tensors → empirical H + static range coder.  
**Result:** mean p(0)≈**0.43**, H≈**1.549** < log₂3≈1.585. Adaptive beats
665 on the sample (~**−0.55 MB** on those tensors). Different **family**
(variable-rate + counts header), not a Law-B flat.  
**Tag:** `measured` (entropy win). **Beats 665 on bytes?** Yes (Pareto:
decode complexity). Not a new geometric container.

### D. Tile ∩ 665/306

GEMM-friendly tiles that are multiples of 5 / near 665 factors; joint
byte + alignment score.

**Probe:** tile scan + joint score.  
**Result:** Exact ×665 densest; pow2 wins alignment, loses density. Design-time
prior only: prefer widths that are ×665 **and** mult of 16/64.  
**Tag:** `null` (density) / `applies_operationally` (greenfield prior).  
**Beats 665?** No.

### E. Inverse / dual (bits → p-ary)

Pack bitstreams into trit containers for metadata/scales.

**Probe:** bit lengths 64…65k → trit count → 665 pack vs raw bytes.  
**Result:** Always expands (log₂3 > 1).  
**Tag:** `false_id` / `null`. **Beats 665?** N/A; never beats raw bits.

### F. Cross-tensor shared remainder dictionary

Many rem = n mod 665 tails; shared codebook?

**Probe:** BitNet rem histogram.  
**Result:** Only **3** rem classes (400, 505, 25). Shared codebook saves
**~13 KB** vs inline rem (~−0.013 MB vs hybrid665). Micro-win.  
**Tag:** `micro-win`. **Beats 665?** Technically yes, negligible.

### G. Streaming chunk = rung

I/O block sizes at 665; fragmentation metric.

**Probe:** chunk trit sizes vs flat 665 stream.  
**Result:** Chunk=665 matches flat 665 (zero extra waste). Misaligned chunks
only add waste.  
**Tag:** `null` as density lever. **Beats 665?** No.

### H. Next flat assembly after 665

Is there a Law-B flat larger than 665, still practical, denser than 132/665?

**Probe:** Law-B sums + surplus scan Q∈[666,4000]; BitNet absolute on best
practical denser.  
**Result:** **Yes — flat Q=1277 (=665+2×306) → 253 B/block**  
(≈**0.19812** B/trit vs 665 ≈**0.19850**). On BitNet-2B: **−0.66 MB vs
hybrid665**, **−3.92 MB vs flat 5_8**; wins all 210 tensors. Same pattern as
665: **flat container of a sum**, not chiral parts. 15601 still denser on
paper but not operational.  
**Tag:** `measured` / `applies_operationally`. **Beats 665?** **Yes**
(new geometric frontier candidate).

### I. Base-5 mini codec stub

One convergent flat block round-trip for p=5.

**Probe:** RT on Q∈{3,28,59}; theory at Q=643.  
**Result:** All RT ok; 643 beats naïve (theory). Pattern transfer confirmed.  
**Tag:** `measured` (non-ternary). **Beats ternary 665?** No / `does_not_apply`.

### J. Blind spots invented

| Idea | Result | Tag |
|------|--------|-----|
| Rem-phase correlation as rem predictor | Phase is rename of rem size; no extra bytes | `applies_as_language` / `null` |
| Dual-radix float16/scale via trits | Expands vs binary scales | `null` |
| Softmax-score tensors as “ternary” | Covered under act shapes; 665 still wins | `measured` |
| Portfolio of 1277+665+306 | Speculative follow-up (compose next flats) | `speculative` |
| Train-time loss term for packed bytes | Untouched | `speculative` |
| Concurrent-scale rem dictionary | Speculative | `speculative` |
| 2-adic vs 3-adic metadata indexing | Speculative; inverse probe already expands | `speculative` |

---

## Probe scoreboard

| # | Idea | Tag | Beats 665-flat on ternary weights? |
|---|------|-----|-------------------------------------|
| 1 | Mixed-ledger portfolio | `null` | No |
| 2 | Act/KV shapes | `measured` | 665 still wins |
| 3 | Adaptive entropy | `measured` | **Yes** (entropy family; sample −0.55 MB) |
| 4 | Tile ∩ 665/306 | `null` / prior | No |
| 5 | Shared rem dict | `micro-win` | **Yes** (~−13 KB) |
| 6 | Streaming chunk=665 | `null` | No |
| 7 | **Next flat 1277** | **`measured`** | **Yes (−0.66 MB)** |
| 8 | Base-5 mini codec | `measured` (other p) | No (ternary) |
| 9 | Inverse bit→trit | `false_id` | No |
| 10 | Blind rem-phase / dual scales | `null` | No |

### Anything that beats 665?

1. **Geometric (same family):** flat **1277** (253 B / 1277) — **primary farther win**.
2. **Entropy family:** adaptive static coder when p(0) elevates — different Pareto.
3. **Bookkeeping:** shared rem codebook — ignore for frontier (~KB).

Static packing frontier update candidate: **1277-flat ≥ 665-flat ≥ hybrid 306 ≥ 5_8**.
Optional packer flag not implemented in this pass (probe-only); wire
`theory_bytes_1277_*` / `--hybrid1277` when promoting.

---

## Stance

The pattern that worked for 665 **repeats**: take a Law-B **sum**, pack it as
one **flat** bigint block. Next practical step is 1277, not chiral frames and
not 15601. Entropy coding is a parallel track when histograms leave uniform.
Everything else on this board is null, micro, or non-ternary transfer.

Do **not** collapse circle / triskelion / multi-scale into “just use 1277.”
