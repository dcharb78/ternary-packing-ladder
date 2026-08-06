# Scale probe + creative connections

*Targeted larger-shape experiments — not a blind bigger flat stream.
See also `repack/scale_probe_results.json` and `EXTENSIONS.md`.*

## Hypothesis (confirmed by design)

A larger model is useful only to probe **axis choice** and **structure
exploitation**. Unstructured flat packing at larger scale will not reveal
new 1-D physics.

## What we ran

```bash
cd repack && python3 scale_probe.py run
```

Realistic LLM-class rectangles (7B–34B-ish shapes), exact integer ledgers:
row-fiber vs col-fiber vs flat block formats; novel tax-0 tile fits;
Kronecker factor packing; stream-slack tension.

## Results (headline)

### Axis choice is free and shape-conditional

| tensor | shape | best axis | bits saved | `fmt_41_65` bytes saved |
|--------|-------|-----------|------------|-------------------------|
| llama7b attn Q | 4096² | — | 0 | 0 |
| llama7b MLP up | 11008×4096 | **cols** | 7936 | 256 |
| llama7b MLP down | 4096×11008 | **rows** | 7936 | 256 |
| llama13b MLP up | 13824×5120 | **cols** | 11264 | 1024 |
| llama34b MLP up | 22016×8192 | **cols** | 17920 | 1024 |
| head slice | 128×4096 | rows | 384 | 384 |
| kv group | 1024×4096 | rows | 3072 | 3072 |

Square projections: **zero** axis gain (symmetric). Rectangular MLPs: always
prefer packing along the **shorter fiber** in these examples (holonomy sign).

Sum of best-axis `fmt_41_65` savings across the probe set: **7040 bytes** —
small vs a full checkpoint, but **zero-cost** and scales with every layer
replica. The bit-level holonomy is larger; byte deltas are often remainder
alignment under 41-trit framing.

### Kronecker / grouped structure still dominates when present

| case | factor bytes | flat bytes | delta |
|------|--------------|------------|-------|
| 16×16 ⊗ 16×16 `fmt_5_8` | 104 | 13108 | −13004 |
| 32×32 ⊗ 16×16 `fmt_5_8` | 257 | 52429 | −52172 |
| shared 16×16 core ×64 groups | 52 | 3277 | −3225 |

### Novel tax-0 frames rarely divide stock LLM dims

Most LLaMA widths (4096, 11008, …) are **not** multiples of catalog tax-0
frame lengths (60, 101, 665, …). Tilers found only on a couple of shapes.
**Creative implication:** treat tax-0 frames as a *tiling grammar to design
toward* (pad/quantize mode length to nearest tax-0 multiple), not as a drop-in
for today’s shapes.

### Stream vs block — opposite layout advice

BitNet measured ~4 B slack / stream. Per-row streams on a 4096×N matrix
would pay ~16 KB slack vs ~4 B for one flat stream. So:

- **Block formats:** choose low-tax axis (`tax_rows` vs `tax_cols`).
- **Stream formats:** prefer **one** stream (flatten); axis splitting hurts slack.

---

## Unseen / creative connections

1. **Torus holonomy = Law B chirality between orientations**  
   `m·bits(n) − n·bits(m) = tax_rows − tax_cols`  
   No need for `bits(MN)`. Axis choice is the 2-D chiral tax.

2. **Stream–block tension**  
   Same tensor; opposite optimal layouts by format family (above).

3. **Tax-0 tile grammar**  
   Catalog frames define preferred mode lengths. Push training/export to
   emit widths in that grammar → novel tax-0 assemblies become practical.

4. **Factor tree = decode tree**  
   Kronecker pack and hierarchical digits share a tree; fused kernels can
   contract packed factors without materialising `A⊗B`.

5. **GEMM microtile resonance**  
   Align catalog `frame_q` with 16/32/64/128/256 compute tiles so pack block
   = matmul block (search already stubbed in `scale_probe.py`).

6. **BitNet module phase × axis choice**  
   Orthogonal levers: per-module adaptive (p₀ mixture) × per-tensor axis.
   METHOD_COMPARISON noted per-module selection as unmeasured — this is the
   natural joint experiment on a real checkpoint.

7. **Pad-to-tax0** (next experiment)  
   Cost of padding N → N' (nearest tax-0 multiple) vs tax saved on the
   padded packing — may beat stock dims on density for some layers.

---

## What NOT to do next

- Flat-repack a bigger unstructured BitNet-style model expecting new 1-D rates.
- Expect phase-offset / Collatz miracles at scale for fixed-block formats.
- Convert a 70B full-precision model just to re-measure 1.585 bpw.

## What TO do next (ordered)

1. On a **real** ternary/near-ternary checkpoint: per-tensor axis choice +
   byte ledger (exact). Prefer 7B–13B with large rectangular MLPs.
2. Any layer with real Kronecker / low-rank / grouped structure → factor pack.
3. Pad-to-tax0 and GEMM-aligned frame search on those tensors.
4. Only then full-model memory / load-time if per-tensor gains look material.

---

## Pad-to-tax0 (ran)

```bash
python3 repack/pad_to_tax0.py run
```

**Hypothesis:** pad mode length `L → L'` (multiple of a tax-0 `frame_q`) to
unlock the tiling grammar; compare framed bytes on `L'` vs `fmt_41_65` on `L`.

**Result:** **9/10** probed LLM mode lengths have a candidate that beats
`fmt_41_65` on the unpadded length, often with tiny pads:

| L | pad | frame | Δ bytes vs 41_65 |
|---|-----|-------|------------------|
| 1024 | 8 | (19,5) q=24 | −9 |
| 4096 | 8 | (19,5) q=24 | −44 |
| 11008 | 1 | (41,41,19) q=101 | −18 |
| 13824 | 0 | (19,5) q=24 | −155 |
| 22016 | 2 | (41,41,19) q=101 | −36 |

So the “stock dims miss the grammar” problem is often fixable with **1–11
pad trits** (or already exact). Caveat: wins are vs fixed `fmt_41_65` fiber
packing; validate on real tensors before claiming checkpoint-level savings.
Full JSON: `repack/pad_to_tax0_results.json`.
