# Two approaches to extreme ternary quantization: a measured comparison

*Prepared 2026-08. Companion to `TERNARY_PACKING_LADDER.md`. Numbers
attributed to us are one of three grades, marked in context: kernel-certified
(Lean theorems: the rung inequalities), exact-verified (formats asserting
byte-exact round-trip), or measured (CPU/MLX benchmark runs, quoted with
their harness). Numbers attributed to the vendor-class approach are
published claims we did not verify.*

## The two methods

**The vendor-class approach** (represented by Syzygy's Mach line, and more
broadly the BitNet/llama.cpp ternary ecosystem): ship an inference engine;
quantize to ~1.6–1.7 bits per weight (llama.cpp-class 5-trits/byte block
formats; Syzygy's line uses trellis-coded weights); report benchmark retention against a full-precision
teacher; optimize decode by fusing it into the compute kernel. Claims are
end-to-end product numbers (tokens/sec, benchmark scores). For the product
lines, methodology and error bars are not published where we looked; the
academic BitNet literature does publish training methodology.

**Our method**: work from the arithmetic upward. Certify the container
mathematics in a proof kernel (the rung inequalities are Lean theorems);
measure every format with exact round-trip assertion before its throughput
counts; preregister experimental verdicts with thresholds fixed before any
run; decompose aggregates into their concurrent registers (per-module,
per-layer, spin/support/order) before believing them; retract in public
ledger when a claim dies. The output is not an engine — it is formats,
laws, and measurements an engine could adopt.

## Results side by side

| dimension | vendor-class | ours (measured/certified) |
|---|---|---|
| storage density (pure trit coding; product bpw adds ~0.1–0.17 scale/metadata overhead — separate ledgers) | ~1.6–1.7 bpw (claimed, product chart) | **1.5850** stream (floor at every tested length, 1..10⁶, under a preregistered slack bound; 37 bits total slack at 10⁶ synthetic trits; separately: 840 bytes total = 4 B/tensor across BitNet's 210 tensors); **1.5553** adaptive on BitNet-2B (2.084B weights, exact round-trip; this rate exists *because* BitNet runs zero-enriched — it is regime-dependent, not universal) |
| on a public artifact (the ternary payload: 521 MB of the 1.18 GB file; bf16 embeddings excluded) | ships at 2.0000 bpw | **20.8–22.2% smaller**, byte-for-byte reversible |
| decode speed | engine-fused, unpublished internals | scalar LUT 3.40–4.10 Gtrit/s (spread across bench runs/harnesses, not format variance); 486-frame 2.17 (1 core) / 11.0 Gtrit/s (18 cores); 8-way interleaved floor-density streams 3.28 Gtrit/s on 8 cores. All exact-match verified |
| capability retention | "95.0%" vs teacher (twelve benchmarks per their site; methodology not seen by us) | preregistered training verdicts: ternary **beats its bits-paired fp twin** by 0.097 bpc (pairing is loose by +13% once fp-kept tensors are counted, as preregistered; 3 seeds, means 1.8522 sd 0.0013 vs 1.9492 sd 0.0047); alphabet trains to 99.98% of the ternary ceiling. *Different experiment — not directly comparable to theirs, and we say so* |
| container theory | we found none published for the product line; the BitNet literature we know covers training method, not container mathematics | kernel-certified: the rung ladder. Exact-integer verified: packing laws A and B (atomicity, chiral tax); law C's decode speedup is a measurement (4.4×). Computed: λ(W) by float-renormalized DP (λ(1)=1 is exact; algebraic exactification of the rest is planned, not done). Interpretive: λ(1)=1 read as the shared cause of the alignment/flush/carry taxes |
| discoveries en route | — | the two-regime observation (our small QAT models train *to* capacity; the one industrial model we measured runs zero-enriched); the module-phase mixture: BitNet's p₀ = 0.4221 is a mixture across seven module types (module means q_proj 0.468 → v_proj 0.376, each itself layer-varying; sparsity tracks functional role; rank-2 field with a boundary anomaly at the last layer's attention output projection, layer-29 o_proj) |

## What each side actually wins

**They win**: a shipping product; kernel engineering on Metal; claimed
scale (a 2B model served on laptops, per their materials); the ecosystem
position. None of our results
dispute any of that.

**We win**: measured density with certified constants (their own artifact,
20.8% smaller by stream, 22.2% by adaptive, reversible to the byte); exactness (constants that are theorems, formats that assert
their own correctness); explanation (the industry knee's *packing core* is
the (5,8) rung at exactly 1.6000 — product bpw adds ~0.1 of scale/metadata
overhead, a separate ledger — and we can derive why no *fixed-block* packing
rate exists between 1.6 and 1.585 — streaming coders are exempt, which is
exactly why ours reaches the floor); and the discoveries above, which aggregate retention
numbers cannot show — a 95% figure cannot tell you that the
model's query projections and value projections are running different
alphabet regimes.

## What we did not do, stated plainly

We did not build or benchmark a serving engine; our decode numbers are
CPU-core benchmarks, not fused-GPU-kernel numbers; we did not train at
industrial scale; our retention experiment (enwik8 byte-level GPT, 10.9M
params) is a controlled scientific instrument, not a product benchmark; and
the vendor's numbers may all be accurate — we simply have no way to check
them, which is itself the difference between the methods.

## The offer

The two approaches are complements, not competitors. An engine like theirs
adopting formats like ours gets: 0.9% (stream) to 2.8% (adaptive) bandwidth
reduction against the measured (5,8) industry baseline — or ~8.5% against
the vendor's unverified 1.7 product figure — with multi-core scaling bringing the floor formats to single-core-LUT
absolute rates (8 threads for interleave, 18 for frame); per-core the LUT
retains a 1.6–3.2× edge, no matched-core or fused-GPU comparison was run,
and scalar single-stream decode remains slower than the LUT — all stated
plainly; an *unmeasured*
opportunity in per-module format choice (the phase mixture suggests it; we
did not run per-module selection); and constants that come with proofs. The mathematics
is in `TERNARY_PACKING_LADDER.md`. The packing benchmarks are single-file,
dependency-free Rust; the training runs use MLX; the artifact repack needs
the public checkpoint. The claims above trace to dated ledger entries —
including the ones that died on the way here.
