# The Ternary Packing Ladder: certified block formats below 1.7 bits/weight

*A note for teams building multiplication-free / extreme-quantization inference
(prepared 2026-08, from the UFRF Collatz parallax project; all power inequalities
below are machine-certified in Lean 4 kernel proofs, standard axioms only).*

## Summary

*Scope: this note addresses packing density and container mathematics only;
it does not claim new end-to-end model quality or fused-kernel performance.*

Ternary weights (w ∈ {−1, 0, +1}) carry log₂3 = 1.58496 bits each. Every packing
of trits into a binary memory system pays a tax against that floor, and the tax
is governed by one arithmetic fact: how powers of 3 interleave with powers of 2.
The best block sizes — the only ones that approach the floor — are the
denominators of the continued-fraction convergents of log₂3. We call them rungs.
The industry's current ~1.6–1.7 bpw formats sit on the first rung. This note
gives the full ladder, three exact structural laws (each verified with exact
integer arithmetic, no floating point at any verdict), a decode strategy that
follows from the ladder's nesting, and measured single-core numbers on an
Apple M5 Max.

## 1. The ladder (each row certified: 3^Q ≤ 2^P, and 2^(P−1) < 3^Q, i.e. tight)

| rung | Q trits | P bits | bits/trit | waste vs floor |
|------|---------|--------|-----------|----------------|
| 1    | 5       | 8      | 1.60000   | 1.5 × 10⁻²     |
| 2    | 41      | 65     | 1.58537   | 4.0 × 10⁻⁴     |
| 3    | 306     | 485    | 1.58497   | 4.8 × 10⁻⁶     |
| 4    | 15601   | 24727  | 1.5849625 | 1.7 × 10⁻⁹     |

Rung 1 is prior art: 3⁵ = 243 ≤ 256 gives 5 trits/byte with a 243-entry lookup
table — the mechanism behind existing ~1.6–1.69 bpw ternary formats (e.g. the
llama.cpp TQ1_0 family). The contribution here is the *rest* of the ladder, the
laws below, and the certified tightness: these four are not merely good block
sizes, they are provably the only Q ≤ 15601 where ⌈Q·log₂3⌉/Q makes a new
record. Between rungs there is nothing — see law A.

The certificates live as kernel-checked lemmas (`3^41 ≤ 2^65`, `2^64 < 3^41`,
etc.) in [`lean/Certificates.lean`](lean/Certificates.lean) (standalone, no imports); they are the
same integers that drive our Collatz descent theorems.

## 2. Three exact laws (verified by exhaustive exact-integer computation)

**A. Rungs are atomic.** Define bits(Q) = (3^Q).bit_length() — the exact minimal
container. For Q = 306, *every* one of the 305 ways to cut the block into two
pieces costs exactly one extra bit: bits(a) + bits(306−a) − bits(306) = 1 for
all 0 < a < 306. Same for Q = 41. A rung is a phase-closure point of the
2-vs-3 rotation; strictly inside it the phase never returns, so every internal
seam pays one comma bit. This is *why* there is no useful format between 1.6
and 1.585: the frontier is a staircase whose steps are the rungs, not a curve.

**B. The split tax is chiral.** Decompositions are not equivalent:
- 306 = 7×41 + 19 → 7×65 + 31 = 486 bits (tax **1** over the flat 485)
- 306 = 5×53 + 41 → 5×85 + 65 = 490 bits (tax **5** — 53 is a rung of the
  *other* ladder, the one where 2^P ≤ 3^Q; those blocks each carry a deficit bit)
- 665 = 2×306 + 53 → 2×485 + 85 = 1055 bits (tax **0** — exact!)

The zero-tax identity is the striking one: 665 is itself a convergent
denominator, but of the deficit family; it assembles from surplus-family rungs
at no cost — the surpluses pay the deficit exactly. Practical consequence:
never allocate a deficit-family block; the same trits fit in the same bits as a
concatenation of surplus rungs, which are smaller and decode in parallel.

**C. Scales nest as digit alphabets.** Each rung is best decoded in the
*previous* rung's alphabet, not trit by trit. A (41,65) block is one 65-bit
integer; extracting base-243 digits (8 divides by 243 + one final trit) yields
5 trits per divide via the rung-1 lookup table — a 5× reduction in divide
operations. Measured effect below: 4.4× throughput. The ladder is not a menu
of alternatives; it is a tower where level k is the digit system of level k+1.

## 3. Measured Pareto (Apple M5 Max, single core, scalar `rustc -O`, 16.7M trits, exact-match verified against reference dot product)

| format                       | bits/trit | Gtrit/s | notes                       |
|------------------------------|-----------|---------|-----------------------------|
| 2-bit shift/mask             | 2.0000    | 5.37    | baseline                    |
| rung (5,8), LUT              | 1.6000    | 3.91    | prior-art class             |
| rung (41,65), trit-serial    | 1.5854    | 0.34    | naive decode                |
| rung (41,65), digit decode   | 1.5854    | **1.48**| law C applied               |
| rung (306,485), digit decode | 1.5850    | 0.24    | multi-limb; not worth it    |
| 486-frame = 7×(41,65)+(19,31)| 1.5882    | 1.40    | law B: 1-bit tax, all-u128  |

Scalar verdict, honestly: rung (5,8) is the Pareto knee, which is presumably
why the industry landed at ~1.6–1.7 bpw. The deeper rungs cost ~2.6× scalar
throughput for ~0.9% density.

But the regime that matters for on-device inference is not scalar-CPU-bound; it
is memory-bandwidth-bound with decode fused into the matmul kernel — exactly
the "decode inside the kernel" architecture. In that regime decode ALU is
cheap and every packed bit is bandwidth: blocks are fully independent
(embarrassingly parallel), the (41,65) digit decode is 8 independent
divide-by-constant ops (compilers lower these to multiply-high), and the
density gain is pure bandwidth savings. On a 35B-weight model, 1.6000 →
1.5854 bpw is ≈ 64 MB; versus a 1.7 bpw stream it is ≈ 480 MB — both recurring
per token in the streamed-weights regime. Whether that pays for the extra
in-kernel ALU on Metal is an empirical question we have not tested; the
benchmark code (single file, no dependencies) is included for anyone who wants
to port it to a fused kernel.

## 4. Two structural remarks from the parent project

**Scales are chart data.** Real formats spend ~0.1–0.17 bpw above their packing
rate on per-block scale factors and metadata. In our framework a quantized
block without its scale is a readout without its chart — meaningless by
itself. The packing rate and the chart overhead are separate ledgers; this
note optimizes only the first. (Chart overhead has its own compression
structure — scales are approximately log-uniform — but that is a different
problem.)

**Why these constants are trustworthy.** The parent project studies the 3n+1
map, which is the minimal multiplication-free computation: multiply-by-3 is
shift-plus-add, the +1 residue accumulates as an exactly-ledgered carry, and
the system's entire behavior is controlled by the same 2-vs-3 phase that
controls this packing problem. The rung inequalities used here are the
project's kernel-certified descent certificates, re-read as container bounds.
The waste of naive 2-bit packing, 2 − log₂3 = 0.41504 bits, is the same
constant as that system's per-step descent rate — one arithmetic fact
(2² − 3 = 1: four states holding three) seen from two sides.

## 5. Results since first writing (all measured, preregistered thresholds)

**Training test (byte-level GPT, enwik8 90/5/5, MLX, 3-seed means):** a
ternary-QAT model at 10.9M params (weights {−1,0,+1}, absmean+STE) reached
1.8522 bpc (sd 0.0013) vs 1.9492 for the fp16 model with the same total
weight-bits — the ternary model beats its bit-matched twin by 0.097 bpc,
and beats same-param fp16 at this training budget. The trained alphabet is
maximum-entropy to three decimals (H = 1.5847 of log2 3 = 1.5850, p0 =
0.330): training fills the ternary alphabet to capacity, so realized bits
= ideal bits. Preregistered support thresholds passed for both ternary and
binary arms; the fp size-curve comparison was declared unreadable at this
budget (undertrained large models) per a pre-run amendment.

**Streaming codec (carry-counting range coder, exact integers):** at 10^6
trits it emits 1.585000 bits/trit — the floor — with 37 bits total slack,
at ANY length (no block containers), finalized-bit schedule tracking
ceil(t·log2 3) within 24 bits, carry buffer never exceeding 2 bytes. On a
real trained checkpoint it beats every block format: 0.792527 of 2-bit,
vs the theoretical log2(3)/2 = 0.792481. For streamed-per-token weights
(the bandwidth-bound regime) the stream codec supersedes the block table;
blocks remain the choice only where random access is required.

**Concurrency addendum (round 2):** the scalar single-stream numbers above
undersell the deep formats. With instruction-level parallelism the
486-frame decodes at 2.17 Gtrit/s on one core (64% of LUT speed, 0.7%
denser) and 11.0 Gtrit/s on 18 cores; 8-way interleaved range streams
reach 3.28 Gtrit/s at the exact floor (1.58498 bpw). Floor-density
storage now decodes at SSD-class rates — the artifact decodes as fast as
it reads. Per-core the LUT retains a 1.6–3.2× edge; the correct summary
is that floor density is no longer speed-prohibitive, not that the LUT
is beaten.

## 6. The public-artifact demonstration (BitNet b1.58 2B-4T)

We repacked the 2,084,044,800 ternary weights of microsoft/bitnet-b1.58-2B-4T
(all 210 packed tensors; every format round-tripped exactly):

| format | size | vs shipped |
|---|---|---|
| shipped 2-bit | 521.0 MB | 1.0000 |
| rung (5,8) | 416.8 MB | 0.8000 |
| stream (floor) | 412.9 MB | 0.7925 |
| adaptive | **405.2 MB** | **0.7776** |

Total: 108–116 MB saved (20.8–22.2%). The stream's total slack over the
exact floor was 840 bytes across 210 tensors — 4 bytes per tensor, the
constant-slack law on industrial data. Census: the model is NOT at the
ternary capacity — p(0) = 0.4221, H = 1.5603 of log2(3) = 1.5850 — which is
why adaptive coding pays +1.87% here (matching our sensitivity curve at
p0 = 0.42), whereas our own QAT checkpoint trained to capacity (p0 = 0.330)
and adaptive gained nothing. Two regimes, both measured. Spin symmetry
holds in both: p(−1)/p(+1) balanced to four decimals. Order structure is
real but small (conditional-entropy savings ≤ 0.002 bits/weight, 2.5×
anisotropic between scan directions): order-0 adaptive is the right
format for this artifact.

## 7. Open question we would genuinely like an answer to

Capability retention at 95% under collapse to the ternary alphabet suggests
the function of a network lives in its sign/sparsity *word*, with magnitudes
recoverable from small per-block data. In the 3n+1 system this exact
statement is a theorem (the orbit value is recoverable from the letter word
plus one constant), and the system obeys hard *composition floors*: any
trajectory that stays subcritical must keep ≥ 2 − log₂3 = 41.5% of its letters
minimal. Is there an analogous floor for ternary networks — a minimum density
of nonzero weights below which retention provably collapses, governed by the
same exchange rate? If quantization-error accumulation across depth has a
partition-sum law (error = Σ over steps of 2-power deviations, as our carry
does), block formats could be chosen to shape it. We have the exact machinery
for such laws and would be glad to compare notes.

## Reproduction

- Benchmark: `exploration/rust_scan/ternary_pack_bench.rs` (single file;
  `rustc -O -C target-cpu=native`; asserts exact dot-product equality for every
  format before reporting).
- Exact split-tax check: ten lines of Python; bits(Q) = `(3**Q).bit_length()`;
  no floats.
- Certificates: `UFRF/ParallaxRungCertificates.lean`; verify with
  `lake env lean` + `#print axioms` (expected: `[propext]` or clean).
