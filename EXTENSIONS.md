# Kronecker / Tax-Graph Extensions — evaluation

*Working prototypes for the five creative extensions of the ternary packing
ladder. Hard rule unchanged: size/tax/pack/unpack verdicts are exact integers;
floats only in display ratios and the pre-transform float front-end.*

## What was built

| # | Idea | Module / bench | Status |
|---|------|----------------|--------|
| 1 | Hierarchical / Kronecker digit decode (Law C) | [`repack/hierarchical_digits.py`](repack/hierarchical_digits.py), **D3** in [`bench/ternary_pack_bench.rs`](bench/ternary_pack_bench.rs) | Measured |
| 2 | Tax graph + zero-tax frames (Law B) | [`repack/tax_graph.py`](repack/tax_graph.py), [`repack/frame_catalog.json`](repack/frame_catalog.json), [`repack/frame_formats.py`](repack/frame_formats.py) | Measured |
| 3 | Recursive packing operators | [`repack/recursive_pack.py`](repack/recursive_pack.py) | Round-trip + ledger |
| 4 | Structured pre-transform | [`repack/pre_transform_probe.py`](repack/pre_transform_probe.py) | Probe (honest limits) |
| 5 | Collatz-adaptive schedule | [`repack/collatz_schedule.py`](repack/collatz_schedule.py) | Round-trip + tax ledger |

Reproduce:

```bash
cd repack && python3 tax_graph.py catalog && python3 tax_graph.py selftest
python3 hierarchical_digits.py && python3 frame_formats.py
python3 recursive_pack.py && python3 collatz_schedule.py && python3 pre_transform_probe.py
cd .. && rustc -O -C target-cpu=native bench/ternary_pack_bench.rs -o /tmp/tpl && /tmp/tpl
```

---

## 1. Hierarchical digit tables (Law C)

**Claim tested:** decode a 485-bit (306-trit) block by extracting base-`3^41`
digits, then running the existing C2 (÷243 + LUT) path — same density as D2,
hopefully less multi-limb ÷243 traffic.

**Honesty:** a materialised Kronecker LUT over `3^41` is impossible (~3.6×10¹⁹
entries). What ships is *structural* nesting plus an optional 243×243 micro-table
(59 049 entries → 10 trits) for C2 acceleration in Python.

**Measured** (Apple host, `rustc -O -C target-cpu=native`, 2²⁴ trits, exact
dot-product assert before each row):

| row | format | Gtrit/s | bpw |
|-----|--------|---------|-----|
| B | rung (5,8) LUT | 4.21 | 1.6000 |
| C2 | (41,65) digit | 1.35 | 1.5854 |
| D2 | (306,485) flat ÷243 | 0.24 | 1.5850 |
| **D3** | **(306,485) hierarchical ÷3⁴¹→C2** | **0.36** | **1.5850** |
| E | 486-frame (tax 1) | 1.36 | 1.5882 |

**Verdict:** D3 is ~1.5× D2 at identical 1.5850 bpw — hierarchical nesting
*helps*, but does **not** approach C2/E rates. The ÷`3^41` multi-limb work
(factored as ÷`3^20`×÷`3^21` to avoid u128 shift overflow) still dominates.
The practical Pareto for speed+density remains the **486-frame** (Law B), not
deeper flat-rung nesting. Python hierarchical unpack round-trips exactly
against `pack_ladder` serial unpack.

---

## 2. Tax graph + zero-tax frames (Law B)

**Primitive:** `tax(parts) = Σ bits(q) − bits(Σ q)` with `bits(Q)=(3^Q).bit_length()`.

**Results** (`python3 tax_graph.py catalog`, max_trits=2000, max_tax=1):

- Recovered documented **486** (tax 1) and **665** (tax 0).
- **2952** assemblies with tax ∈ {0,1}; **870** tax-0; **65** novel tax-0
  assemblies that involve a deficit piece (19 or 53) and are not the hand 665.
- Examples: `(41,19)` → 60 trits / 96 bits tax 0; `(306,53)` → 359/570 tax 0;
  `(306,306,53)` → 665/1055 tax 0.
- Register-friendly (every part ≤128 bits): hundreds of assemblies (41- and
  5-based); 306-based frames need 256-bit or multi-limb paths.

**665-frame** pack/unpack in `frame_formats.py`: exact RT; full-block
1055/665 ≈ 1.5865 bpw (tax 0 vs flat).

**Verdict:** systematising Law B works. The hand examples are not isolated —
deficit/surplus cancellation generates a **library** of zero-tax frames. For
parallel decode, prefer assemblies whose parts fit in u128 (e.g. 7×41+19)
even when they pay tax 1.

---

## 3. Recursive packing operators

`P_2 = 8×5+1` (digit nesting), `P_3 = 7×41+19` (486-frame composition).
Exact tax ledgers: P_3 pays **1 bit/frame** vs flat 306 (67 vs 61 bytes per
306 trits with byte rounding). Round-trips pass.

**Verdict:** elegant mirror of the CF recurrence and Law C/B; cache-friendly
blocked layout coincides with the already-known 486-frame. No density win
over flat rungs — the tax correction is exactly Law B.

---

## 4. Structured pre-transform probe

Walsh–Hadamard Kronecker factors `(4,4,4)` → 64×64 block; absmean ternarize;
compare pack sizes to baseline ternarize on the same Gaussian matrix.

**Sample** (seed 0): H₀≈1.5824, H₁≈1.5829; `Δstream=0`, `Δ5_8=0` bytes on
this draw. Phase-slack proxy for equal 41-blocks is a pure Law-B identity
(independent of the weight values).

**Verdict:** as predicted, this is **joint design**, not container magic. It
does not beat log₂3 on a fixed trit stream. Useful as a harness if someone
wants to search transforms that *change* the ternary word toward
zero-tax-friendly block statistics; not a packing-format improvement by
itself.

---

## 5. Collatz-adaptive schedule

Phase from exact `C_i = (3^i).bit_length()`; descending windows prefer the
largest tax-0 catalog frame that fits; otherwise atomic surplus rungs.
Kronecker product of short schedule segments enumerates schedule families.
Exact RT; per-frame tax ledger identity holds; global tax may exceed the sum
of per-frame taxes because of inter-frame seams (Law A atomicity).

**Verdict:** low implementation cost and philosophically continuous with the
project. On uniform random trits it mostly emits atomic rungs (phase is
schedule-driven, not data-driven). Likely more interesting when wired to the
stream coder’s carry/pending state.

---

## Ranking after measurement (update)

1. **Tax graph / frame catalog** — clear win: infinite family from Law B.
2. **486-style frames for decode** — still the speed/density knee (bench E).
3. **Hierarchical D3** — real but modest speedup on flat 306; keep as decode
   option, not a replacement for frames.
4. **Recursive P_n** — same object as (2)/(3) with a nice recurrence API.
5. **Collatz schedule** — correct plumbing; needs a richer phase signal.
6. **Pre-transform** — research harness only; does not amplify the ladder
   without changing the weights.

The “single recursive data structure” that is simultaneously a convergent,
a zero-tax assembly, a Kronecker-hierarchical table, and a Collatz-descending
schedule is approximately **the tax-graph catalog + hierarchical digit
decode of its parts** — with the Collatz schedule selecting among catalog
frames. That stack is now runnable in this repo.

---

# Phase 2 — Multi-linear / tensor-axis packing

*The 1-D ladder treats weights as a flat trit stream. Phase 2 asks what
happens when the same arithmetic (irrational rotation of log₂3, chiral tax,
hierarchical digits) lives on the tensor axes themselves.*

## Core claim under test

Fractional parts `{Q log₂ 3}` are independent per mode. Tax becomes a
multi-linear form; cancellation can happen **across axes**, not only along
a line. Exact bridge (no floats):

```
tax_rows(q0, q1) = q0 * bits(q1) − bits(q0·q1)   # = 1-D tax of (q1,) * q0
tax_cols(q0, q1) = q1 * bits(q0) − bits(q0·q1)
```

The 1-D tax graph is the **rank-1 slice** of this form.

## What was built

| # | Direction | Module | Status |
|---|-----------|--------|--------|
| 1 | Tax as a tensor | [`repack/tax_tensor.py`](repack/tax_tensor.py), `tax_tensor_catalog.json` | Measured |
| 2 | Mode-wise phase alignment | [`repack/mode_pack.py`](repack/mode_pack.py) | Measured |
| 3 | Kronecker-compatible pack | [`repack/kronecker_tensor_pack.py`](repack/kronecker_tensor_pack.py) | Measured |
| 4 | Hierarchical decode without flatten | [`repack/tensor_hierarchical.py`](repack/tensor_hierarchical.py) | RT |
| 5 | Per-mode adaptive schedule | [`repack/mode_schedule.py`](repack/mode_schedule.py) | RT |
| 6 | Packing map as intertwiner | note below | Doc only |

```bash
cd repack && python3 tax_tensor.py catalog && python3 tax_tensor.py selftest
python3 mode_pack.py && python3 kronecker_tensor_pack.py
python3 tensor_hierarchical.py && python3 mode_schedule.py
```

---

### 2.1 Tax as a tensor

Enumerated mode block sizes → **192** frames with best-axis tax ≤ 1
(area ≤ 50k); **147** tax-0; **100** asymmetric tax-0 (`tax_rows ≠ tax_cols`).
Rank-1 identity verified: `tax_rows(k,q) == split_tax((q,)*k)`.

Examples: `tax_rows(7,41)=0` (seven row-containers of 41); `tile_41×5`
best-axis tax 0; axis choice matters whenever `tax_rows ≠ tax_cols`.

**Verdict:** the multi-linear form is real and exactly computable. Pure
geometry already tells you **which axis to pack along**. That is new
relative to flat 1-D packing.

### 2.2 Mode-wise pack + phase offsets

Row-flat and row-framed pack/unpack round-trip. For fixed frame parts,
**cyclic phase offsets do not change byte size** (length-determined
containers) — they only permute seams. The win is **axis selection** via
`tax_rows` vs `tax_cols`, not offset search.

Example `(7,41)`: `tax_rows=0`, row-flat bytes = flat `fmt_41_65` bytes (63).

**Verdict:** mode-wise packing helps when the tax form is asymmetric; phase
offsets are a wash for fixed-block formats (may matter for streaming
coders with carry state — not measured here).

### 2.3 Kronecker-respecting pack

For exact ternary `A⊗B`, packing factors separately vs flattening `W`:
`16×16 ⊗ 16×16` → factor **104** bytes vs flat **13108** bytes
(`fmt_5_8`). Structure-aware packing wins whenever the model has (or
approximates) factor structure — this is storage of the factors, not a
claim about unstructured dense layers.

**Verdict:** highest practical upside in Phase 2 **when structure exists**.
Does not help opaque dense tensors unless a factorisation is maintained.

### 2.4 Non-flattening hierarchical decode

Per-row `fmt_41_65` hierarchical digit unpack matches flat per-row unpack;
digit tensors `(M, n_full, 8)` expose concurrent Law-C state along rows
without a global flatten.

**Verdict:** correct lift of Law C onto matrices; enables axis-parallel
decode. Density unchanged vs packing each row independently.

### 2.5 Per-mode schedule

Phase vector from `C_i` per mode; `choose_axis` picks min of `tax_rows` /
`tax_cols` (ties broken by descending flags); framed pack/unpack RT.

**Verdict:** connects Collatz/schedule machinery to the tax form. On
uniform random matrices the signal is mostly the tax form, not Collatz.

### 2.6 Packing map as intertwiner (note)

Requiring the map “ternary tensor → packed binary tensor” to commute with
continued-fraction / Kronecker recurrences is an algebraic constraint
language for generating new zero-tax layouts. Phase 2 does not implement a
solver; the runnable stand-in is: **enumerate tax_tensor frames whose
axis layout matches a Kronecker factor tree**.

---

## Ranking after Phase 2

1. **Kronecker-factor packing** — huge when structure is real (measured).
2. **Tax tensor / axis choice** — free exact win on rectangular layouts.
3. **1-D tax graph + 486-frames** — still the dense-unstructured knee.
4. **Mode hierarchical decode** — engineering for parallel decode.
5. **Mode schedule / phase offsets** — plumbing; offsets weak for block formats.
6. **Pre-transform / intertwiner algebra** — research frontier.

**Bottom line:** Phase 1 is the correct 1-D theory. Phase 2 shows the
packing tax is genuinely multi-linear: axis choice and factor structure
are where extra cancellation lives. The torus picture is not mystical —
`tax_rows` / `tax_cols` are its exact integer coordinates.

---

# Scale probe (targeted, not blind scale-up)

See [`SCALE_PROBE.md`](SCALE_PROBE.md) and `repack/scale_probe_results.json`.

Ran exact ledgers on LLM-class rectangles (7B–34B shapes). Headline:

- Rectangular MLPs: free axis win (e.g. 11008×4096 saves 7936 bits / 256 B at
  `fmt_41_65` by packing cols not rows). Squares: zero.
- Kronecker factor pack still ~100× when structure exists.
- Stock dims rarely divisible by novel tax-0 frames → design-toward grammar.
- **Stream vs block tension:** streams want flatten (slack/stream); blocks want
  low-tax axis — opposite advice.

Creative links: torus holonomy = `m·bits(n)−n·bits(m)`; factor tree = decode
tree; BitNet module-p₀ × axis as joint unmeasured experiment.

---

## Real BitNet checkpoint (ran)

See [`CHECKPOINT_PROBE.md`](CHECKPOINT_PROBE.md). On microsoft/bitnet-b1.58-2B-4T
(210 tensors): axis choice saves **169 KB** within fiber-41; pad-to-tax0 ~**26 MB**
vs fiber-41 but still does not beat flat `fmt_5_8`; **0** identical-tile /
Kronecker structure hits. Free levers confirmed; 5-per-byte remains the
unstructured density knee on this artifact.
