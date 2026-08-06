# Packet / seam framing — packing applicability

*Does the 0→1 prime-context packet language apply to ternary packing?*  
Measured, tagged, not assumed.

Code: [`repack/packet_seam_probe.py`](repack/packet_seam_probe.py)  
Results: [`repack/packet_seam_results.json`](repack/packet_seam_results.json)

Related: [`PACKING_PIPELINE.md`](PACKING_PIPELINE.md), [`ORGANIZING_PRINCIPLES.md`](ORGANIZING_PRINCIPLES.md)

## Final stance

The **0.33%** is real, but it is **exactly** the known rung-block effect:

\[
\frac{61}{306}\approx 0.199346
\quad(-1\text{ byte per 306 trits vs a 0.2 baseline}).
\]

Not a new discovery from seam/packet language or twin centers — the ordinary
frame-ledger advantage whenever a 306-block (or hybrid 306-prefix + remainder)
is available.

| Approach | True density gain | Notes |
|----------|------------------:|-------|
| Align widths to ×306 + frame | ~0.33% | Absolute drop is mostly fewer trits if you also shrink |
| Pad existing widths to ×306 | Hurts | Useful B/trit rises to ~0.225 |
| Hybrid (306 prefix + rem `5_8`) | ~0.33% (~1.3 MB on BitNet) | No width change required |

Projected true density vs 0.2: ~1.3 MB (BitNet-2B measured) · ~4.2 MB (7B
synthetic) · ~51 MB (70B synthetic). See [`LARGE_SCALE.md`](LARGE_SCALE.md).
Vs ceil-62 the relative gap is ~1.6%, same −1 B/306 law. BitNet modes are not
closer to ×306 than chance.

**Keep only two practical extracts:**

1. **Design-time** widths ∣ 306 → frame ledger gives 0.33% for free.
2. **Fixed widths** → optional hybrid prefix packer captures the same small gain.

Everything else (seam language, twin centers, packet geometry) does not beat
ordinary CF-phase + three-ledger rules. No new default packer rules, no live
padding, no theory re-opening. Useful negative result + small positive.

### Verdict (probe summary)

**Mostly language + one small greenfield prior.** Packet chart restates the
frame-divisibility gate and typed odd/even split. It does **not** beat CF phase
as an event predictor. “Preserve every 0→1 step” ≡ Law-C digit nesting is a
**false identification**. See density dissection below for the 61/306 numbers.

## Five dimensions

### 1. Pattern — `(⌊n/p⌋, n mod p)` vs CF phase

Event window Q∈[1,800] (rung hits, strong zero-tax, 41/306/665 aligns).

| Predictor | Precision | Recall | F1 |
|-----------|----------:|-------:|---:|
| CF surplus `{Qα}≥0.9` | **1.00** | 0.69 | **0.82** |
| Seam (local=0, any tested p) | 0.16 | 0.75 | 0.26 |
| Twin sheets (±1) | 0.14 | 0.96 | 0.25 |

**Tag:** seam-as-independent-predictor → `does_not_apply`.  
Seam-as-ledger-quantum (p∣n for p∈{306,665}) → `applies_operationally` (already in packer).

### 2. Recursion — packets of packets / scale transport

CF ladder `5 → 41 → 306 → 15601`:

| Parent | CF child | 2p | p² | err(2p) | err(p²) |
|-------:|---------:|---:|---:|--------:|--------:|
| 5 | 41 | 10 | 25 | 31 | 16 |
| 41 | 306 | 82 | 1681 | 224 | 1375 |
| 306 | 15601 | 612 | 93636 | huge | huge |

**Tag:** `applies_as_language` — local bound p is not global; next child needs
α/CF, not 2p or p².

Recursive rechart does **not** beat typed center for odd-sheet vs even-306:
odd p → odd twins (same parity fact). Even 306 remains a CF rung-block, not a
typed sheet. **Tag:** `does_not_apply` (as a superior explanation).

### 3. Nesting — hierarchy ≠ digit nest

| Claim | Tag |
|-------|-----|
| UFRF “preserve every 0→1 step” ≡ Law-C hierarchical digit nesting | **`false_identification`** |

Correct reading: retain *(cycle, local)* is a **typed chart** discipline.
Digit nesting is a **container composition** choice. Prior pipeline already:
flat beats nested ~92–98% on tax; 306 nest decode ~8× slower than `5_8`.
Keep `never_nested` for inference; allow `fmt_306_485` for archive density
when length ∣ 306.

### 4. Hierarchy — who owns which operator

| Layer | UFRF | Packing owner | Flatten risk |
|-------|------|---------------|--------------|
| C0 | signed ±1; no div/mod chart | none | calling seed “completion” |
| C1 | 1→{1,3} | not `fmt_5_8` | equating seed grammar with CF rung 5 |
| C≥3 odd primes | sheets `2mc±1`, twin seams | odd rungs / complements / fiber-41 | every odd length as sheet |
| Even CF 306 | not a typed sheet | **`fmt_306_485` only** | forcing 306 into twin language (= 0=1 collapse) |

At `n=p`: completion∩seed is a **typed rechart**, not `0=1`. Packing analogue:
exact block boundary (remainder 0) where two ledgers meet — they are not the
same symbol. **Tag:** `applies_as_language`.

### 5. Correct-application filter (claim board)

| ID | Tag |
|----|-----|
| P1 seam predicts events better than CF | `does_not_apply` |
| P1 seam = ledger quantum | `applies_operationally` |
| P2 local bound ≠ global | `applies_as_language` |
| P2 recursion ≫ typed center for odd/even | `does_not_apply` |
| P3 nesting ≡ preserve 0→1 | `false_identification` |
| P4 hierarchy / no flatten | `applies_as_language` |
| P4 typed rechart ≠ 0=1 | `applies_as_language` |
| M twin vs interior (rung/surplus) | `does_not_apply` |
| M BitNet nearest×306 redesign | `applies_operationally` (greenfield caveat) |
| M 0.33% is new seam geometry | `does_not_apply` (restates 61/306) |

## Practical metrics

### Twin seam vs interior

Across primes×k: twin rung-hit rate ≈ **2.5%**, interiors ≈ **2.7%**. Twins are
**not** closer to surplus-1 on average. Twin language remains useful for
*odd-sheet incidence*, not for byte selection.

### Synthetic multiples of 306 / 665

| Shape | Path | Δ vs `5_8` |
|-------|------|----------:|
| 1×306 … 306×306 | `fmt_306_485` | −1 … −62 B |
| ×665 | `flat58` (tie) | 0 |
| 2560² | `flat58` | 0 |

### BitNet-2B redesign → nearest ×306 (not pad)

| | Before | After |
|--|-------:|------:|
| Selected bytes | 416,808,960 | 398,705,760 |
| Trits | 2,084,044,800 | 2,000,064,960 |
| Bytes / trit | 0.200000 | **0.199346** |
| Frame path count | 0 | **210 / 210** |

Density Δ ≈ **−0.00065 B/trit** (~0.33%) — exactly the known rung-block gap.
Widths move (e.g. **2560 → 2448**). Absolute byte drop is mostly fewer trits.

**Operational rule:** if you *design* a ternary width, prefer a multiple of 306
(2448 or 2754 near BitNet’s 2560) so the packer can take `fmt_306_485`. Do not
pad existing checkpoints. Capacity/accuracy is a training question, not a
packing theorem.

### Density signal dissection (~0.33%)

Treat the 0.200 → 0.199346 figure seriously — then separate confounders.

#### Exact theory

| Packing | Bytes on 306 trits | B/trit |
|---------|-------------------:|-------:|
| `fmt_306_485` | **61** | **61/306 ≈ 0.199346** |
| `fmt_5_8` ceil on that block | 62 | 62/306 ≈ 0.202614 |
| `fmt_5_8` on exact ÷5 lengths | n/5 | **0.200000** |

Reported after-density **is** 61/306. Baseline 0.200 is BitNet’s ÷5 lengths under
flat `5_8`, not the 62/306 ceil-on-block figure. Gap vs 0.2 ≈ **0.327%**; gap vs
ceil-block = **1 B / 306 ≈ 1.61%** of the 62-byte block. Same −1 B/306 already
measured in the nesting / rung-block test.

#### Confounder split (nearest×306 redesign)

| | Bytes |
|--|------:|
| Absolute Δ selected | −18,103,200 |
| From fewer trits (capacity shrink) | −16,795,968 (**92.8%**) |
| From true density (61/306 vs 0.2) | −1,307,232 (**7.2%**) |

#### Counterfactuals A–D (full 210 tensors)

| | Trits | Selected bytes | B/trit | Notes |
|--|------:|---------------:|-------:|-------|
| Baseline | 2,084,044,800 | 416,808,960 | 0.200000 | flat `5_8` |
| **A** redesign DOWN (≤×306) | 1,932,647,040 | 385,266,240 | **0.199346** | 640→612, 2560→2448, 6912→6732 |
| **Nearest** (tie→down) | 2,000,064,960 | 398,705,760 | **0.199346** | 6912→7038 (closer up) |
| **B** redesign UP (≥×306) | 2,351,199,960 | 468,703,260 | **0.199346** | more capacity; same density |
| **C** pad UP | useful 2.08B / pad 2.35B | 468,703,260 | **0.2249 useful** | hurts; do not pad |
| **D** hybrid prefix+`5_8` rem | 2,084,044,800 (fixed) | 415,446,930 | 0.19934645 | −1.36 MB vs flat; no redesign |

A/B/nearest all land on **exactly 61/306** once every mode is ÷306. Pad (C)
improves packed-trit density but **wastes** ~12.5% useful density. Hybrid (D)
gets almost the full density win **without** changing widths by allowing
`fmt_306_485` on the divisible prefix (current packer gate requires exact ÷306).

#### Mode proximity to ×306 (N=3 unique modes)

| Mode | Residue | Abs dist | Nearest |
|-----:|--------:|---------:|--------:|
| 640 | 28 | **28** | 612 |
| 2560 | 112 | 112 | 2448 |
| 6912 | 180 | 126 | 7038 |

Mean abs dist ≈ **88.7** (trit-weighted ≈ **115.7**) vs uniform-residue null
E ≈ **76.5**. Modes are **not** systematically near 306 seams (1/3 closer than
null; small-N descriptive only).

#### Scale projection (fully 306-aligned → `fmt_306_485` vs flat `5_8`)

| Class | Save vs 0.2 (~0.327%) | Save vs ceil-62/306 (~1.61%) |
|-------|----------------------:|-----------------------------:|
| BitNet-2B measured | **~1.3 MB** | ~6.5 MB |
| ~2B trits | ~1.2 MB | ~6.2 MB |
| ~7B trits | ~4.4 MB | ~21.8 MB |
| ~70B trits | ~43.6 MB | ~218 MB |

Formula: ≈ `trit_count/1530` B vs 0.2 baseline; ≈ `trit_count/306` B vs
ceil-on-306-blocks.

#### Verdict

**Not a new geometric signal.** Restates known rung-block density (61 vs 62 /
−1 B per 306). Worth a **greenfield width prior** if capacity-neutral; not worth
reopening packet/seam theory; never post-hoc pad. Optional packer tweak: allow
hybrid prefix (D) without exact divisibility — small fixed-width win (~0.33%).

## What changes in the packer

**Nothing required.** Seam-as-quantum and odd/even are already in
[`ledger_packer.py`](repack/ledger_packer.py). Optional doc-only prior: publish
306-multiples alongside surplus×align64 as greenfield width hints.
Optional later: lift the exact-÷306 gate to allow hybrid prefix packing (D).

## What not to do

- Do not reopen characters / Stokes / higher associators on this evidence.
- Do not treat packet nesting as a reason to re-enable digit nesting.
- Do not snap live BitNet tensors to 306 by padding.
- Do not treat 0.199346 as evidence that BitNet modes sit on 306 seams.

## Run

```bash
python3 repack/packet_seam_probe.py selftest
python3 repack/packet_seam_probe.py run
python3 repack/packet_seam_probe.py run --ckpt path/to/model.safetensors
```
