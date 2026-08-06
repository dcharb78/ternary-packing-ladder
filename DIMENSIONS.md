# Dimensions — multi-track map

*Not a single-track plan. Keep dimensions separate; show where they touch
without collapsing them. Concurrent / all-at-once relationships.*

This note is the durable map for the geometry opened by the packing work.
Engineering harvest lives mainly in **static packing**; the other five tracks
remain live as coordinate system, dynamics, dual packing, hierarchy, and
training economics.

Related measurements:
[`PATTERN_OF_PATTERNS.md`](PATTERN_OF_PATTERNS.md) ·
[`PACKING_PIPELINE.md`](PACKING_PIPELINE.md) ·
[`BETTER_DENSITY.md`](BETTER_DENSITY.md) ·
[`PACKET_SEAM.md`](PACKET_SEAM.md) ·
[`LARGE_SCALE.md`](LARGE_SCALE.md) ·
[`HARMONIC.md`](HARMONIC.md) ·
[`COLLATZ_BRIDGE.md`](COLLATZ_BRIDGE.md) ·
[`ORGANIZING_PRINCIPLES.md`](ORGANIZING_PRINCIPLES.md) ·
[`NEXT_PROBES.md`](NEXT_PROBES.md) ·
[`BASE_P.md`](BASE_P.md) ·
[`FARTHER.md`](FARTHER.md)

### Latest probe results

Dynamical / multi-scale / base-p cadence run:
[`NEXT_PROBES.md`](NEXT_PROBES.md) ·
[`repack/next_probes_results.json`](repack/next_probes_results.json) ·
[`BASE_P.md`](BASE_P.md) (\(p=3,5,7,11\) locus + naïve-vs-block).

**Farther board** ([`FARTHER.md`](FARTHER.md) ·
[`repack/farther_probes.py`](repack/farther_probes.py)): many underexplored
angles after 665. **Geometric win:** flat Law-B-sum **1277** (=665+2×306,
253 B/block) beats hybrid665 on BitNet (**−0.66 MB**). Entropy adaptive also
beats 665 when p(0) elevates (different family). Most other ideas null /
micro. See scoreboard in FARTHER.

**Measured stop signal (ternary):** static leftovers closed at 665 for the
*prior* frontier; farther scan lifts geometric candidate to **1277-flat**
(probe-only; packer flag not wired yet). Dynamical/multi-scale null as *new*
levers. **Base-p:** same three loci for every \(p\); p=5 “0” seeds =
deficit-near-0 (CF **146**, **4004**, …); theory flat-block beats naïve
(p=5 Q=**643**, p=7 Q=**571**) — codec interest only, not a ternary lever.

---

## Claim tags (how to read status)

| Tag | Meaning |
|-----|---------|
| `measured` | Byte / tax / BitNet / synthetic numbers in this repo |
| `applies_operationally` | Changes packer choice or design-time width prior |
| `applies_as_language` | Faithful rename / coordinate; does not by itself densify |
| `speculative` | Potential; not yet probed or only sketched |
| `false_identification` | Named equivalence that measurement rejects |
| `does_not_apply` | Claim fails as stated for packing bytes |

**Number correction:** the informal write-up said “−1 bit per 306.” The measured
rung-block gap is **−1 byte per 306 trits** (61 vs 62 under `fmt_5_8` ceil-on-block;
≈0.33% vs the 0.2 B/trit flat baseline). Document **bytes**, not bits.

**Concurrent ranking update:** pattern-of-patterns showed flat
`fmt_665_1055` (132 B / 665) beats hybrid 306. Static packing is **not**
“freeze at 0.33% / 306” anymore. Hybrid 306 remains a valid prior lever;
665-flat is the current measured density frontier.

---

## 1. Static packing dimension

**Status:** nearly closed — **close after** final docs + one larger absolute
measurement. Do not keep hunting static density as the main program.

### Concepts that survived

- Three (now four) ledger families selected by length grammar: flat `fmt_5_8`,
  fiber-41, rung-block `fmt_306_485`, Law-B-sum flat `fmt_665_1055`.
- Mild reshape scored by actual bytes of the chosen ledger; axis choice as
  free orientation.
- Flat beats nested (`never_nested` for decode-critical).
- Odd rungs admit typed-sheet / complement descriptions; even frame lengths
  (especially 306) live in CF / Law B composition — not twin sheets.
- Law B is useful as a **naming / container** language that points to larger
  usable flat blocks; chiral multi-part frames are **not** the density win.

### What we measured

| Ledger | B / quantum | B/trit | BitNet-2B vs flat `5_8` | vs hybrid 306 |
|--------|------------:|-------:|-----------------------:|--------------:|
| flat `fmt_5_8` | 1/5 | 0.200000 | 0 | +1.36 MB |
| hybrid / `fmt_306_485` | **61/306** | 0.199346 | **−1.36 MB** | 0 |
| **flat `fmt_665_1055`** | **132/665** | **0.198496** | **−3.26 MB** | **−1.89 MB** |
| **flat 1277** (665+2×306) | **253/1277** | **0.198121** | **−3.92 MB** | **−0.66 MB vs 665** |
| chiral 665-frame | 133/665 | ≈0.200 | −0.09 MB | loses |
| 486-frame / fiber-41 | — | worse | big losses | loses |

Farther geometric candidate (probe-only): see [`FARTHER.md`](FARTHER.md).
665 remains the shipped optional frontier (`--hybrid665`); 1277 not wired yet.

Sources: [`BETTER_DENSITY.md`](BETTER_DENSITY.md),
[`PATTERN_OF_PATTERNS.md`](PATTERN_OF_PATTERNS.md),
[`repack/pattern_of_patterns_results.json`](repack/pattern_of_patterns_results.json).
Oracle: F (`hybrid_665_flat`) on **all 210** BitNet tensors. Synthetic 7B/13B/70B:
F vs B ≈ **−5.5 / −10.8 / −65.7 MB** (same law, larger absolute).

Packer: optional `--hybrid` (306) and `--hybrid665` (665-flat). Default remains
exact-quantum three-ledger (flat knee on stock BitNet widths).

### Next CF flat block (15601) — theory only

Next surplus rung 15601 packs as `ceil(bit_length(3^15601)/8) = 3091` B
(≈0.19813 B/trit). Computing `3**15601` for **theory bytes is cheap**
(~24 727-bit integer). Using it as an **operational** flat block format is
probably not practical: huge bigint encode/decode per block, rare exact
alignment on real tensors, and only a modest further density step past 665.
Do not treat 15601 as the next packing sprint unless a greenfield width
design forces the quantum.

### Evaluation (static)

| Claim | Tag |
|-------|-----|
| 665-flat beats hybrid 306 and flat `5_8` on BitNet + synthetics | `measured` / `applies_operationally` |
| Flat 1277 (665+2×306) beats 665 on BitNet (farther probe) | `measured` (probe-only; see [`FARTHER.md`](FARTHER.md)) |
| Hybrid 306 ≈ −1 **byte**/306 ≈ 0.33% vs 0.2 baseline | `measured` / `applies_operationally` |
| Chiral 486/665 frames denser than flat sum | `does_not_apply` |
| Fiber-41 as primary density lever on BitNet | `does_not_apply` |
| Digit nesting / “preserve 0→1” densifies | `false_identification` |
| Freeze forever at 306 / 0.33% | **stale** — superseded by 665-flat frontier |
| Close static after docs + larger absolute measure | cadence (below) |

### Touches (without flattening)

Engineering projection of the whole structure onto **final byte counts**.
Uses circle coordinates and Law B names; does not exhaust triskelion,
sphere-and-cube, or multi-scale questions.

---

## 2. Circle / phase dimension

**Status:** coordinate system — not an experiment by itself.

### Core object

The circle \(\mathbb{R}/\mathbb{Z}\) with rotation by \(\alpha = \log_2 3\).
Every length \(Q\) sits at \(\{Q\alpha\}\). Surplus (near 1), deficit (near 0),
and the self-complementary locus (near ½) are distinguished places.

### Concepts / measured language

- Tax, axis choice, and zero-tax conditions are statements about fractional
  parts and how they multiply ([`HARMONIC.md`](HARMONIC.md)).
- Additive bit-length formulas are the integer chart of the same geometry.
- Complements \(\phi+\psi\approx 1\) explain classic Law B assemblies
  (e.g. 53+306 → 665 grammar; 41+19 → 486-frame grammar).
- BitNet’s 2560-square sits near the self-complementary point
  (`{2560α}≈0.504`) — architectural accident visible only on the circle.

### Open (coordinate, not density hunts)

- Is the phase distribution across real models random, or do training /
  hardware pressures create attractors?
- Soft architectural prior on phase without harming loss?
- How phases move under progressive quantization, sparsity, or shape morphing?

### Evaluation

| Claim | Tag |
|-------|-----|
| Tax = flat chart of `{Qα}` | `measured` / `applies_as_language` |
| Complements name Law B assemblies | `applies_as_language` (and names 665 container) |
| Living on the circle alone densifies BitNet past flat/`665` | `does_not_apply` |
| Phase as soft train prior | `speculative` |

### Touches

Supplies the **language** in which the other dimensions are written. Static
packing reads integer ledgers off this circle; triskelion moves on it;
multi-scale is many circles.

---

## 3. Triskelion / transition dimension

**Status:** open — dynamical.

### Image

Three legs around a common center: surplus, ½, deficit. Rotation is
amplification, training, quantization, or architectural change — process,
not a static assignment.

### Why 0.33% (and even ~0.75% at 665) felt “too small”

Those numbers are density differences between **frozen snapshots**. The
triskelion is the **motion between** snapshots. Interesting cost may be
rotation itself — moving a network from one privileged phase regime to
another.

### Concepts / experiments (not yet measured here)

- Seed → amplify → harmonize as process
  ([`COLLATZ_BRIDGE.md`](COLLATZ_BRIDGE.md) process language).
- Privileged phase rotated deliberately across stages or network parts.
- Track phase cloud under quantization / sparsification.
- Cost elementary transition operators (reshape, re-orient, re-tile)
  geometrically, not only by final density.

### Lean contact

The Collatz/UFRF repo does **not** use the word “triskelion.” The packing
side reads the formalized **frozen frames** (Trinity, kissing→13, flip at ½)
as capable of rotation — interpretive stance, **not new Lean work in this
repo**.

### Evaluation

| Claim | Tag |
|-------|-----|
| Dissatisfaction with pure density → dynamics matter | `applies_as_language` |
| Transition cost measurable in packing/quantize/train | `speculative` |
| “Triskelion” as a Lean theorem name | **not present** — do not invent |

### Touches

Dynamics **on** the circle. Static packing freezes a frame; this dimension
asks about the path between frames.

---

## 4. Sphere-and-cube / node-formation dimension

**Status:** interpretive / mostly conceptual.

### Intuition

Each rung is a candidate center. Around it sit 12 others (kissing number in
3-D). A stable **node** appears when spherical (3-related) packing and cubic /
power-of-2 packing close simultaneously. Local figure: 13. Same 2-vs-3 tension
that generates \(\alpha=\log_2 3\).

### Explorations (not closed)

- Local neighborhoods of known rungs for 12-fold / 13-fold signatures.
- Node formation = surplus-phase ∧ power-of-2 alignment?
- Typed sheets (odd) vs frame construction (306…) as two ways dual packing
  closes.

### Evaluation

| Claim | Tag |
|-------|-----|
| Kissing / 13 appear in UFRF Lean (see table below) | `measured` (in Collatz repo) |
| That skeleton explains privileged packing lengths beyond CF | `applies_as_language` / partly `speculative` |
| UFRF “sphere packing” ≡ ternary weight packing | `false_identification` (different object; see [`COLLATZ_BRIDGE.md`](COLLATZ_BRIDGE.md)) |

### Touches

Discrete-geometric reason why certain **points on the circle** are centers.
Does not replace CF rung measurement for bytes.

---

## 5. Multi-scale / concurrent-scales dimension

**Status:** open / interpretive.

### Source

Collatz/UFRF concurrent scales, breathing scores, recursive towers — and
packing’s own multi-mode phase cloud ([`HARMONIC.md`](HARMONIC.md)).

### Idea

A model is not one circle but phases at several hierarchical levels.
Alignment or controlled misalignment across scales may matter for stability,
memory, or trainability.

### Caveat from Collatz negatives

Breathing score / coarse-scale compensation as **predictors** were falsified
in Collatz Q4–Q7 ([`COLLATZ_BRIDGE.md`](COLLATZ_BRIDGE.md)). Keep as cloud
*summary* language; do not import as packing certificates.

### Possible probes

- Multi-scale phase-coherence score vs training metrics.
- Law C hierarchical digits / recursive operators as multi-scale phase
  extractions (language only until probed).
- Resonance when phases at different scales approach complements together.

### Evaluation

| Claim | Tag |
|-------|-----|
| Multi-mode phase cloud on BitNet (3 unique modes) | `measured` |
| Concurrent scales as packing certificate | `does_not_apply` / Collatz-negative |
| Coherence score ↔ trainability | `speculative` |
| Digit nesting densifies via “multi-scale” | `false_identification` |

### Touches

Lifts circle / triskelion questions to a **hierarchy of circles**. Prevents
collapse into a single-scale story.

---

## 6. Training and scaling dimension

**Status:** speculative — inferred potentials.

### Potentials (not claims)

If widths are chosen with phase or frame structure in mind, second-order
effects may appear in:

- memory traffic / fragmentation during training,
- checkpoint size and streaming cost,
- architecture search optimizing loss against **packed bytes**,
- progressive quantization schedules that respect phase transitions.

### Near-term tests (only after static close + small dynamical probes)

- Two small models identical except packing-phase of key widths.
- Soft packing-phase term in NAS reward.
- Statistical survey of phase distribution in large open models.

Greenfield priors already measured as **packing** priors (×306 / ×665 widths,
surplus×align64) — training economics remain unproven.

### Evaluation

| Claim | Tag |
|-------|-----|
| Design-time ×306 / ×665 improves packed density | `applies_operationally` (packing) |
| Same choice improves train memory / loss | `speculative` |
| NAS with packed-byte reward | `speculative` |

### Touches

Asks whether structure can act **before** the model is frozen. Depends on
circle + static ledgers; must not swallow the dynamical tracks.

---

## How the dimensions relate (without flattening)

- The **circle** is the coordinate system.
- The **triskelion** is the dynamics on that coordinate system.
- **Sphere-and-cube** is a discrete-geometric explanation for why certain
  points on the circle are centers.
- **Multi-scale** lifts the same questions to a hierarchy of circles.
- **Static packing** is the engineering projection onto final byte counts
  (shipped frontier: **665-flat**; probe candidate: **1277-flat**; prior:
  hybrid 306).
- **Training/scaling** asks whether the structure can act before freeze.

Do **not** collapse the program into “just use 665” or “just use 1277.”
That closes (or nearly closes) only the static track. Dynamical,
dual-packing, and multi-scale questions remain open on their own legs.

```text
                    multi-scale (many circles)
                           ↑
  sphere-and-cube  →  circle / phase  ←  triskelion (motion)
         ↓                 ↓
      node centers    static packing (bytes)
                           ↓
                  training / scaling (before freeze)
```

---

## Lean / UFRF formalized skeleton ↔ packing language

Paths under `/Users/dcharb/Documents/collatz/UFRF0-Lean4-Explore-v2/UFRF/`
(verified present locally). Cite what exists; do not invent theorems.

| Concept (packing write-up) | Lean module | What it says (as formalized) | Packing reading |
|----------------------------|-------------|------------------------------|-----------------|
| Three-fold seed / poles | `Trinity.lean` | Conserved triplet `{-½, 0, +½}`, sum = 0 — sole starting definition | Three distinguished regimes (deficit / mediator / surplus) on the chart |
| Sphere packing, 12 around 1 | `KissingEigen.lean`, `KissingHierarchy.lean` | `K(3)=12` as eigenstructure; `kissing_plus_center_is_cycle`: K(3)+1 = 13 | Local spherical arrangement around a center |
| Center + 12 = node | `Structure13.lean`, `FibonacciKissing.lean` | 13 as projective plane \(a^2+a+1=13\); uniqueness of balanced \(a=3\) | Stable **node** = completed local figure |
| Critical point at ½ | `BreathingCycle.lean` (`flip_at_half`), `Manifold.lean` | Flip at 6.5 → \(6.5/13 = ½\) | Self-complementary locus on \(\mathbb{R}/\mathbb{Z}\) (e.g. BitNet 2560) |
| Rotational / cyclic structure | Breathing / phase / hierarchy modules | 13-position cycle, concurrent scales, recursive towers | Frozen frames of a **triskelion** — interpretive; word “triskelion” not in Lean |

**Map to packing:**

- **Rungs** = candidate centers.
- **12-around-1** = local spherical arrangement that wants to form around a center.
- A stable **node** appears only when that arrangement also satisfies cubic /
  power-of-2 constraints (same 2-vs-3 tension as \(\alpha\)).
- **Triskelion** = rotational motion moving one 13-center configuration into
  another — the dynamical picture when a frozen density % felt too small.

The Lean development formalizes the **static skeleton** (Trinity → kissing →
13 → critical flip at ½). Open work on the packing side is to treat that
skeleton as capable of **rotation and transition**, and to ask whether those
transitions have measurable cost in packing, quantization, or training — not
to re-prove the skeleton here.

---

## Suggested cadence

1. **Close static packing** with documentation (this map + frontier notes) and
   one larger-model absolute measurement (`ledger_packer.py bitnet --ckpt …
   --hybrid665` when a >2B ternary ckpt exists; synthetic suites already
   amplify). **Status:** leftovers confirmed in [`NEXT_PROBES.md`](NEXT_PROBES.md);
   stop draining attention into further static density hunting.
2. **Small dynamical probes** — phase cloud, transition operators, snap/rotation
   proxy ([`repack/next_probes.py`](repack/next_probes.py)). Read nulls honestly.
3. Keep **sphere-and-cube** and **multi-scale** as interpretive frames while
   reading data from those probes (do not expand into theory hunts first).
4. **Base-p map** (parallel, not a ternary lever): same circle loci for
   \(\alpha_p=\log_2 p\) — [`BASE_P.md`](BASE_P.md). Actionable only as
   greenfield non-ternary codec interest.
5. **Only if** a clear anomaly or positive signal appears — open formal
   experiments in architecture search or multi-scale coherence.

The geometry is multi-dimensional. Next steps should stay multi-dimensional
with it.

---

## Practical baseline (static only)

| Priority | Lever | Flag / prior |
|----------|-------|----------------|
| **Farther geometric candidate** | flat Law-B-sum 253/1277 | probe-only ([`FARTHER.md`](FARTHER.md)); not wired |
| **Shipped density frontier** | flat Law-B-sum blocks 132/665 | `--hybrid665` / `fmt_665_1055` |
| Prior frontier | rung-block 61/306 | `--hybrid` / `fmt_306_485` |
| Default on stock BitNet | flat `fmt_5_8` | no flag (no ×306/×665 modes) |
| Parallel (entropy family) | adaptive when p(0)↑ | [`FARTHER.md`](FARTHER.md) C |
| Inferior for density | chiral frames, fiber-41 | controls only |

Closing the static dimension means: document, measure once at larger absolute
scale, then **stop draining attention** into further static density hunting —
not that 665 (or probe 1277) answers the other five dimensions.
