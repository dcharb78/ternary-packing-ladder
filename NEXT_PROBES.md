# Next probes — dynamical, multi-scale, base-p

*DIMENSIONS cadence step 2. Do **not** collapse into more static density
hunting. Fiber-41 = control only. No characters / Stokes reopen.*

Code: [`repack/next_probes.py`](repack/next_probes.py) ·
[`repack/base_p_ladder.py`](repack/base_p_ladder.py)  
Results: [`repack/next_probes_results.json`](repack/next_probes_results.json)  
Map: [`DIMENSIONS.md`](DIMENSIONS.md) · base-p template: [`BASE_P.md`](BASE_P.md)

```bash
PYTHONPATH=repack python3 repack/next_probes.py selftest
PYTHONPATH=repack python3 repack/next_probes.py run
```

---

## What was tested

| # | Probe | Status |
|---|--------|--------|
| 1 | Static leftovers: confirm `fmt_665_1055` / hybrid665; cheaper-than-665?; >2B ckpt? | `measured` |
| 2 | Phase cloud of BitNet unique modes on α-circle | `measured` |
| 3 | Transition ops: identity / best reshape aspect≤16 / axis flip under flat58·306·665 | `measured` → typically `null` |
| 4 | Rotation proxy: snap mode to best surplus in ±64/±128 | `measured` → typically `null` (O4) |
| 5 | Multi-scale: \((\phi_m,\phi_n,\phi_{mn})\) / holonomy × ledger winner | `measured` → typically `null` |
| 6 | Sphere/node: cheap ±1..12 / ±13 neighborhood around rungs | `speculative` / `null` |
| 7 | Base-p: \(p\in\{3,5,7,11\}\) CF rungs, loci, naïve vs flat block | `measured` / map |

---

## 1. Static leftovers (minimal)

- Reconfirm BitNet-2B: **F / hybrid665** ≈ **−3.26 MB** vs flat `5_8`, **−1.89 MB** vs hybrid 306; oracle F on all tensors.
- Next theory surplus **15601** is denser on paper (~0.19813 vs ~0.19850 B/trit) but **not** an operational flat block.
- **No** real >2B ternary checkpoint in-tree; synthetics already amplify.

**Verdict:** close static leftovers; **stop** static density hunting. Tag: `measured`.

---

## 2–4. Dynamical / phase (primary)

**Phase cloud:** BitNet has three unique modes (`640`, `2560`, `6912`) — mid / half, not surplus. Plot-ready table in JSON.

**Transitions:** For each unique shape, cost of identity, best volume-preserving reshape (aspect≤16), and axis flip under flat58 / hybrid306 / hybrid665. Question: cheap path toward surplus without hurting bytes?

**Snap proxy:** Nearest better-surplus length in ±Δ; expect pad/snap often hurts (prior O4).

Tags in JSON: `measured` wins or `null`.

---

## 5. Multi-scale concurrent

Record \((\phi_m,\phi_n,\phi_{mn})\) (or holonomy/tax); score complementary / both-mid / aligned-surplus / half; correlate with ledger winner. On stock BitNet, **665-flat wins everything** — pattern cannot beat that global dominance. Tag: usually `null`.

---

## 6. Sphere / node

One short ±12 / ±13 neighborhood table around known rungs. Interpretive only. Tag: `null` as packing lever.

---

## 7. Base-p (parallel dimension — not a ternary lever)

See [`BASE_P.md`](BASE_P.md). Same rules, new angle: \(\alpha_5=\log_2 5\approx 2.321928\).

For **p=5** results explicitly list:

- strongest **deficit** (near 0) \(Q\)s — alphabet **“0” seeds** (ternary 53/665 analogues)
- strongest **surplus** (near 1) \(Q\)s — flat-block candidates (ternary 5/41/306 analogues)
- any **near-½**

Three loci = right geometric dynamical image for every \(p\); extra training regimes = `speculative`.

---

## Honest verdict

| Track | Actionable now? |
|-------|-----------------|
| Ternary static | Shipped frontier = **665-flat** (−3.26 MB vs flat). Later farther board found probe **1277-flat** (−0.66 MB vs 665; [`FARTHER.md`](FARTHER.md)) — document, do not reopen a density hunt |
| Dynamical / multi-scale on BitNet | **No new lever** — multi-scale `null` (665 wins all 210); reshape micro-hit only under hybrid306; snap 640→612 restates ×306 prior |
| Base-p | **Only** as greenfield non-ternary codec (p=5 block **643**, p=7 block **571**); else conceptual map |

**Stop** draining attention into further static density hunting (1277 is already measured as probe-only). Three loci remain the right geometric image for every \(p\) (`applies_as_language`); training needing more regimes stays `speculative`. Do not reopen fiber-41 / characters / Stokes.
