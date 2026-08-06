# Base-p — same geometric machine, odd primes

*Ternary is the smallest interesting case. Replace 3 by odd prime \(p\)
(or \(b\geq 3\)); the machine still runs with \(\alpha_p = \log_2 p\).*

Code: [`repack/base_p_ladder.py`](repack/base_p_ladder.py)  
Results: [`repack/base_p_ladder_results.json`](repack/base_p_ladder_results.json)
(also nested under [`repack/next_probes_results.json`](repack/next_probes_results.json)).

Multi-track map: [`DIMENSIONS.md`](DIMENSIONS.md) · next probes: [`NEXT_PROBES.md`](NEXT_PROBES.md).

---

## Template (what generalizes)

| Object | Ternary (\(p=3\)) | General \(p\) |
|--------|-------------------|---------------|
| Circle | \(\{Q\alpha\}\), \(\alpha=\log_2 3\) | \(\{Q\alpha_p\}\), \(\alpha_p=\log_2 p\) |
| Container bits | \((3^Q).\mathrm{bit\_length()}\) | \((p^Q).\mathrm{bit\_length()}\) |
| CF rungs | dens of convergents of \(\alpha\) | dens of convergents of \(\alpha_p\) |
| Three loci | surplus≈1, ½, deficit≈0 | **identical definitions** on \(\mathbb{R}/\mathbb{Z}\) |
| Law-B style | complements → assemblies (53+306→665) | same language; new part lists |
| Flat large container | `fmt_665_1055` (132 B / 665) | flat multi-limb at a large surplus/deficit quantum |
| Ledger choice | flat vs hybrid vs fiber | naïve digit-byte vs flat block vs nested |
| Operational moves | axis, mild reshape by bytes, surplus prior, **flat beats nested** | same moves |

**What changes with \(p\):** Shannon floor \(\log_2 p\) bits/symbol (bytes: \(\log_2 p/8\)); different rung \(Q\); naïve \(k\) with \(p^k\leq 256\) shrinks (\(p=3\to5\), \(p=5\to3\), \(p=7\to2\)); hardware / SIMD friendliness usually **worse** than 5-in-a-byte ternary.

**Why ternary felt privileged:** smallest odd prime \(>2\); practical 5-trit byte; BitNet-scale sizes; not because the geometry is unique.

---

## Three loci (every \(p\)) — identical definitions

On \(\mathbb{R}/\mathbb{Z}\) with rotation \(\alpha_p\):

| Locus | Definition | Role |
|-------|------------|------|
| **Deficit / “0”** | \(\{Q\alpha_p\}\) very close to **0** | Hungry; just under a power of two. **Natural “0” seeds** for that alphabet |
| **Surplus** | \(\{Q\alpha_p\}\) very close to **1** from below | Flat-block / rung candidates |
| **Self-complementary / ½** | \(\{Q\alpha_p\}\approx\tfrac12\) | Flip / mediator locus |

**Tag:** three regimes remain the right dynamical *image* for every \(p\) at the **geometric** level. Whether **training** dynamics need more regimes is still `speculative`.

### Starting point — “what thinks it is 0”

In the **5-world** (\(\alpha_5=\log_2 5\approx 2.321928\)): lengths with \(\{Q\log_2 5\}\) nearest to 0 are the strong **deficit** rungs — analogues of ternary **53** and **665**. They are the natural “0” seeds for that alphabet. Surplus-near-1 lengths are the ternary-**5/41/306** analogues (flat-block candidates).

---

## Measured rung / locus tables

Run:

```bash
PYTHONPATH=repack python3 repack/base_p_ladder.py run
# or via next_probes
PYTHONPATH=repack python3 repack/next_probes.py run
```

Exact numbers live in JSON. Structure (every \(p\in\{3,5,7,11\}\)):

- CF convergent \(Q\) list (first ~8–10)
- Strongest deficit-near-0 / surplus-near-1 / near-½ among scan+CF
- Naïve \(k\)-per-byte vs flat large block density (theory bytes)

### p=5 explicit (from measured run)

\(\alpha_5 \approx 2.321928\). CF dens: `1, 3, 28, 59, 146, 643, 4004, 8651, …`

| Family | Strong Qs (early) | Role |
|--------|-------------------|------|
| **Deficit ≈0** (“0” seeds) | CF: **146**, **4004**, … · scan near-0: 2718, 2075, 1432, 789, … | Analogues of ternary **53 / 665** |
| **Surplus ≈1** | CF: **3, 59, 643, 8651**, … · scan: 643, 1286, 1929, … | Analogues of ternary **5 / 41 / 306**; **643** is first large flat-block candidate (theory **0.291** B/sym vs naïve **0.333**) |
| **Near ½** | 2002, 2645, 1359, 716, 73, … | Same locus type as BitNet 2560 on \(\alpha_3\) |

Ternary CF control dens for comparison: `1, 2, 5, 12, 41, 53, 306, 665, 15601, …` (surplus 5/41/306; deficit 53/665).

---

## Density: naïve vs flat block (theory)

| \(p\) | Naïve \(k\) | Naïve B/sym | Best surplus/CF block \(Q\) | Block B/sym | Δ vs naïve |
|------:|------------:|------------:|----------------------------:|------------:|-----------:|
| 3 | 5 | 0.200 | 665 known / scan surplus | ≈0.1985 (665) | beats (known) |
| 5 | 3 | 0.333 | **643** | **0.291** | **−0.043** |
| 7 | 2 | 0.500 | **571** | **0.352** | **−0.148** |
| 11 | 2 | 0.500 | **37** | **0.432** | **−0.068** |

**Honest:** a theory B/symbol win is **not** a product codec until encode/decode and hardware are priced. It does **not** move the ternary BitNet frontier past 665-flat.

---

## Claim tags

| Claim | Tag |
|-------|-----|
| Circle / CF / three loci / flat-vs-nested generalize to any \(p\) | `applies_as_language` |
| Three regimes = right geometric dynamical image for every \(p\) | `applies_as_language` |
| Training needs more than three regimes | `speculative` |
| p=5 deficit-near-0 \(Q\)s are alphabet “0” seeds | `measured` |
| Flat large block beats naïve digit pack (theory, some \(p\)) | `measured` |
| Implement p=5 first rung as product codec **now** | `speculative` |
| Base-p densifies ternary BitNet past 665-flat | `does_not_apply` |
| UFRF twin-sheet forced onto base 5 | `speculative` / do not invent |

---

## Verdict

**Actionable only if** someone wants a **non-ternary** codec — then p=5 first surplus flat block is the natural starting implement. Otherwise base-p is a **conceptual map**. Ternary static frontier remains **665-flat** (`--hybrid665`).
