# Better density probe — can we beat flat `5_8` without pad/redesign?

Code: [`repack/better_density.py`](repack/better_density.py)  
Results: [`repack/better_density_results.json`](repack/better_density_results.json)  
Packer flag: `ledger_packer.py … --hybrid`

## Question

Is there a fixed-width packing strategy that beats flat `fmt_5_8` on BitNet
beyond (or within) the known **61/306 ≈ 0.33%** rung-block effect
(−1 **byte** per 306 trits)?

*(Later: yes — flat 665 blocks; this probe scoped the 306 family only.
See [`PATTERN_OF_PATTERNS.md`](PATTERN_OF_PATTERNS.md) / [`DIMENSIONS.md`](DIMENSIONS.md).)*

## Strategies tested (theory bytes)

| ID | Strategy |
|----|----------|
| A | flat `fmt_5_8` |
| B | best-axis fiber `5_8` |
| C | best layout `theory_bytes_306_485` (306 blocks + bigint rem) — **no ÷306 gate** |
| D | best layout hybrid 306 + `5_8` rem |
| E | best layout `theory_bytes_41_65` |
| F | cascade 306 → 41 → `5_8` |
| G | 306 full + rem-oracle min(`5_8`, `41_65`, bigint) |

## BitNet-2B result (210 tensors)

| Strategy | Δ vs flat `5_8` |
|----------|----------------:|
| A / B / default packer | **0** |
| **C / D / G** | **−1,362,030 B (−1.36 MB)** |
| F cascade | −1,361,220 B (slightly worse) |
| E fiber-41 | **+40.5 MB** (loses) |

- Oracle = **C** on all 210 tensors.
- Pure all-trit 61/306 gap ≈ **1,362,121 B** → oracle captures **99.99%**.
- Cascade / rem-oracle do **not** beat simple C/D (`beyond_simple_hybrid=false`).
- **No new geometry within 306** — same −1 **byte** per full 306 in the chosen layout.

## Understanding

1-D scan: a length beats flat `5_8` iff enough full 306-blocks accrue the
byte gap before rem packing erases it. Law: save ≈ `floor(n/306)` bytes vs
`ceil(n/5)` when rem is well-behaved.

## Packer change

Optional `--hybrid` enables C in [`ledger_packer.py`](repack/ledger_packer.py)
without requiring `length % 306 == 0`. Default remains exact-quantum three-ledger
(flat knee on BitNet). With `--hybrid`, expect ~**−1.36 MB** / `beats_flat58`.

```bash
python3 repack/better_density.py selftest
python3 repack/better_density.py run
python3 repack/ledger_packer.py selftest
python3 repack/ledger_packer.py bitnet --hybrid
```

## Stance

Within the **306-family** strategies tested here, lifting the exact-÷306 gate
captures essentially the full known rung-block gap (−1 **byte** per 306 trits
≈ 0.33% vs the 0.2 baseline). Cascade, 41-blocks, and rem tricks do **not**
beat simple hybrid 306 on BitNet-2B.

**Superseded as overall static frontier:** flat Law-B-sum blocks
`fmt_665_1055` (132/665) beat hybrid 306 — see
[`PATTERN_OF_PATTERNS.md`](PATTERN_OF_PATTERNS.md) and the multi-track map
[`DIMENSIONS.md`](DIMENSIONS.md). Probe past 665: flat **1277**
([`FARTHER.md`](FARTHER.md)). Do not read this note as “freeze at
0.33%/306”; close static packing after docs + a larger absolute measure, with
665-flat as the shipped best measured lever.

**At 7B–70B synthetic scale** ([`LARGE_SCALE.md`](LARGE_SCALE.md)): same 306
law — hybrid Δ scales ~linearly (−4 / −8 / −51 MB); no new 306-phenomenon.
(665-flat deltas are larger; measured in pattern-of-patterns.)
