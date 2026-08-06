# Large-scale density probe (≥7B synthetic)

*No ternary safetensors larger than BitNet-2B found locally.*  
Synthetic LLaMA-class inventories with layer multiplicity.

Code: [`repack/large_scale_probe.py`](repack/large_scale_probe.py)  
Results: [`repack/large_scale_results.json`](repack/large_scale_results.json)

Related: [`BETTER_DENSITY.md`](BETTER_DENSITY.md), [`PACKING_PIPELINE.md`](PACKING_PIPELINE.md)

## Verdict

**Scale does not expose a new packing phenomenon** within the 306-family.
It only amplifies the known **61/306 ≈ 0.33%** hybrid / rung-block gap
(−1 **byte** per 306 trits). Fiber-41, cascade, and rem-oracle still lose to
(or tie) simple `theory_bytes_306_485` hybrid at 7B–70B.

**Static density frontier update:** flat `fmt_665_1055` beats hybrid 306 at
these same scales (see [`PATTERN_OF_PATTERNS.md`](PATTERN_OF_PATTERNS.md);
map: [`DIMENSIONS.md`](DIMENSIONS.md)). This note remains the 306-amplification
measurement; it is not “freeze forever at 0.33%.”

## Suites measured

| Suite | Trits | Flat `5_8` | Hybrid Δ | B/trit hybrid | New? |
|-------|------:|-----------:|---------:|--------------:|:----:|
| 7B-class | 6.48B | 1295 MB | **−4.2 MB** | 0.199346 | no |
| 13B-class | 12.7B | 2538 MB | **−8.3 MB** | 0.199346 | no |
| 70B-class | 77.8B | 15.6 GB | **−51.3 MB** | 0.199341 | no |
| 7B ×306-aligned | 6.23B | 1246 MB | **−4.1 MB** | **exact 61/306** | no |

- Default packer (no `--hybrid`): all **flat58** on stock LLaMA dims.
- With `--hybrid` / C: all hybrid; captures ~**100%** of the pure 61/306 gap.
- Design-time **7B ×306-aligned**: default packer takes **`frame` on 224/224**
  tensors (no flag needed) — confirms the greenfield prior.

## What scale does *not* change

- Fiber-41 still **worse** (+126 MB @7B, +1.5 GB @70B).
- Cascade / rem-oracle do not beat C.
- No residue / seam effect appears only at large MN.

## Run

```bash
PYTHONPATH=repack python3 repack/large_scale_probe.py selftest
PYTHONPATH=repack python3 repack/large_scale_probe.py run
PYTHONPATH=repack python3 repack/large_scale_probe.py run --suite 70b
```

When a real >2B ternary ckpt exists: `ledger_packer.py bitnet --ckpt … --hybrid`.
