#!/usr/bin/env python3
"""Large-scale density probe — synthetic 7B / 70B-class shape suites.

No larger ternary safetensors found locally beyond BitNet-2B. This builds
realistic LLaMA-class weight inventories (with layer multiplicity) and measures
flat fmt_5_8 vs hybrid 306-prefix vs default packer vs fiber-41.

Question: does scale expose phenomena beyond the known 61/306 ≈ 0.33% gap?
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from better_density import (
    best_layout,
    cascade_306_41_58,
    evaluate_shape,
    hybrid_306_58,
    rem_oracle_306,
)
from ledger_packer import pack_tensor
from pack_ladder import theory_bytes_41_65, theory_bytes_5_8, theory_bytes_306_485

OUT = Path(__file__).resolve().parent / "large_scale_results.json"

# (name, m, n, multiplicity) — multiplicity ≈ layers × copies of that mat
# Trit totals target ~param counts for dense ternary weights (ignore embeds/ln).

SUITE_7B: Tuple[Tuple[str, int, int, int], ...] = (
    # LLaMA-7B-ish: hidden=4096, intermediate=11008, n_layers=32
    ("attn_q", 4096, 4096, 32),
    ("attn_k", 4096, 4096, 32),
    ("attn_v", 4096, 4096, 32),
    ("attn_o", 4096, 4096, 32),
    ("mlp_gate", 11008, 4096, 32),
    ("mlp_up", 11008, 4096, 32),
    ("mlp_down", 4096, 11008, 32),
)

SUITE_13B: Tuple[Tuple[str, int, int, int], ...] = (
    ("attn_q", 5120, 5120, 40),
    ("attn_k", 5120, 5120, 40),
    ("attn_v", 5120, 5120, 40),
    ("attn_o", 5120, 5120, 40),
    ("mlp_gate", 13824, 5120, 40),
    ("mlp_up", 13824, 5120, 40),
    ("mlp_down", 5120, 13824, 40),
)

SUITE_70B: Tuple[Tuple[str, int, int, int], ...] = (
    # LLaMA-70B-ish: hidden=8192, intermediate=28672, n_layers=80
    ("attn_q", 8192, 8192, 80),
    ("attn_k", 8192, 8192, 80),  # full; GQA would be smaller — stress test
    ("attn_v", 8192, 8192, 80),
    ("attn_o", 8192, 8192, 80),
    ("mlp_gate", 28672, 8192, 80),
    ("mlp_up", 28672, 8192, 80),
    ("mlp_down", 8192, 28672, 80),
)

# Design-time ×306 near common hiddens (greenfield prior stress)
SUITE_7B_ALIGN306: Tuple[Tuple[str, int, int, int], ...] = (
    ("attn_q", 3978, 3978, 32),   # 13*306
    ("attn_k", 3978, 3978, 32),
    ("attn_v", 3978, 3978, 32),
    ("attn_o", 3978, 3978, 32),
    ("mlp_gate", 11016, 3978, 32),  # 36*306, 13*306
    ("mlp_up", 11016, 3978, 32),
    ("mlp_down", 3978, 11016, 32),
)

SUITES = {
    "7b": SUITE_7B,
    "13b": SUITE_13B,
    "70b": SUITE_70B,
    "7b_align306": SUITE_7B_ALIGN306,
}


def measure_suite(
    name: str,
    suite: Sequence[Tuple[str, int, int, int]],
) -> Dict[str, Any]:
    totals = defaultdict(int)
    path_default: Dict[str, int] = defaultdict(int)
    path_hybrid: Dict[str, int] = defaultdict(int)
    oracle_names: Dict[str, int] = defaultdict(int)
    per_unique: List[Dict[str, Any]] = []
    seen = set()
    n_tensors = 0
    trit_total = 0

    for tname, m, n, mult in suite:
        ev = evaluate_shape(m, n)
        pd = pack_tensor(m, n, allow_hybrid=False)
        ph = pack_tensor(m, n, allow_hybrid=True)
        trits = m * n
        n_tensors += mult
        trit_total += trits * mult

        for k, v in ev["candidates"].items():
            totals[k] += v * mult
        totals["packer_default"] += pd["decision"]["bytes"] * mult
        totals["packer_hybrid"] += ph["decision"]["bytes"] * mult
        path_default[pd["decision"]["path"]] += mult
        path_hybrid[ph["decision"]["path"]] += mult
        oracle_names[ev["oracle_name"]] += mult

        key = (m, n)
        if key not in seen:
            seen.add(key)
            per_unique.append(
                {
                    "name": tname,
                    "shape": [m, n],
                    "mult": mult,
                    "candidates": ev["candidates"],
                    "delta_C_vs_flat": ev["delta_C_vs_flat"],
                    "oracle": ev["oracle_name"],
                    "packer_default": pd["decision"]["path"],
                    "packer_hybrid": ph["decision"]["path"],
                    "mod306_m": m % 306,
                    "mod306_n": n % 306,
                    "mod306_mn": (m * n) % 306,
                }
            )

    flat = totals["A_flat_5_8"]
    pure_306 = int(round(trit_total * (61 / 306)))
    c_save = flat - totals["C_306_bigint_rem"]
    hy_save = flat - totals["packer_hybrid"]
    dens = {k: totals[k] / trit_total for k in totals}

    # New-at-scale checks
    fiber_beats_flat = totals["E_41_65"] < flat
    cascade_beats_C = totals["F_cascade_306_41_58"] < totals["C_306_bigint_rem"]
    rem_beats_C = totals["G_306_rem_oracle"] < totals["C_306_bigint_rem"]
    default_uses_frame = path_default.get("frame", 0) > 0
    hybrid_pct_of_pure = (100.0 * hy_save / (flat - pure_306)) if flat > pure_306 else None

    return {
        "suite": name,
        "n_tensors": n_tensors,
        "n_unique_shapes": len(per_unique),
        "trits": trit_total,
        "trits_B": round(trit_total / 1e9, 3),
        "totals": dict(totals),
        "bytes_per_trit": dens,
        "delta_vs_flat": {k: totals[k] - flat for k in totals},
        "delta_vs_flat_MB": {
            k: round((totals[k] - flat) / 1e6, 3) for k in totals
        },
        "path_counts_default": dict(path_default),
        "path_counts_hybrid": dict(path_hybrid),
        "oracle_path_counts": dict(oracle_names),
        "pure_306_density_bytes": pure_306,
        "gap_flat_to_pure_306": flat - pure_306,
        "hybrid_save_bytes": hy_save,
        "C_save_bytes": c_save,
        "hybrid_vs_pure_gap_pct": (
            round(hybrid_pct_of_pure, 2) if hybrid_pct_of_pure is not None else None
        ),
        "scale_checks": {
            "fiber41_beats_flat": fiber_beats_flat,
            "cascade_beats_C": cascade_beats_C,
            "rem_oracle_beats_C": rem_beats_C,
            "default_packer_uses_frame": default_uses_frame,
            "new_phenomenon_beyond_61_306": bool(
                fiber_beats_flat or cascade_beats_C or rem_beats_C
            ),
        },
        "per_unique": per_unique,
    }


def run(which: Sequence[str] | None = None) -> Dict[str, Any]:
    t0 = time.time()
    names = list(which) if which else list(SUITES.keys())
    suites_out = {}
    for name in names:
        suites_out[name] = measure_suite(name, SUITES[name])

    any_new = any(
        s["scale_checks"]["new_phenomenon_beyond_61_306"] for s in suites_out.values()
    )
    summary_rows = []
    for name, s in suites_out.items():
        summary_rows.append(
            {
                "suite": name,
                "trits_B": s["trits_B"],
                "flat_MB": round(s["totals"]["A_flat_5_8"] / 1e6, 2),
                "hybrid_delta_MB": s["delta_vs_flat_MB"]["packer_hybrid"],
                "C_delta_MB": s["delta_vs_flat_MB"]["C_306_bigint_rem"],
                "hybrid_pct_of_pure_gap": s["hybrid_vs_pure_gap_pct"],
                "B_per_trit_hybrid": s["bytes_per_trit"]["packer_hybrid"],
                "new_phenomenon": s["scale_checks"]["new_phenomenon_beyond_61_306"],
            }
        )

    out = {
        "stance": (
            "Large-scale synthetic suites (no >2B ternary ckpt locally). "
            "Measure whether scale exposes wins beyond known 61/306."
        ),
        "local_ckpt_search": {
            "bitnet_2b": str(
                Path(__file__).resolve().parent / "data" / "bitnet" / "model.safetensors"
            ),
            "larger_found": False,
            "note": "Only BitNet-2B safetensors present under repack/data/",
        },
        "suites": suites_out,
        "summary": summary_rows,
        "verdict": {
            "new_phenomenon_at_scale": any_new,
            "interpretation": (
                "Scale only amplifies the same ~0.33% (61/306) hybrid gap; "
                "fiber-41 / cascade / rem-oracle do not overtake C at 7B–70B."
                if not any_new
                else "Unexpected strategy beat C/flat — investigate."
            ),
        },
        "elapsed_s": time.time() - t0,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def selftest() -> None:
    r = measure_suite("tiny", (("q", 4096, 4096, 1),))
    assert r["trits"] == 4096 * 4096
    assert r["delta_vs_flat"]["C_306_bigint_rem"] < 0
    assert r["bytes_per_trit"]["packer_hybrid"] < 0.2 + 1e-12
    print("large_scale_probe selftest OK")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("selftest", "run"), nargs="?", default="run")
    p.add_argument(
        "--suite",
        action="append",
        choices=list(SUITES.keys()),
        help="Suite name (repeatable). Default: all.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "selftest":
        selftest()
        return 0
    out = run(args.suite)
    print(json.dumps({"verdict": out["verdict"], "summary": out["summary"]}, indent=2))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
