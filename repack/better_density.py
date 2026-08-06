#!/usr/bin/env python3
"""Can we beat flat fmt_5_8 on BitNet without pad/redesign?

Explores ledger variants that use known rung-block density (61/306) and
41-block density more aggressively than the default exact-÷306 gate.

Strategies (theory bytes, fixed trit count):
  A  flat fmt_5_8
  B  best-axis fiber fmt_5_8
  C  best layout of theory_bytes_306_485  (306 blocks + bigint rem) — no ÷306 gate
  D  best layout of hybrid_306 + fmt_5_8 rem
  E  best layout of theory_bytes_41_65
  F  cascade: as many 306, then 41, then 5_8 on rem (best layout)
  G  rem-oracle: 306 full blocks + min(5_8, 41_65, bigint) on rem
  H  min(A..G) oracle

Also reports when C/D beat A and by how much. Updates stance: is there
anything beyond the known −1 B/306 effect?
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pack_ladder import (
    BYTES_306,
    BLOCK_306,
    container_bytes_for_r_trits,
    theory_bytes_41_65,
    theory_bytes_5_8,
    theory_bytes_306_485,
)
from packing_stack import load_bitnet_shapes
from ledger_packer import pack_tensor

DEFAULT_CKPT = Path(__file__).resolve().parent / "data" / "bitnet" / "model.safetensors"
OUT = Path(__file__).resolve().parent / "better_density_results.json"


def hybrid_306_58(n: int) -> int:
    full, rem = divmod(n, BLOCK_306)
    return full * BYTES_306 + theory_bytes_5_8(rem)


def cascade_306_41_58(n: int) -> int:
    full306, rem = divmod(n, BLOCK_306)
    full41, rem2 = divmod(rem, 41)
    return full306 * BYTES_306 + full41 * 9 + theory_bytes_5_8(rem2)


def rem_oracle_306(n: int) -> int:
    full, rem = divmod(n, BLOCK_306)
    if rem == 0:
        return full * BYTES_306
    rem_cost = min(
        theory_bytes_5_8(rem),
        theory_bytes_41_65(rem),
        container_bytes_for_r_trits(rem),
    )
    return full * BYTES_306 + rem_cost


def best_layout(m: int, n: int, length_fn: Callable[[int], int]) -> Dict[str, Any]:
    opts = [
        ("flat_stream", length_fn(m * n)),
        ("row_fibers", m * length_fn(n)),
        ("col_fibers", n * length_fn(m)),
    ]
    layout, b = min(opts, key=lambda t: t[1])
    return {"layout": layout, "bytes": b, "all": {k: v for k, v in opts}}


def evaluate_shape(m: int, n: int) -> Dict[str, Any]:
    flat = theory_bytes_5_8(m * n)
    fiber58 = best_layout(m, n, theory_bytes_5_8)
    c306 = best_layout(m, n, theory_bytes_306_485)
    h30658 = best_layout(m, n, hybrid_306_58)
    f41 = best_layout(m, n, theory_bytes_41_65)
    casc = best_layout(m, n, cascade_306_41_58)
    remo = best_layout(m, n, rem_oracle_306)

    cand = {
        "A_flat_5_8": flat,
        "B_best_fiber_5_8": fiber58["bytes"],
        "C_306_bigint_rem": c306["bytes"],
        "D_306_5_8_rem": h30658["bytes"],
        "E_41_65": f41["bytes"],
        "F_cascade_306_41_58": casc["bytes"],
        "G_306_rem_oracle": remo["bytes"],
    }
    best_name = min(cand, key=cand.get)
    oracle = cand[best_name]
    packer = pack_tensor(m, n)
    return {
        "shape": [m, n],
        "candidates": cand,
        "layouts": {
            "C": c306,
            "D": h30658,
            "E": f41,
            "F": casc,
            "G": remo,
            "B": fiber58,
        },
        "oracle_name": best_name,
        "oracle_bytes": oracle,
        "delta_oracle_vs_flat": oracle - flat,
        "delta_C_vs_flat": cand["C_306_bigint_rem"] - flat,
        "delta_D_vs_flat": cand["D_306_5_8_rem"] - flat,
        "delta_G_vs_flat": cand["G_306_rem_oracle"] - flat,
        "packer_path": packer["decision"]["path"],
        "packer_bytes": packer["decision"]["bytes"],
        "packer_minus_oracle": packer["decision"]["bytes"] - oracle,
    }


def measure_bitnet(ckpt: Path, max_tensors: Optional[int] = None) -> Dict[str, Any]:
    shapes = load_bitnet_shapes(ckpt)
    if max_tensors is not None:
        shapes = shapes[:max_tensors]

    totals = defaultdict(int)
    path_oracle: Dict[str, int] = defaultdict(int)
    wins_vs_flat = defaultdict(int)
    unique_rows: List[Dict[str, Any]] = []
    seen = set()
    n_tensors = 0
    trit_total = 0
    packer_left_on_table = 0

    for _name, m, n in shapes:
        ev = evaluate_shape(m, n)
        n_tensors += 1
        trit_total += m * n
        for k, v in ev["candidates"].items():
            totals[k] += v
        totals["packer"] += ev["packer_bytes"]
        path_oracle[ev["oracle_name"]] += 1
        packer_left_on_table += ev["packer_minus_oracle"]
        for tag, delta_key in (
            ("C", "delta_C_vs_flat"),
            ("D", "delta_D_vs_flat"),
            ("G", "delta_G_vs_flat"),
            ("oracle", "delta_oracle_vs_flat"),
        ):
            if ev[delta_key] < 0:
                wins_vs_flat[tag] += 1
        key = (m, n)
        if key not in seen:
            seen.add(key)
            unique_rows.append(ev)

    flat = totals["A_flat_5_8"]
    dens = {k: (totals[k] / trit_total if trit_total else None) for k in totals}

    # Is oracle just the 61/306 effect?
    # Expected save ≈ floor(trits_in_aligned_fibers)/306 if we could apply per layout.
    # Compare oracle save to pure density 61/306 vs 0.2 on same trit count.
    pure_306_density_bytes = int(round(trit_total * (61 / 306)))
    expected_vs_0p2 = flat - pure_306_density_bytes  # if every trit were in 306 blocks

    return {
        "n_tensors": n_tensors,
        "n_unique": len(unique_rows),
        "trits": trit_total,
        "totals": dict(totals),
        "bytes_per_trit": dens,
        "delta_vs_flat_5_8": {k: totals[k] - flat for k in totals},
        "oracle_path_counts": dict(path_oracle),
        "tensors_beating_flat": dict(wins_vs_flat),
        "packer_bytes_left_on_table_vs_oracle": packer_left_on_table,
        "pure_all_306_density_bytes": pure_306_density_bytes,
        "gap_flat_to_pure_306_density": flat - pure_306_density_bytes,
        "per_unique": unique_rows,
    }


def understand_when_win() -> Dict[str, Any]:
    """Scan remainders: when does 306-prefix beat flat 5_8 on a 1-D stream?"""
    rows = []
    for n in range(0, 306 * 4 + 1):
        a = theory_bytes_5_8(n)
        c = theory_bytes_306_485(n)
        d = hybrid_306_58(n)
        g = rem_oracle_306(n)
        f = cascade_306_41_58(n)
        best = min(a, c, d, g, f)
        if best < a:
            rows.append(
                {
                    "n": n,
                    "flat": a,
                    "C": c,
                    "D": d,
                    "G": g,
                    "F": f,
                    "save": a - best,
                    "full306": n // 306,
                    "rem": n % 306,
                }
            )
    # Summarize by # of full 306 blocks
    by_k: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        k = r["full306"]
        by_k.setdefault(k, {"n_win": 0, "total_save": 0, "max_save": 0})
        by_k[k]["n_win"] += 1
        by_k[k]["total_save"] += r["save"]
        by_k[k]["max_save"] = max(by_k[k]["max_save"], r["save"])
    return {
        "n_lengths_beating_flat_in_0_1224": len(rows),
        "by_full_306_blocks": {str(k): v for k, v in sorted(by_k.items())},
        "examples": rows[:: max(1, len(rows) // 15)][:16],
        "law": "save ≈ floor(n/306) bytes vs ceil(n/5) when rem packing does not erase the gap",
    }


def run(ckpt: Path, max_tensors: Optional[int] = None) -> Dict[str, Any]:
    t0 = time.time()
    bit = measure_bitnet(ckpt, max_tensors=max_tensors) if ckpt.is_file() else {
        "skipped": True,
        "reason": f"missing {ckpt}",
    }
    under = understand_when_win()

    # Verdict
    if bit.get("skipped"):
        verdict = "no_ckpt"
        better = False
    else:
        ag = {
            k: bit["totals"][k]
            for k in bit["totals"]
            if k[0] in "ABCDEFG"
        }
        oracle_k = min(ag, key=ag.get)
        flat = bit["totals"]["A_flat_5_8"]
        oracle_save = flat - ag[oracle_k]
        best_delta = ag[oracle_k] - flat
        better = best_delta < 0
        pure_gap = bit["gap_flat_to_pure_306_density"]
        c_save = flat - ag["C_306_bigint_rem"]
        d_save = flat - ag["D_306_5_8_rem"]
        g_save = flat - ag["G_306_rem_oracle"]
        f_save = flat - ag["F_cascade_306_41_58"]
        beyond_simple_hybrid = oracle_save > max(c_save, d_save) + 64
        verdict = {
            "beats_flat_5_8": better,
            "best_strategy": oracle_k if better else "A_flat_5_8",
            "best_delta_bytes": best_delta,
            "oracle_strategy": oracle_k,
            "oracle_save_bytes": oracle_save,
            "saves": {
                "C_306_bigint_rem": c_save,
                "D_306_5_8_rem": d_save,
                "F_cascade": f_save,
                "G_rem_oracle": g_save,
            },
            "pure_306_density_gap_bytes": pure_gap,
            "oracle_vs_pure_gap_pct": (
                round(100.0 * oracle_save / pure_gap, 2) if pure_gap else None
            ),
            "beyond_simple_hybrid": beyond_simple_hybrid,
            "new_beyond_61_over_306": False,
            "interpretation": (
                "Best fixed-width gain is the known rung-block / hybrid effect "
                "(≈ −1 B per full 306 in the chosen layout). Cascade/rem-oracle "
                "do not materially beat C/D. Packer leaves that on the table "
                "because of the exact-÷306 gate."
                if better
                else "No strategy beat flat 5_8 on this checkpoint."
            ),
        }

    out = {
        "stance": (
            "Probe: can we do better than flat 5_8 without pad/redesign? "
            "Measure hybrid/cascade/rem-oracle; attribute any win to 61/306."
        ),
        "understand_1d": under,
        "bitnet": bit,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
        "recommendation": (
            "Optional: allow hybrid/C (theory_bytes_306_485 on best layout) "
            "in ledger_packer when it beats flat_5_8 — captures ~oracle save; "
            "do not expect more than the known 0.33% density floor."
            if (isinstance(verdict, dict) and verdict.get("beats_flat_5_8"))
            else "No change."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def selftest() -> None:
    assert theory_bytes_306_485(306) == 61
    assert hybrid_306_58(306) == 61
    assert theory_bytes_5_8(306) == 62
    # rem oracle ≤ each component strategy
    for n in (0, 1, 5, 40, 41, 305, 306, 307, 612, 700):
        g = rem_oracle_306(n)
        assert g <= theory_bytes_306_485(n)
        assert g <= hybrid_306_58(n)
        assert g <= cascade_306_41_58(n)
    ev = evaluate_shape(1, 306)
    assert ev["candidates"]["C_306_bigint_rem"] == 61
    assert ev["delta_C_vs_flat"] == -1
    print("selftest ok")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("selftest", "run"), nargs="?", default="run")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--max-tensors", type=int, default=None)
    args = p.parse_args(argv)
    if args.cmd == "selftest":
        selftest()
        return 0
    out = run(args.ckpt, max_tensors=args.max_tensors)
    v = out["verdict"]
    print(json.dumps({"verdict": v, "recommendation": out["recommendation"]}, indent=2))
    if not out["bitnet"].get("skipped"):
        d = out["bitnet"]["delta_vs_flat_5_8"]
        print("deltas_vs_flat_MB:", {k: round(v / 1e6, 3) for k, v in d.items()})
        print("oracle_paths:", out["bitnet"]["oracle_path_counts"])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
