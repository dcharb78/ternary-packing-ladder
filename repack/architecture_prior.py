#!/usr/bin/env python3
"""Architecture prior — surplus-friendly sizes on an align grid.

Design-time catalogue: lengths Q ≡ 0 (mod align) with high {Q·α}, usable as
hidden/MLP widths without post-hoc padding. Hypothesis, not a theorem.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Sequence

from harmonic_tax import bits_from_alpha, frac_Q_alpha, max_zero_tax_m
from packing_stack import ledger_bytes

getcontext().prec = 120

SURPLUS = Decimal("0.9")
ALIGN_DEFAULT = 64


def scan_align_grid(
    align: int = ALIGN_DEFAULT,
    q_min: int = 64,
    q_max: int = 8192,
    top_k: int = 40,
) -> Dict[str, Any]:
    rows = []
    for Q in range(align, q_max + 1, align):
        if Q < q_min:
            continue
        phi = frac_Q_alpha(Q)
        rows.append(
            {
                "Q": Q,
                "phi": format(phi, "f"),
                "surplus": phi >= SURPLUS,
                "dist_to_1": format(1 - phi, "f"),
                "max_m_zero_tax": max_zero_tax_m(Q),
                "bits": bits_from_alpha(Q),
            }
        )
    surplus = [r for r in rows if r["surplus"]]
    surplus.sort(key=lambda r: float(r["phi"]), reverse=True)
    # Also top by max_m_zero_tax
    by_zt = sorted(rows, key=lambda r: r["max_m_zero_tax"], reverse=True)[:top_k]
    # Pair smoke: surplus Q × another surplus / deficit partner on grid
    partners = []
    for a in surplus[:15]:
        for b in surplus[:15]:
            if b["Q"] <= a["Q"]:
                continue
            led = ledger_bytes(a["Q"], b["Q"])
            partners.append(
                {
                    "m": a["Q"],
                    "n": b["Q"],
                    "best_tax": led["best_tax"],
                    "best_fiber_41": led["best_fiber_41"],
                    "flat_5_8": led["flat_5_8"],
                    "fiber41_minus_flat58": led["best_fiber_41"] - led["flat_5_8"],
                }
            )
    partners.sort(key=lambda p: (p["best_tax"], p["best_fiber_41"]))
    return {
        "align": align,
        "q_min": q_min,
        "q_max": q_max,
        "n_grid": len(rows),
        "n_surplus": len(surplus),
        "top_surplus": surplus[:top_k],
        "top_by_zero_tax_m": by_zt[:20],
        "best_surplus_pairs": partners[:15],
        "note": (
            "Design-time sizes only. Does not beat flat 5_8 on dense unstructured "
            "payloads; prefer when choosing new widths."
        ),
    }


def selftest() -> None:
    r = scan_align_grid(align=64, q_min=64, q_max=1024, top_k=10)
    assert r["n_grid"] > 0
    print("architecture_prior selftest OK")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: architecture_prior.py {selftest|run} [--align N] [--q-max N]")
        return 0
    cmd = argv[0]
    align = 64
    q_max = 8192
    if "--align" in argv:
        align = int(argv[argv.index("--align") + 1])
    if "--q-max" in argv:
        q_max = int(argv[argv.index("--q-max") + 1])
    if cmd == "selftest":
        selftest()
        return 0
    if cmd == "run":
        out = scan_align_grid(align=align, q_max=q_max)
        path = Path(__file__).resolve().parent / "architecture_prior_results.json"
        summary = {
            "align": out["align"],
            "n_surplus": out["n_surplus"],
            "n_grid": out["n_grid"],
            "top_surplus_Q": [r["Q"] for r in out["top_surplus"][:15]],
            "best_pair": out["best_surplus_pairs"][0] if out["best_surplus_pairs"] else None,
        }
        print(json.dumps(summary, indent=2))
        path.write_text(json.dumps(out, indent=2))
        print(f"wrote {path}")
        return 0
    print(f"unknown: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
