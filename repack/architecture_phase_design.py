#!/usr/bin/env python3
"""Architecture-level phase design — seed / amplify / harmonize as a prior.

Phase 5 tests (exploratory; no theorems):

  T1  Per-shape design on LLM / BitNet-like rectangles
  T2  Global phase budget across the shape suite as a fake net
  T3  Half (self-complement) vs surplus pad on squares

Uses settled circle identities only. Does not claim to beat flat (5,8) on
dense BitNet. Process language from COLLATZ_BRIDGE.md is a hypothesis that
already passed amplify fidelity — here we ask whether it is a useful design prior.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from harmonic_multi import multi_mode_pad
from harmonic_tax import (
    bits_from_alpha,
    frac_Q_alpha,
    max_zero_tax_m,
    tax_cols_harmonic,
    tax_rows_harmonic,
)
from pad_to_tax0 import pad_toward_surplus_phase
from scale_probe import LLM_SHAPES, axis_bits

getcontext().prec = 120

SURPLUS_BAND = Decimal("0.9")
HALF_BAND = Decimal("0.05")  # |φ − ½| < this ⇒ near_half


def _phi(Q: int) -> Decimal:
    return frac_Q_alpha(Q)


def _side(phi: Decimal) -> str:
    if abs(phi - Decimal("0.5")) < HALF_BAND:
        return "near_half"
    if phi >= SURPLUS_BAND:
        return "surplus"
    if phi <= Decimal("0.1"):
        return "deficit"
    return "mid"


def rect_tax(m: int, n: int) -> Dict[str, Any]:
    tr = tax_rows_harmonic(m, n)
    tc = tax_cols_harmonic(m, n)
    ax = axis_bits(m, n)
    return {
        "m": m,
        "n": n,
        "phi_m": format(_phi(m), "f"),
        "phi_n": format(_phi(n), "f"),
        "side_m": _side(_phi(m)),
        "side_n": _side(_phi(n)),
        "tax_rows": tr,
        "tax_cols": tc,
        "best_tax": min(tr, tc),
        "complement_dist": format(abs(_phi(m) + _phi(n) - 1), "f"),
        "best_axis": ax["best_axis"],
        "axis_bits_saved": ax["bits_saved_vs_worse"],
        "row_bits": ax["row_bits"],
        "col_bits": ax["col_bits"],
    }


# ---------------------------------------------------------------------------
# T1 — per-shape design
# ---------------------------------------------------------------------------

def design_shape(name: str, m: int, n: int, max_pad: int = 64) -> Dict[str, Any]:
    base = rect_tax(m, n)
    surplus_m = pad_toward_surplus_phase(m, max_pad=max_pad)
    surplus_n = pad_toward_surplus_phase(n, max_pad=max_pad)
    after_surplus = rect_tax(surplus_m["L_prime"], surplus_n["L_prime"])
    harm = multi_mode_pad([m, n], max_pad=max_pad, target="harmonize")
    after_harm = rect_tax(harm["padded"][0], harm["padded"][1])
    # Also try amplify_surplus on both modes
    amp = multi_mode_pad([m, n], max_pad=max_pad, target="amplify_surplus")
    after_amp = rect_tax(amp["padded"][0], amp["padded"][1])

    def delta(after: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "best_tax_delta": after["best_tax"] - base["best_tax"],
            "axis_bits_delta": after["axis_bits_saved"] - base["axis_bits_saved"],
            "complement_after": after["complement_dist"],
        }

    return {
        "name": name,
        "baseline": base,
        "surplus_pad": {
            "pads": [surplus_m["pad_trits"], surplus_n["pad_trits"]],
            "total_pad": surplus_m["pad_trits"] + surplus_n["pad_trits"],
            "after": after_surplus,
            **delta(after_surplus),
        },
        "harmonize_pad": {
            "pads": harm["pads"],
            "total_pad": harm["total_pad"],
            "score_before": harm["score_before"],
            "score_after": harm["score_after"],
            "after": after_harm,
            **delta(after_harm),
        },
        "amplify_surplus_pad": {
            "pads": amp["pads"],
            "total_pad": amp["total_pad"],
            "after": after_amp,
            **delta(after_amp),
        },
    }


def run_t1(max_pad: int = 64) -> Dict[str, Any]:
    rows = [design_shape(n, m, k, max_pad=max_pad) for n, m, k in LLM_SHAPES]
    # Aggregate: how often each strategy lowers best_tax
    def wins(key: str) -> int:
        return sum(1 for r in rows if r[key]["best_tax_delta"] < 0)

    return {
        "max_pad": max_pad,
        "shapes": rows,
        "n_shapes": len(rows),
        "surplus_wins_tax": wins("surplus_pad"),
        "harmonize_wins_tax": wins("harmonize_pad"),
        "amplify_wins_tax": wins("amplify_surplus_pad"),
        "mean_surplus_pad": sum(r["surplus_pad"]["total_pad"] for r in rows) / len(rows),
        "mean_harmonize_pad": sum(r["harmonize_pad"]["total_pad"] for r in rows) / len(rows),
        "mean_tax_delta_surplus": sum(r["surplus_pad"]["best_tax_delta"] for r in rows) / len(rows),
        "mean_tax_delta_harmonize": sum(r["harmonize_pad"]["best_tax_delta"] for r in rows) / len(rows),
        "mean_tax_delta_amplify": sum(r["amplify_surplus_pad"]["best_tax_delta"] for r in rows) / len(rows),
    }


# ---------------------------------------------------------------------------
# T2 — global phase budget
# ---------------------------------------------------------------------------

def run_t2(max_pad: int = 64) -> Dict[str, Any]:
    """Treat unique mode lengths in LLM_SHAPES as a simultaneous inventory."""
    modes = sorted({m for _, m, n in LLM_SHAPES} | {n for _, m, n in LLM_SHAPES})
    before = {_m: _phi(_m) for _m in modes}
    pads = {m: pad_toward_surplus_phase(m, max_pad=max_pad) for m in modes}
    after_L = {m: pads[m]["L_prime"] for m in modes}
    after = {m: _phi(after_L[m]) for m in modes}

    total_pad = sum(pads[m]["pad_trits"] for m in modes)
    surplus_before = sum(1 for m in modes if before[m] >= SURPLUS_BAND)
    surplus_after = sum(1 for m in modes if after[m] >= SURPLUS_BAND)
    near_half_before = sum(1 for m in modes if abs(before[m] - Decimal("0.5")) < HALF_BAND)

    # Net-wide tax: sum of best_tax over shapes before/after replacing modes
    tax_before = 0
    tax_after = 0
    for _, m, n in LLM_SHAPES:
        tax_before += min(tax_rows_harmonic(m, n), tax_cols_harmonic(m, n))
        m2, n2 = after_L[m], after_L[n]
        tax_after += min(tax_rows_harmonic(m2, n2), tax_cols_harmonic(m2, n2))

    # Bits of payload implied by mode inventory (rough budget unit)
    trit_inventory = sum(modes)
    pad_fraction = total_pad / trit_inventory if trit_inventory else 0.0

    return {
        "unique_modes": modes,
        "n_modes": len(modes),
        "total_pad_trits": total_pad,
        "trit_inventory": trit_inventory,
        "pad_fraction_of_inventory": pad_fraction,
        "surplus_count_before": surplus_before,
        "surplus_count_after": surplus_after,
        "near_half_before": near_half_before,
        "sum_phi_before": format(sum(before.values()), "f"),
        "sum_phi_after": format(sum(after.values()), "f"),
        "suite_best_tax_before": tax_before,
        "suite_best_tax_after": tax_after,
        "suite_tax_delta": tax_after - tax_before,
        "per_mode": [
            {
                "L": m,
                "phi": format(before[m], "f"),
                "side": _side(before[m]),
                "L_prime": after_L[m],
                "phi_prime": format(after[m], "f"),
                "side_prime": _side(after[m]),
                "pad": pads[m]["pad_trits"],
                "max_m_zero_tax": max_zero_tax_m(m),
                "max_m_zero_tax_prime": max_zero_tax_m(after_L[m]),
            }
            for m in modes
        ],
        "verdict": (
            "cheap_simultaneous_surplus"
            if pad_fraction < 0.01 and surplus_after > surplus_before
            else "pad_budget_tension"
            if pad_fraction >= 0.01
            else "limited_surplus_gain"
        ),
    }


# ---------------------------------------------------------------------------
# T3 — half vs surplus on squares
# ---------------------------------------------------------------------------

SQUARE_QS = (640, 2560, 4096, 5120, 8192)


def square_tradeoff(Q: int, max_pad: int = 64) -> Dict[str, Any]:
    phi = _phi(Q)
    stay = rect_tax(Q, Q)
    surplus = pad_toward_surplus_phase(Q, max_pad=max_pad)
    Qp = surplus["L_prime"]
    leave = rect_tax(Qp, Qp)
    # Also pad only one side (rectangle break of square)
    one_side = rect_tax(Qp, Q)
    return {
        "Q": Q,
        "phi": format(phi, "f"),
        "side": _side(phi),
        "dist_to_half": format(abs(phi - Decimal("0.5")), "f"),
        "stay_square": {
            "best_tax": stay["best_tax"],
            "complement_dist": stay["complement_dist"],
            "max_m_zero_tax": max_zero_tax_m(Q),
        },
        "both_sides_surplus": {
            "Q_prime": Qp,
            "pad_each": surplus["pad_trits"],
            "best_tax": leave["best_tax"],
            "complement_dist": leave["complement_dist"],
            "max_m_zero_tax": max_zero_tax_m(Qp),
            "tax_delta": leave["best_tax"] - stay["best_tax"],
        },
        "one_side_surplus": {
            "shape": [Qp, Q],
            "pad": surplus["pad_trits"],
            "best_tax": one_side["best_tax"],
            "complement_dist": one_side["complement_dist"],
            "tax_delta": one_side["best_tax"] - stay["best_tax"],
        },
        "prefer": (
            "stay"
            if stay["best_tax"] <= leave["best_tax"]
            and stay["best_tax"] <= one_side["best_tax"]
            else "surplus_tie"
            if leave["best_tax"] == one_side["best_tax"]
            else "leave_to_surplus"
            if leave["best_tax"] < one_side["best_tax"]
            else "one_side_surplus"
        ),
        # Self-complement (φ+φ≈1) is NOT zero-tax. Zero-tax needs φ ≳ 1−1/m.
        "complement_vs_tax_note": (
            "self_complement_high_tax"
            if abs(phi - Decimal("0.5")) < HALF_BAND and stay["best_tax"] > Q // 4
            else "ok"
        ),
    }


def run_t3(max_pad: int = 64) -> Dict[str, Any]:
    rows = [square_tradeoff(Q, max_pad=max_pad) for Q in SQUARE_QS]
    return {
        "max_pad": max_pad,
        "squares": rows,
        "prefer_counts": {
            k: sum(1 for r in rows if r["prefer"] == k)
            for k in ("stay", "surplus_tie", "leave_to_surplus", "one_side_surplus")
        },
        "self_complement_high_tax": sum(
            1 for r in rows if r["complement_vs_tax_note"] == "self_complement_high_tax"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_all(max_pad: int = 64) -> Dict[str, Any]:
    t1 = run_t1(max_pad=max_pad)
    t2 = run_t2(max_pad=max_pad)
    t3 = run_t3(max_pad=max_pad)
    return {
        "stance": (
            "Exploratory Phase 5. Architecture phase design prior. "
            "No claim vs flat (5,8) on dense BitNet."
        ),
        "T1_per_shape": {
            k: t1[k]
            for k in t1
            if k != "shapes"
        },
        "T1_shapes": t1["shapes"],
        "T2_global_budget": t2,
        "T3_half_vs_surplus": t3,
        "summary": {
            "T1_harmonize_wins": t1["harmonize_wins_tax"],
            "T1_surplus_wins": t1["surplus_wins_tax"],
            "T1_amplify_wins": t1["amplify_wins_tax"],
            "T1_n": t1["n_shapes"],
            "T1_mean_tax_delta_harmonize": t1["mean_tax_delta_harmonize"],
            "T1_mean_tax_delta_surplus": t1["mean_tax_delta_surplus"],
            "T2_verdict": t2["verdict"],
            "T2_pad_fraction": t2["pad_fraction_of_inventory"],
            "T2_suite_tax_delta": t2["suite_tax_delta"],
            "T2_surplus_before_after": [t2["surplus_count_before"], t2["surplus_count_after"]],
            "T3_prefer_counts": t3["prefer_counts"],
        },
    }


def selftest() -> None:
    # Amplify identity still holds on a designed shape
    d = design_shape("selftest", 2560, 6912, max_pad=32)
    assert d["baseline"]["best_tax"] >= 0
    # Global budget modes non-empty
    t2 = run_t2(max_pad=16)
    assert t2["n_modes"] >= 3
    # Square 2560 near half
    s = square_tradeoff(2560, max_pad=32)
    assert s["side"] == "near_half"
    print("architecture_phase_design selftest OK")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: architecture_phase_design.py {selftest|run} [--max-pad N]")
        return 0
    cmd = argv[0]
    max_pad = 64
    if "--max-pad" in argv:
        max_pad = int(argv[argv.index("--max-pad") + 1])
    if cmd == "selftest":
        selftest()
        return 0
    if cmd == "run":
        out = run_all(max_pad=max_pad)
        path = Path(__file__).resolve().parent / "architecture_phase_results.json"
        # Compact summary print
        print(json.dumps(out["summary"], indent=2))
        path.write_text(json.dumps(out, indent=2))
        print(f"wrote {path}")
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
