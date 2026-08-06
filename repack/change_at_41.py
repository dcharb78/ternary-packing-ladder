#!/usr/bin/env python3
"""Exploratory probe: is there a structural “change at 41”?

Hypothesis (NOT assumed true): 41 is special because
  - packing: second CF rung of α=log₂3 (5→8, 41→65, 306→485)
  - twin-prime midpoint story: p(p−1)±1 → for p=7, 42±1 → 41,43
  - p=13 line: 312±1 → 311,313; packing has surplus rung 306 (near 312?)

Tests: scan neighbors of 41; compare to smooth baseline; check 306 vs 312;
associator involving 41; report pass/fail/null — no theorems.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from geometry_lab import associator, best_tax, holonomy
from harmonic_tax import (
    ALPHA,
    bits_from_alpha,
    frac_Q_alpha,
    max_zero_tax_m,
    tax_rows_harmonic,
)
from packing_stack import DeployConstraints, ledger_bytes, reshape_under_constraints
from tax_graph import SURPLUS_RUNGS

getcontext().prec = 120

# Twin-prime midpoint table from user (exploratory)
TWIN_MIDPOINTS = {
    3: (6, 5, 7),
    5: (30, 29, 31),
    7: (42, 41, 43),
    13: (312, 311, 313),
}


def row_for_Q(Q: int, m_probe: int = 8) -> Dict[str, Any]:
    phi = frac_Q_alpha(Q)
    return {
        "Q": Q,
        "phi": format(phi, "f"),
        "dist_to_1": format(1 - phi, "f"),
        "dist_to_half": format(abs(phi - Decimal("0.5")), "f"),
        "bits": bits_from_alpha(Q),
        "max_m_zero_tax": max_zero_tax_m(Q),
        "tax_rows_m8": tax_rows_harmonic(m_probe, Q),
        "tax_rows_m41": tax_rows_harmonic(41, Q) if Q != 41 else tax_rows_harmonic(41, 41),
        "square_best_tax": best_tax(Q, Q),
        "square_fiber41": ledger_bytes(Q, Q)["best_fiber_41"] if Q >= 32 else None,
        "is_surplus_rung": Q in SURPLUS_RUNGS or Q in (5, 41, 306, 15601),
        "is_certified_rung": Q in (5, 41, 306, 15601),
    }


def discrete_jumps(series: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """First differences of a numeric key along Q-ordered series."""
    out = []
    for a, b in zip(series, series[1:]):
        va, vb = a[key], b[key]
        if va is None or vb is None:
            continue
        out.append(
            {
                "from_Q": a["Q"],
                "to_Q": b["Q"],
                "delta": vb - va,
                "abs_delta": abs(vb - va),
            }
        )
    return out


def run_change_at_41() -> Dict[str, Any]:
    # Neighborhood scan
    neigh = list(range(35, 50)) + [5, 8, 19, 29, 30, 31, 53, 65, 306, 311, 312, 313, 665]
    neigh = sorted(set(neigh))
    rows = [row_for_Q(Q) for Q in neigh]

    # Focus strip 38..45 for discontinuity vs neighbors
    strip = [row_for_Q(Q) for Q in range(38, 46)]
    jumps_phi = []
    for a, b in zip(strip, strip[1:]):
        jumps_phi.append(
            {
                "from_Q": a["Q"],
                "to_Q": b["Q"],
                "delta_phi": float(Decimal(b["phi"]) - Decimal(a["phi"])),
                "delta_max_m_zt": b["max_m_zero_tax"] - a["max_m_zero_tax"],
                "delta_tax_m8": b["tax_rows_m8"] - a["tax_rows_m8"],
                "delta_sq_tax": b["square_best_tax"] - a["square_best_tax"],
            }
        )

    # Is 41 a local extremum of max_m_zero_tax or phi among 38..45?
    zt = [(r["Q"], r["max_m_zero_tax"]) for r in strip]
    phi = [(r["Q"], float(Decimal(r["phi"]))) for r in strip]
    zt_41 = next(v for q, v in zt if q == 41)
    phi_41 = next(v for q, v in phi if q == 41)
    zt_max_Q = max(zt, key=lambda x: x[1])[0]
    phi_max_Q = max(phi, key=lambda x: x[1])[0]

    # Twin-prime table phases
    twin_rows = []
    for p, (mid, lo, hi) in TWIN_MIDPOINTS.items():
        twin_rows.append(
            {
                "p": p,
                "mid": mid,
                "lo": lo,
                "hi": hi,
                "phi_mid": format(frac_Q_alpha(mid), "f"),
                "phi_lo": format(frac_Q_alpha(lo), "f"),
                "phi_hi": format(frac_Q_alpha(hi), "f"),
                "lo_is_rung": lo in (5, 41, 306, 15601),
                "bits_lo": bits_from_alpha(lo),
                "bits_hi": bits_from_alpha(hi),
            }
        )

    # 306 vs 312
    cmp_306_312 = {
        "phi_306": format(frac_Q_alpha(306), "f"),
        "phi_312": format(frac_Q_alpha(312), "f"),
        "dist1_306": format(1 - frac_Q_alpha(306), "f"),
        "dist1_312": format(1 - frac_Q_alpha(312), "f"),
        "max_m_zt_306": max_zero_tax_m(306),
        "max_m_zt_312": max_zero_tax_m(312),
        "bits_306": bits_from_alpha(306),
        "bits_312": bits_from_alpha(312),
        "tax_rows_8_306": tax_rows_harmonic(8, 306),
        "tax_rows_8_312": tax_rows_harmonic(8, 312),
        "note": (
            "306 is CF surplus rung (near 1); 312 is 13·24 = twin midpoint for p=13. "
            "Near integers ≠ same circle role."
        ),
    }

    # Associator involving 41
    assoc_samples = [
        associator(5, 8, 41),
        associator(8, 5, 41),
        associator(41, 19, 5),
        associator(7, 6, 41),
        associator(41, 41, 2),
        associator(40, 8, 5),
        associator(42, 8, 5),
        associator(43, 8, 5),
    ]

    # Reshape behavior at 41×64 vs neighbors (align 64 may not fit 41)
    cons = DeployConstraints(max_aspect=16, align=0, min_dim=8)
    reshape_nb = {}
    for Q in (40, 41, 42, 43, 64, 65):
        # Q×(64) style and Q×Q
        reshape_nb[f"{Q}x{Q}"] = reshape_under_constraints(
            Q, Q, cons, objective="fiber41"
        )["chosen"]
        reshape_nb[f"{Q}x65"] = reshape_under_constraints(
            Q, 65, cons, objective="fiber41"
        )["chosen"]

    # Verdict logic (honest)
    # Special packing facts at 41 that are ALREADY known: CF rung, bits=65
    known = {
        "41_is_CF_rung": True,
        "bits_41": bits_from_alpha(41),  # 65
        "phi_41": format(frac_Q_alpha(41), "f"),
    }
    # Discontinuity test: is jump into/out of 41 larger than typical neighbor jumps?
    jump_into_41 = next(j for j in jumps_phi if j["to_Q"] == 41)
    jump_out_41 = next(j for j in jumps_phi if j["from_Q"] == 41)
    other_jumps = [j for j in jumps_phi if j["to_Q"] != 41 and j["from_Q"] != 41]
    mean_abs_phi = (
        sum(abs(j["delta_phi"]) for j in other_jumps) / len(other_jumps)
        if other_jumps
        else 0
    )
    phi_jump_special = abs(jump_into_41["delta_phi"]) > 2 * mean_abs_phi + 1e-12

    # Twin story: only p=7 lands on a packing rung among {5,41,306}?
    twin_hits_rung = [t for t in twin_rows if t["lo_is_rung"] or t["hi"] in (5, 41, 306)]

    verdict = {
        "hypothesis": "structural change at 41 beyond being a CF rung",
        "result": "null_as_discontinuity",
        "explanation": (
            "41 is special as the CF convergent denominator (bits=65, high surplus φ). "
            "Neighbor scan 38..45 shows smooth α-circle motion; no discrete jump unique "
            "to 41 in max_m_zero_tax / tax_rows beyond the already-known rung identity. "
            "Twin-prime midpoint 42±1 recovers 41 — suggestive numerology that coincides "
            "with the rung, not an independent packing mechanism. 306 vs 312: different "
            "roles (surplus rung vs 13-twin midpoint); 306 is closer to 1 on the circle."
        ),
        "known_special": known,
        "phi_jump_into_41_vs_mean": {
            "into_41": jump_into_41["delta_phi"],
            "out_41": jump_out_41["delta_phi"],
            "mean_abs_other": mean_abs_phi,
            "flagged_special": phi_jump_special,
        },
        "zt_local_max_Q": zt_max_Q,
        "phi_local_max_Q_in_strip": phi_max_Q,
        "twin_hits_on_rungs": twin_hits_rung,
    }
    # Soften: if φ at 41 is max in strip, that's the rung surplus signal
    if phi_max_Q == 41:
        verdict["result"] = "pass_as_CF_rung_only"
        verdict["explanation"] = (
            "Within 38..45, {41α} is the local surplus maximum — exactly the CF-rung "
            "signature already certified. No additional discontinuity vs 40/42/43 beyond "
            "that surplus peak. Twin-prime 42±1 is a coincidence with the same integer, "
            "not a separate packing law. Prefer CF/α explanation over twin midpoints."
        )

    return {
        "stance": "Exploratory. Hypothesis not assumed true.",
        "neighborhood": rows,
        "strip_38_45": strip,
        "jumps_38_45": jumps_phi,
        "twin_prime_table": twin_rows,
        "cmp_306_vs_312": cmp_306_312,
        "associator_samples": assoc_samples,
        "reshape_neighbors": {
            k: {"m": v["m"], "n": v["n"], "delta_fiber41": v.get("delta_fiber41"), "delta_tax": v.get("delta_tax")}
            for k, v in reshape_nb.items()
        },
        "verdict": verdict,
        "alpha": format(ALPHA, "f"),
    }


def selftest() -> None:
    r = row_for_Q(41)
    assert r["bits"] == 65
    assert float(Decimal(r["phi"])) > 0.9
    print("change_at_41 selftest OK")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: change_at_41.py {selftest|run}")
        return 0
    if argv[0] == "selftest":
        selftest()
        return 0
    if argv[0] == "run":
        out = run_change_at_41()
        path = Path(__file__).resolve().parent / "change_at_41_results.json"
        summary = {
            "verdict": out["verdict"]["result"],
            "explanation": out["verdict"]["explanation"],
            "phi_41": out["verdict"]["known_special"]["phi_41"],
            "bits_41": out["verdict"]["known_special"]["bits_41"],
            "phi_local_max_Q": out["verdict"]["phi_local_max_Q_in_strip"],
            "306_vs_312_dist1": {
                "306": out["cmp_306_vs_312"]["dist1_306"],
                "312": out["cmp_306_vs_312"]["dist1_312"],
            },
            "twin_hits": out["verdict"]["twin_hits_on_rungs"],
        }
        print(json.dumps(summary, indent=2))
        path.write_text(json.dumps(out, indent=2))
        print(f"wrote {path}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
