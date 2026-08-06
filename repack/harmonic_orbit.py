#!/usr/bin/env python3
"""Fractional-part orbits and complementary phases — multi-D tax geometry.

The exact value of φ = {Q·α} (not merely “near 0/1”) controls how taxes
combine under integer multiplication by tensor dimensions:

  m · φ  →  floor(m·φ) determines tax_rows(m,Q) = m − 1 − floor(m·φ)
  {m · φ} = m·φ − floor(m·φ) is the residual phase after m copies

Surplus φ↗1: m·φ stays just below successive integers for many m (tax 0).
Deficit φ↘0: floor stays 0 until m ~ 1/φ; tax grows as m−1 (hungry).
Mid φ (~0.11 for Q=19): scales more linearly; overflow timing is irregular.

Complementary pairs: φ(a)+φ(b) ≈ 1  → especially clean Law-B cancellations
(e.g. 306+53 ≈ 1.0015 explains the 665 zero-tax identity).
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from harmonic_tax import (
    ALPHA,
    floor_dec,
    frac,
    frac_Q_alpha,
    tax_cols_harmonic,
    tax_rows_harmonic,
)
from tax_graph import DEFICIT_PIECES, SURPLUS_RUNGS

getcontext().prec = 100

# Lengths of geometric interest
ORBIT_QS: Tuple[int, ...] = (
    5,
    19,
    41,
    53,
    306,
    665,
    640,
    2560,
    6912,
    4096,
    11008,
)


def phase_profile(Q: int) -> Dict[str, Any]:
    phi = frac_Q_alpha(Q)
    return {
        "Q": Q,
        "phi": format(phi, "f"),
        "dist_to_0": format(phi, "e"),
        "dist_to_1": format(1 - phi, "e"),
        "side": (
            "from_below_1"
            if phi >= Decimal("0.5")
            else "from_above_0"
        ),
        "class": (
            "surplus"
            if phi >= Decimal("0.9")
            else ("deficit" if phi <= Decimal("0.1") else "mid")
        ),
    }


def orbit(Q: int, m_max: int = 24) -> List[Dict[str, Any]]:
    """Orbit of φ={Qα} under multiplication by m=1..m_max."""
    phi = frac_Q_alpha(Q)
    out = []
    for m in range(1, m_max + 1):
        prod = Decimal(m) * phi
        fl = floor_dec(prod)
        fr = frac(prod)
        tax = tax_rows_harmonic(m, Q)
        # Identity check
        assert tax == m - 1 - fl
        out.append(
            {
                "m": m,
                "m_phi": format(prod, "f"),
                "floor": fl,
                "frac_m_phi": format(fr, "f"),
                "tax_rows": tax,
                "overflowed": fl >= 1,
            }
        )
    return out


def first_overflow_m(Q: int, m_cap: int = 100_000) -> int:
    """Smallest m≥2 with floor(m·φ)≥1; 1 if φ≥1 (impossible)."""
    phi = frac_Q_alpha(Q)
    if phi == 0:
        return m_cap
    # floor(m*phi)>=1 iff m*phi >= 1 iff m >= ceil(1/phi)
    # exact via loop for modest caps
    for m in range(1, m_cap + 1):
        if floor_dec(Decimal(m) * phi) >= 1:
            return m
    return m_cap


def complementary_pairs(
    qs: Sequence[int],
    tol: Decimal = Decimal("0.05"),
) -> List[Dict[str, Any]]:
    """Pairs whose fractional parts sum nearly to an integer (esp. 1)."""
    pairs = []
    phis = {q: frac_Q_alpha(q) for q in qs}
    qs = list(qs)
    for i, a in enumerate(qs):
        for b in qs[i:]:
            s = phis[a] + phis[b]
            # distance to nearest integer
            nearest = s.to_integral_value(rounding="ROUND_HALF_UP")
            dist = abs(s - nearest)
            dist_to_1 = abs(s - 1)
            if dist <= tol or dist_to_1 <= tol:
                pairs.append(
                    {
                        "a": a,
                        "b": b,
                        "phi_a": format(phis[a], "f"),
                        "phi_b": format(phis[b], "f"),
                        "sum": format(s, "f"),
                        "dist_to_1": format(dist_to_1, "f"),
                        "dist_to_integer": format(dist, "f"),
                        "law_b_hint": (
                            "665=2×306+53"
                            if {a, b} == {306, 53}
                            else (
                                "486 uses 41+19"
                                if {a, b} == {41, 19}
                                else None
                            )
                        ),
                    }
                )
    pairs.sort(key=lambda p: float(p["dist_to_1"]))
    return pairs


def axis_interaction(m: int, n: int) -> Dict[str, Any]:
    """How each mode's φ is multiplied by the other mode's length."""
    phi_n = frac_Q_alpha(n)
    phi_m = frac_Q_alpha(m)
    return {
        "shape": [m, n],
        "phi_m": format(phi_m, "f"),
        "phi_n": format(phi_n, "f"),
        "m_times_phi_n": format(Decimal(m) * phi_n, "f"),
        "n_times_phi_m": format(Decimal(n) * phi_m, "f"),
        "floor_m_phi_n": floor_dec(Decimal(m) * phi_n),
        "floor_n_phi_m": floor_dec(Decimal(n) * phi_m),
        "tax_rows": tax_rows_harmonic(m, n),
        "tax_cols": tax_cols_harmonic(m, n),
        "best_axis": (
            "rows"
            if tax_rows_harmonic(m, n) <= tax_cols_harmonic(m, n)
            else "cols"
        ),
        "reading": (
            "Orientation that multiplies the *larger* φ tends to accumulate "
            "more floor mass → lower tax on that axis."
        ),
    }


def selftest() -> int:
    # Surplus: many tax-0 steps; first overflow m is large
    assert first_overflow_m(41) == 2  # 2*0.983 → floor 1, but tax still 0
    # For surplus, tax stays 0 while floor = m-1; first tax>0 later
    orb41 = orbit(41, 60)
    assert all(r["tax_rows"] == 0 for r in orb41)
    # Deficit 53: tax = m-1 for early m (floor stays 0)
    orb53 = orbit(53, 20)
    assert orb53[0]["tax_rows"] == 0
    assert all(orb53[m - 1]["tax_rows"] == m - 1 for m in range(2, 21))
    # Mid 19: tax grows until overflow then continues
    orb19 = orbit(19, 12)
    assert orb19[0]["tax_rows"] == 0
    assert orb19[1]["tax_rows"] == 1
    # Complementary 306+53
    comps = complementary_pairs([306, 53, 41, 19], tol=Decimal("0.02"))
    assert any(p["a"] == 53 and p["b"] == 306 for p in comps) or any(
        p["a"] == 306 and p["b"] == 53 for p in comps
    )
    # BitNet axis interaction
    ai = axis_interaction(6912, 2560)
    assert ai["best_axis"] == "cols"
    print("HARMONIC_ORBIT PASS")
    return 0


def run() -> Dict[str, Any]:
    profiles = [phase_profile(Q) for Q in ORBIT_QS]
    orbits = {str(Q): orbit(Q, 16) for Q in (5, 19, 41, 53, 306, 665, 2560)}
    comps = complementary_pairs(
        list(SURPLUS_RUNGS) + list(DEFICIT_PIECES) + [665, 19, 640, 2560, 6912],
        tol=Decimal("0.12"),
    )
    bitnet = [
        axis_interaction(2560, 2560),
        axis_interaction(2560, 6912),
        axis_interaction(6912, 2560),
        axis_interaction(640, 2560),
    ]
    return {
        "thesis": (
            "Exact {Qα} decimals are circle coordinates; multi-D tax is the "
            "orbit under multiplication by mode lengths. Complementary phases "
            "explain Law-B cancellations; surplus/deficit sides explain chirality."
        ),
        "profiles": profiles,
        "orbits_preview": orbits,
        "complementary_pairs": comps[:24],
        "bitnet_axis_interactions": bitnet,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path(__file__).resolve().parent / "harmonic_orbit_results.json"
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args and args[0] == "selftest":
        return selftest()
    selftest()
    report = run()
    dest = Path(args[1]) if len(args) > 1 else out
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"HARMONIC_ORBIT wrote {dest}")
    print("phase profiles:")
    for p in report["profiles"]:
        print(
            f"  Q={p['Q']:>5}  class={p['class']:<8}  "
            f"phi≈{p['phi'][:12]}  side={p['side']}"
        )
    print("best complementary pairs (by dist_to_1):")
    for c in report["complementary_pairs"][:8]:
        hint = f"  [{c['law_b_hint']}]" if c.get("law_b_hint") else ""
        print(
            f"  {c['a']}+{c['b']}: sum≈{c['sum'][:10]}  "
            f"d1={c['dist_to_1'][:10]}{hint}"
        )
    print("BitNet interactions:")
    for b in report["bitnet_axis_interactions"]:
        print(
            f"  {b['shape'][0]}x{b['shape'][1]}  best={b['best_axis']}  "
            f"tax_r={b['tax_rows']} tax_c={b['tax_cols']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
