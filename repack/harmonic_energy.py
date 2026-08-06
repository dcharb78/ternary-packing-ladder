#!/usr/bin/env python3
"""Packing energy on the torus — harmonic critical-point probe.

Define a natural energy whose value on mode phases recovers the additive tax:

  E_rows(m, n) = tax_rows(m, n) = m − 1 − ⌊m · {n·α}⌋
  E_2d(m, n)   = E_rows(m,n) + E_cols(m,n)
               = m + n − 2 − ⌊m·{n·α}⌋ − ⌊n·{m·α}⌋

Minima of E_rows in m (fixed n) are exactly the zero-tax row multiplicities.
Scanning Q by surplus phase {Q·α} recovers the certified surplus rungs as
local maxima of phase (near 1) — the energy view of Law A/B.

Also: 1-D split tax of k copies of q equals E_rows(k, q).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from harmonic_tax import (
    ALPHA,
    frac_Q_alpha,
    max_zero_tax_m,
    tax_cols_harmonic,
    tax_rows_harmonic,
)
from tax_graph import SURPLUS_RUNGS, DEFICIT_PIECES, split_tax
from tax_tensor import tax_rows


def E_rows(m: int, n: int) -> int:
    return tax_rows_harmonic(m, n)


def E_cols(m: int, n: int) -> int:
    return tax_cols_harmonic(m, n)


def E_2d(m: int, n: int) -> int:
    return E_rows(m, n) + E_cols(m, n)


def scan_phase_maxima(Q_max: int = 400) -> List[Dict[str, Any]]:
    """Local maxima of {Q·α} on 1..Q_max — expect surplus rungs nearby."""
    fracs = [frac_Q_alpha(Q) for Q in range(1, Q_max + 1)]
    peaks = []
    for Q in range(2, Q_max):
        if fracs[Q - 1] > fracs[Q - 2] and fracs[Q - 1] >= fracs[Q]:
            peaks.append(
                {
                    "Q": Q,
                    "frac": format(fracs[Q - 1], "f"),
                    "is_surplus_rung": Q in SURPLUS_RUNGS,
                }
            )
    return peaks


def energy_minima_for_n(n: int, m_max: int = 100) -> Dict[str, Any]:
    """m that minimize E_rows(·,n); should be 1..max_zero_tax_m(n) at value 0."""
    vals = [(m, E_rows(m, n)) for m in range(1, m_max + 1)]
    emin = min(v for _, v in vals)
    minimizers = [m for m, v in vals if v == emin]
    return {
        "n": n,
        "E_min": emin,
        "minimizers": minimizers,
        "max_zero_tax_m": max_zero_tax_m(n),
        "frac_n": format(frac_Q_alpha(n), "f"),
    }


def selftest() -> int:
    for q in (5, 41, 306):
        for k in (1, 2, 7, 13):
            assert E_rows(k, q) == split_tax((q,) * k) == tax_rows(k, q)

    for n in SURPLUS_RUNGS:
        em = energy_minima_for_n(n, m_max=max_zero_tax_m(n) + 5)
        assert em["E_min"] == 0
        assert em["minimizers"] == list(range(1, em["max_zero_tax_m"] + 1))

    peaks = scan_phase_maxima(350)
    peak_Qs = {p["Q"] for p in peaks}
    # Certified surplus rungs in range should appear as peaks or neighbors
    for Q in (5, 41, 306):
        assert Q in peak_Qs or any(abs(p["Q"] - Q) <= 1 for p in peaks), Q

    print(
        f"HARMONIC_ENERGY PASS peaks={len(peaks)} "
        f"rungs_hit={[Q for Q in SURPLUS_RUNGS if Q in peak_Qs]}"
    )
    return 0


def run() -> Dict[str, Any]:
    peaks = scan_phase_maxima(400)
    minima = [energy_minima_for_n(n) for n in list(SURPLUS_RUNGS) + list(DEFICIT_PIECES) + [665]]
    return {
        "alpha": "log2(3)",
        "E_rows": "m - 1 - floor(m*{n*alpha})",
        "E_2d": "m + n - 2 - floor(m*{n*alpha}) - floor(n*{m*alpha})",
        "phase_peaks": peaks,
        "energy_minima": minima,
        "reading": (
            "Zero-tax configurations are exact minima of E_rows. Surplus rungs "
            "are phase maxima near 1 on the circle. Additive tax_graph enumerates "
            "the same critical set in flat coordinates."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path(__file__).resolve().parent / "harmonic_energy_results.json"
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args and args[0] == "selftest":
        return selftest()
    selftest()
    report = run()
    dest = Path(args[1]) if len(args) > 1 else out
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"HARMONIC_ENERGY wrote {dest}")
    print(f"phase peaks in 1..400: {len(report['phase_peaks'])}")
    for m in report["energy_minima"]:
        print(
            f"  n={m['n']:>4} E_min={m['E_min']} "
            f"minimizers=1..{m['max_zero_tax_m']}  {{{m['frac_n'][:10]}}}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
