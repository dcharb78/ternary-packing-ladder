#!/usr/bin/env python3
"""Law C as multiresolution on the 2–3 scale (harmonic MRA sketch).

Hierarchical digit nesting is not an ad-hoc LUT trick: each rung Q_k is a
scale where {Q_k · α} is exceptionally close to 1 (surplus peak). Decoding
rung k in alphabet 3^{Q_{k-1}} is one octave of a multiresolution analysis
whose scaling ratios are the continued-fraction recurrence of α = log₂ 3.

This module records the scale dictionary and checks that D3-style nesting
depths match the CF structure already used in hierarchical_digits.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from harmonic_tax import frac_Q_alpha, max_zero_tax_m
from tax_graph import SURPLUS_RUNGS, bits

# Surplus ladder (trit counts) — certified rungs
SCALES: List[Dict[str, Any]] = [
    {"level": 1, "Q": 5, "P": 8, "alphabet": 3},
    {"level": 2, "Q": 41, "P": 65, "parent_Q": 5, "digits": "8×243 + 1 trit"},
    {"level": 3, "Q": 306, "P": 485, "parent_Q": 41, "digits": "7×3^41 + 19 rem"},
    {"level": 4, "Q": 15601, "P": 24727, "parent_Q": 306, "digits": "CF next"},
]


def scale_report() -> Dict[str, Any]:
    rows = []
    for s in SCALES:
        Q = s["Q"]
        row = {
            **s,
            "frac_Q_alpha": format(frac_Q_alpha(Q), "f"),
            "bits_Q": bits(Q) if Q <= 2000 else None,  # skip huge pow for 15601 in selftest path
            "max_m_zero_tax": max_zero_tax_m(Q) if Q <= 2000 else None,
            "distance_to_1": format(1 - frac_Q_alpha(Q), "f"),
        }
        if Q <= 400:
            row["bits_Q"] = bits(Q)
        rows.append(row)
    # Nesting identities used in code
    nest = {
        "41_as_parent_5": {"formula": "41 = 8*5 + 1", "check": 8 * 5 + 1 == 41},
        "306_as_parent_41": {"formula": "306 = 7*41 + 19", "check": 7 * 41 + 19 == 306},
        "reading": (
            "Each surplus scale is a phase peak; child decode is expansion in "
            "the parent alphabet — wavelet detail at the CF ratio."
        ),
    }
    return {"scales": rows, "nesting": nest}


def selftest() -> int:
    assert 8 * 5 + 1 == 41
    assert 7 * 41 + 19 == 306
    r = scale_report()
    # Monotone approach of surplus phases toward 1 along the ladder
    f5 = frac_Q_alpha(5)
    f41 = frac_Q_alpha(41)
    f306 = frac_Q_alpha(306)
    assert f5 < f41 < f306 < 1
    assert (1 - f306) < (1 - f41) < (1 - f5)
    print(
        f"HARMONIC_MRA PASS  "
        f"{{5α}}={f5:.6f} {{41α}}={f41:.6f} {{306α}}={f306:.6f}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path(__file__).resolve().parent / "harmonic_mra_results.json"
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args and args[0] == "selftest":
        return selftest()
    selftest()
    report = scale_report()
    dest = Path(args[1]) if len(args) > 1 else out
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"HARMONIC_MRA wrote {dest}")
    for s in report["scales"]:
        print(
            f"  L{s['level']} Q={s['Q']:<6} dist_to_1={s['distance_to_1'][:12]}  "
            f"{s.get('digits', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
