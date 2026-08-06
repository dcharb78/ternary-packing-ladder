#!/usr/bin/env python3
"""Harmonic / fractional-part geometry of the packing tax.

The additive tax form is a coordinate chart for a log-domain object:

  α = log₂ 3
  bits(Q) = floor(Q·α) + 1                         (exact; 3^Q never 2^k)
  tax_rows(m, n) = m − 1 − floor(m · {n·α})         (exact)
  tax_cols(m, n) = n − 1 − floor(n · {m·α})         (exact)

Zero-tax along rows: tax_rows(m,n)=0  ⟺  {n·α} ≥ 1 − 1/m.

Surplus rungs have {Q·α} near 1 (almost spilled into the next bit);
deficit pieces have {Q·α} near 0. That is the harmonic reading of Law B.

Holonomy (axis choice) needs no bits(MN):
  tax_rows − tax_cols = m·bits(n) − n·bits(m).

Uses Decimal for {Q·α}; size verdicts still cross-checked against exact
integer bits() on feasible sizes. Floats only for display.
"""

from __future__ import annotations

import json
import sys
from decimal import ROUND_FLOOR, Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from tax_graph import SURPLUS_RUNGS, DEFICIT_PIECES, bits, split_tax

getcontext().prec = 120
ALPHA = Decimal(3).ln() / Decimal(2).ln()  # log2(3)


def floor_dec(x: Decimal) -> int:
    return int(x.to_integral_value(rounding=ROUND_FLOOR))


def frac(x: Decimal) -> Decimal:
    return x - x.to_integral_value(rounding=ROUND_FLOOR)


def frac_Q_alpha(Q: int) -> Decimal:
    return frac(Decimal(Q) * ALPHA)


def bits_from_alpha(Q: int) -> int:
    """floor(Q·α)+1 — must match bits(Q)."""
    if Q <= 0:
        return 0
    return floor_dec(Decimal(Q) * ALPHA) + 1


def tax_rows_harmonic(m: int, n: int) -> int:
    """m − 1 − floor(m·{n·α})."""
    if m <= 0 or n <= 0:
        return 0
    return m - 1 - floor_dec(Decimal(m) * frac_Q_alpha(n))


def tax_cols_harmonic(m: int, n: int) -> int:
    if m <= 0 or n <= 0:
        return 0
    return n - 1 - floor_dec(Decimal(n) * frac_Q_alpha(m))


def zero_tax_rows_threshold(m: int) -> Decimal:
    """{n·α} must be ≥ 1 − 1/m for tax_rows(m,n)=0."""
    if m <= 1:
        return Decimal(0)
    return 1 - Decimal(1) / Decimal(m)


def max_zero_tax_m(n: int, m_cap: int = 10_000) -> int:
    """Largest m with tax_rows(m,n)=0 (searched up to m_cap)."""
    fn = frac_Q_alpha(n)
    best = 1
    for m in range(1, m_cap + 1):
        if floor_dec(Decimal(m) * fn) == m - 1:
            best = m
        else:
            # once it fails, larger m only need higher threshold — may recover?
            # Actually threshold 1-1/m increases with m, so once {nα} < 1-1/m it
            # fails for all larger m. Stop.
            break
    return best


def tau(m: int, n: int) -> Decimal:
    """m{nα} − n{mα} — raw torus 2-form before flooring."""
    return Decimal(m) * frac_Q_alpha(n) - Decimal(n) * frac_Q_alpha(m)


def classify_phase(Q: int) -> str:
    f = frac_Q_alpha(Q)
    if f >= Decimal("0.9"):
        return "surplus_near_1"
    if f <= Decimal("0.1"):
        return "deficit_near_0"
    return "mid"


def rung_phase_table(qs: Sequence[int] | None = None) -> List[Dict[str, Any]]:
    if qs is None:
        qs = list(SURPLUS_RUNGS) + list(DEFICIT_PIECES) + [1, 60, 101, 665, 306]
        qs = sorted(set(qs))
    rows = []
    for Q in qs:
        f = frac_Q_alpha(Q)
        rows.append(
            {
                "Q": Q,
                "bits": bits(Q),
                "bits_alpha": bits_from_alpha(Q),
                "frac_Q_alpha": format(f, "f"),
                "phase_class": classify_phase(Q),
                "max_m_zero_tax_rows": max_zero_tax_m(Q),
                "threshold_at_that_m": format(
                    zero_tax_rows_threshold(max_zero_tax_m(Q)), "f"
                ),
            }
        )
    return rows


def selftest() -> int:
    # bits identity
    for Q in list(range(1, 200)) + [306, 665, 2560, 4096, 6912, 11008]:
        assert bits(Q) == bits_from_alpha(Q), Q

    # tax identity on a dense small grid (uses exact bits(mn))
    for m in range(1, 60):
        for n in range(1, 60):
            assert tax_rows_harmonic(m, n) == m * bits(n) - bits(m * n)
            assert tax_cols_harmonic(m, n) == n * bits(m) - bits(m * n)

    # zero-tax iff
    for n in (5, 41, 306):
        fn = frac_Q_alpha(n)
        for m in range(1, max_zero_tax_m(n) + 1):
            assert tax_rows_harmonic(m, n) == 0
            assert fn >= zero_tax_rows_threshold(m) - Decimal("1e-30")
        assert tax_rows_harmonic(max_zero_tax_m(n) + 1, n) >= 1

    # Surplus vs deficit phase classes
    for Q in SURPLUS_RUNGS:
        assert classify_phase(Q) == "surplus_near_1", Q
    assert classify_phase(53) == "deficit_near_0"
    assert classify_phase(665) == "deficit_near_0"
    # 19 is a Law-B remainder with mid fractional part (~0.11), not near 0
    assert classify_phase(19) == "mid"

    # Holonomy matches on BitNet shapes without bits(mn)
    for m, n in ((640, 2560), (2560, 6912), (6912, 2560), (2560, 2560)):
        hol = m * bits(n) - n * bits(m)
        assert tax_rows_harmonic(m, n) - tax_cols_harmonic(m, n) == hol

    print("HARMONIC_TAX PASS")
    return 0


def run_report() -> Dict[str, Any]:
    table = rung_phase_table()
    bitnet = []
    for m, n in ((2560, 2560), (2560, 6912), (6912, 2560), (640, 2560), (2560, 640)):
        hr = tax_rows_harmonic(m, n)
        hc = tax_cols_harmonic(m, n)
        bitnet.append(
            {
                "shape": [m, n],
                "tax_rows": hr,
                "tax_cols": hc,
                "holonomy": hr - hc,
                "best_axis": "rows" if hr <= hc else "cols",
                "frac_n_alpha": format(frac_Q_alpha(n), "f"),
                "frac_m_alpha": format(frac_Q_alpha(m), "f"),
                "tau": format(tau(m, n), "f"),
                "floor_m_frac_n": floor_dec(Decimal(m) * frac_Q_alpha(n)),
                "floor_n_frac_m": floor_dec(Decimal(n) * frac_Q_alpha(m)),
            }
        )
    return {
        "alpha": "log2(3)",
        "identities": {
            "bits_Q": "floor(Q*alpha)+1",
            "tax_rows": "m - 1 - floor(m * {n*alpha})",
            "tax_cols": "n - 1 - floor(n * {m*alpha})",
            "zero_tax_rows": "{n*alpha} >= 1 - 1/m",
        },
        "rung_phases": table,
        "bitnet_shapes": bitnet,
        "reading": (
            "Additive tax is the flat chart; {Q α} on the circle is the intrinsic "
            "coordinate. Surplus rungs sit near 1; deficits near 0. Axis choice "
            "is comparison of floor(m{nα}) and floor(n{mα}) (holonomy)."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path(__file__).resolve().parent / "harmonic_tax_results.json"
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args and args[0] == "selftest":
        return selftest()
    selftest()
    report = run_report()
    dest = Path(args[1]) if len(args) > 1 and args[0] == "run" else out
    if args and args[0] not in ("run", "selftest") and args[0].endswith(".json"):
        dest = Path(args[0])
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"HARMONIC wrote {dest}")
    print("rung phases:")
    for row in report["rung_phases"]:
        print(
            f"  Q={row['Q']:>5}  {{{row['frac_Q_alpha'][:12]}}}  "
            f"{row['phase_class']:<16}  max_m_tax0={row['max_m_zero_tax_rows']}"
        )
    print("BitNet shape holonomy (harmonic):")
    for b in report["bitnet_shapes"]:
        print(
            f"  {b['shape'][0]}x{b['shape'][1]}  best={b['best_axis']}  "
            f"hol={b['holonomy']}  τ={b['tau'][:12]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
