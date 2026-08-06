#!/usr/bin/env python3
"""Organizing-principles lab — test missed operators, not assume them.

Hypotheses (exploratory):
  O1  Typed twin-center  center(c,m)=2mc, sheets=2mc±1  (UFRF contextual primes)
  O2  Rung hit rate on typed sheets vs naive M(p)=p(p−1) and random controls
  O3  Complement pairing among twin table + rungs
  O4  Counterfactual design: snap BitNet dims to nearest surplus (align grid)
  O5  Rung-native frames when a mode length IS a rung (41, 306, …)
  O6  Chiral split tax on twin-derived triples (Law B style)

Kill or keep each operator by byte/tax evidence. Nulls are progress.

Typed contexts (from UFRF audit — hypothesis, not packing theorem):
  C0=0: sheets −1,+1 (signed source/mirror)
  C1=1: center 2 → sheets 1,3 (seed + classical 3)
  C≥3: classical-prime incidence on both sheets
"""

from __future__ import annotations

import json
import math
import struct
import sys
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from frame_formats import theory_bytes_frame
from harmonic_orbit import complementary_pairs, phase_profile
from harmonic_tax import frac_Q_alpha, max_zero_tax_m
from packing_stack import (
    DeployConstraints,
    apply_stack,
    ledger_bytes,
)
from tax_graph import DEFICIT_PIECES, LAW_B_486, LAW_B_665, SURPLUS_RUNGS, split_tax

getcontext().prec = 120

SURPLUS_RUNG_SET = {5, 41, 306, 15601}
DEFICIT_RUNG_SET = {19, 53, 665}
ALIGN = 64
SURPLUS_PHI = Decimal("0.9")

PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
RUNGS_ALL = SURPLUS_RUNG_SET | DEFICIT_RUNG_SET

# Screenshot / audit exemplars: (c, m) → center 2mc
TYPED_EXEMPLARS = (
    (0, 0),  # special: sheets ±1 conceptually; skip numeric
    (1, 1),  # center 2 → 1,3
    (3, 1),  # center 6 → 5,7
    (5, 3),  # center 30 → 29,31
    (7, 3),  # center 42 → 41,43
    (13, 12),  # center 312 → 311,313
)


def tag_Q(Q: int) -> Dict[str, Any]:
    if Q <= 0:
        return {"Q": Q, "phi": None, "surplus": False, "rung_surplus": False, "rung_deficit": False}
    phi = frac_Q_alpha(Q)
    return {
        "Q": Q,
        "phi": format(phi, "f"),
        "surplus": phi >= SURPLUS_PHI,
        "rung_surplus": Q in SURPLUS_RUNG_SET,
        "rung_deficit": Q in DEFICIT_RUNG_SET,
        "max_m_zt": max_zero_tax_m(Q),
    }


def typed_center_sheets(c: int, m: int) -> Dict[str, Any]:
    """center(c,m)=2mc; sheets=2mc±1. Typed by context c (hypothesis)."""
    if c == 0:
        return {
            "c": 0,
            "m": m,
            "mid": 0,
            "lo": tag_Q(-1),
            "hi": tag_Q(1),
            "context": "signed_source_mirror",
            "any_rung": False,
        }
    mid = 2 * m * c
    lo, hi = mid - 1, mid + 1
    lo_t, hi_t = tag_Q(lo), tag_Q(hi)
    if c == 1:
        ctx = "seed_plus_classical_3"
    else:
        ctx = "classical_prime_incidence"
    return {
        "c": c,
        "m": m,
        "mid": mid,
        "lo": lo_t,
        "hi": hi_t,
        "mid_tag": tag_Q(mid),
        "context": ctx,
        "any_rung": lo in RUNGS_ALL or hi in RUNGS_ALL,
        "lo_rung": lo in RUNGS_ALL,
        "hi_rung": hi in RUNGS_ALL,
    }


def naive_p_pm1(p: int) -> Dict[str, Any]:
    """Legacy (incorrect for screenshot) M(p)=p(p−1). Kept as ablation."""
    mid = p * (p - 1)
    lo, hi = mid - 1, mid + 1
    return {
        "p": p,
        "mid": mid,
        "lo": tag_Q(lo),
        "hi": tag_Q(hi),
        "any_rung": lo in RUNGS_ALL or hi in RUNGS_ALL,
    }


def run_O1_O2() -> Dict[str, Any]:
    # Exemplars from typed audit / screenshot
    exemplars = [typed_center_sheets(c, m) for c, m in TYPED_EXEMPLARS if not (c == 0)]
    exemplar_hits = [r for r in exemplars if r["any_rung"]]

    # Systematic sweep: c in primes∪{1}, m=1..20; collect unique rung hits
    sweep = []
    rung_hits_detail = []
    seen_mid = set()
    for c in (1,) + PRIMES:
        for m in range(1, 21):
            row = typed_center_sheets(c, m)
            if row["mid"] in seen_mid or row["mid"] <= 0:
                continue
            seen_mid.add(row["mid"])
            sweep.append(row)
            if row["any_rung"]:
                rung_hits_detail.append(
                    {
                        "c": c,
                        "m": m,
                        "mid": row["mid"],
                        "lo": row["lo"]["Q"],
                        "hi": row["hi"]["Q"],
                        "which": "lo" if row["lo_rung"] else "hi",
                        "rung": row["lo"]["Q"] if row["lo_rung"] else row["hi"]["Q"],
                    }
                )

    # Naive ablation
    naive = [naive_p_pm1(p) for p in PRIMES if p * (p - 1) > 2]
    naive_hits = sum(1 for r in naive if r["any_rung"])

    # Random control: same count of mids, sheets mid±1
    rng_mids = [2 * k * 11 + 7 for k in range(1, 41)]
    random_rung = sum(
        1 for mid in rng_mids for q in (mid - 1, mid + 1) if q in RUNGS_ALL
    )

    # Can every surplus/deficit rung appear as a typed sheet?
    rungs_as_sheets = []
    for R in sorted(RUNGS_ALL):
        # Solve 2mc ± 1 = R → 2mc = R∓1 → mc = (R∓1)/2
        # Requires R odd (2mc always even ⇒ sheets always odd).
        ways = []
        for sheet_side, mid in (("lo", R + 1), ("hi", R - 1)):
            if mid % 2:
                continue
            half = mid // 2  # = m*c
            for c in (1,) + PRIMES:
                if c > 0 and half % c == 0:
                    m = half // c
                    if m >= 1:
                        ways.append({"c": c, "m": m, "mid": mid, "as": sheet_side})
        rungs_as_sheets.append(
            {
                "rung": R,
                "family": "surplus" if R in SURPLUS_RUNG_SET else "deficit",
                "parity": "odd" if R % 2 else "even",
                "n_ways": len(ways),
                "ways_head": ways[:6],
                "representable": len(ways) > 0,
            }
        )

    odd_ok = all(r["representable"] for r in rungs_as_sheets if r["parity"] == "odd")
    even_miss = [r["rung"] for r in rungs_as_sheets if r["parity"] == "even"]
    return {
        "operator": "center(c,m)=2mc, sheets=2mc±1 (typed by c)",
        "typed_contexts": {
            "C0": "signed ±1 (not classical-prime census)",
            "C1": "center 2 → sheets 1,3",
            "C_ge_3": "classical-prime incidence on both sheets",
        },
        "exemplars": exemplars,
        "exemplar_rung_hits": exemplar_hits,
        "sweep_n_mids": len(sweep),
        "sweep_rung_hits": rung_hits_detail,
        "n_sweep_rung_hits": len(rung_hits_detail),
        "naive_p_pm1_hits": naive_hits,
        "naive_note": "p(p-1) is NOT the screenshot rule (5→20 not 30); ablation only",
        "rung_hit_random_control": random_rung,
        "rungs_as_typed_sheets": rungs_as_sheets,
        "odd_rungs_all_representable": odd_ok,
        "even_rungs_impossible_as_sheets": even_miss,
        "structural_fact": (
            "2mc is always even ⇒ sheets 2mc±1 are always odd. "
            "Even packing rung 306 can never be a typed sheet — it lives in "
            "Law B / CF composition, not the twin-center operator."
        ),
        "verdict": (
            "typed_center_covers_odd_rungs_misses_even_306"
            if odd_ok and even_miss == [306]
            else "typed_center_partial"
            if len(exemplar_hits) >= 2
            else "typed_center_coincidence_only"
        ),
        "note": (
            "Correction: census omitting C0/C1 is a typing filter, not a missing start. "
            "Odd packing rungs 5,19,41,53,665,15601 are typed sheets for some (c,m). "
            "Even rung 306 is a different operator (CF / Law B)."
        ),
    }


def run_O3() -> Dict[str, Any]:
    """Complement pairs among typed sheets + certified rungs."""
    qs = sorted(
        SURPLUS_RUNG_SET
        | DEFICIT_RUNG_SET
        | {1, 3, 5, 7, 29, 31, 41, 43, 311, 312, 313}
    )
    pairs = complementary_pairs(qs, tol=Decimal("0.02"))
    # Typed sheets from exemplars + small sweep
    twin_qs = []
    for c, m in TYPED_EXEMPLARS:
        if c == 0:
            continue
        row = typed_center_sheets(c, m)
        twin_qs.extend([row["mid"], row["lo"]["Q"], row["hi"]["Q"]])
    for c in (1, 3, 5, 7, 13):
        for m in range(1, 13):
            row = typed_center_sheets(c, m)
            twin_qs.extend([row["lo"]["Q"], row["hi"]["Q"]])
    twin_qs = sorted(set(q for q in twin_qs if q and q > 0))
    twin_pairs = complementary_pairs(twin_qs, tol=Decimal("0.02"))
    # Cross: typed sheet with rungs
    cross = []
    for t in twin_qs:
        for r in sorted(RUNGS_ALL):
            phi_t = frac_Q_alpha(t)
            phi_r = frac_Q_alpha(r)
            d = abs(phi_t + phi_r - 1)
            if d < Decimal("0.02"):
                led = ledger_bytes(min(t, r), max(t, r))
                cross.append(
                    {
                        "a": t,
                        "b": r,
                        "complement_dist": format(d, "f"),
                        "best_tax": led["best_tax"],
                        "best_fiber_41": led["best_fiber_41"],
                    }
                )
    cross.sort(key=lambda x: x["best_tax"])
    return {
        "operator": "complement φ+ψ≈1 on typed sheets × rungs",
        "n_rungs_pairs": len(pairs),
        "n_twin_internal_pairs": len(twin_pairs),
        "best_cross_twin_rung": cross[:12],
        "verdict": (
            "complement_links_twin_to_rungs"
            if cross and cross[0]["best_tax"] <= 2
            else "complement_weak_on_twin_table"
        ),
    }


def nearest_surplus_on_grid(Q: int, align: int = ALIGN, max_delta: int = 128) -> Optional[Dict[str, Any]]:
    best = None
    for d in range(0, max_delta + 1):
        for sign in (-1, 1) if d else (0,):
            Qp = Q + sign * d
            if Qp < align or Qp % align:
                continue
            phi = frac_Q_alpha(Qp)
            if phi < SURPLUS_PHI:
                continue
            cand = {"Q": Qp, "delta": Qp - Q, "phi": format(phi, "f")}
            if best is None or abs(cand["delta"]) < abs(best["delta"]):
                best = cand
    return best


def run_O4() -> Dict[str, Any]:
    """Counterfactual: BitNet unique dims snapped to nearest surplus (no pad inflate)."""
    bitnet_dims = [640, 2560, 6912]
    cons = DeployConstraints(max_aspect=16, align=ALIGN)
    rows = []
    total_before_f41 = 0
    total_after_f41 = 0
    total_before_58 = 0
    total_after_58 = 0
    for Q in bitnet_dims:
        snap = nearest_surplus_on_grid(Q)
        before = ledger_bytes(Q, Q)
        if snap:
            after = ledger_bytes(snap["Q"], snap["Q"])
            stack = apply_stack(
                snap["Q"],
                snap["Q"],
                cons,
                pad_gate="never",
                reshape_objective="fiber41",
            )
        else:
            after = before
            stack = apply_stack(Q, Q, cons, pad_gate="never", reshape_objective="fiber41")
        # Count as if all tensors at this width (weighted by BitNet multiplicity)
        mult = {640: 60, 2560: 60, 6912: 30}.get(Q, 1)
        if Q == 6912:
            mult = 90  # 30+60 for both orientations approx
        total_before_f41 += before["best_fiber_41"] * mult
        total_after_f41 += (
            stack["after"]["best_fiber_41"] if snap else stack["after"]["best_fiber_41"]
        ) * mult
        total_before_58 += before["flat_5_8"] * mult
        total_after_58 += stack["after"]["flat_5_8"] * mult
        rows.append(
            {
                "original": Q,
                "snap": snap,
                "mult_approx": mult,
                "before_f41": before["best_fiber_41"],
                "after_f41": stack["after"]["best_fiber_41"],
                "before_58": before["flat_5_8"],
                "after_58": stack["after"]["flat_5_8"],
                "reshape": stack["reshape"]["chosen"],
            }
        )
    return {
        "operator": "design-time snap to surplus grid (|Δ|≤128, no post-hoc pad)",
        "rows": rows,
        "aggregate_delta_f41": total_after_f41 - total_before_f41,
        "aggregate_delta_58": total_after_58 - total_before_58,
        "verdict": (
            "design_snap_helps_fiber41"
            if total_after_f41 < total_before_f41
            else "design_snap_null"
        ),
        "note": "Counterfactual width choice only; does not rewrite BitNet weights",
    }


def run_O5() -> Dict[str, Any]:
    """When mode length is a rung Q, compare rung-native frame vs flat/fiber."""
    led41 = ledger_bytes(41, 65)
    frame486_306 = theory_bytes_frame(LAW_B_486, 306)
    frame665_665 = theory_bytes_frame(LAW_B_665, 665)
    flat306 = ledger_bytes(306, 306)["flat_5_8"]
    flat665 = ledger_bytes(665, 665)["flat_5_8"]
    cases = [
        {
            "Q_rung": 41,
            "pair_shape": [41, 65],
            "flat_5_8": led41["flat_5_8"],
            "best_fiber_41": led41["best_fiber_41"],
            "best_tax": led41["best_tax"],
            "note": "canonical rung pair (41,65)",
        },
        {
            "Q_rung": 306,
            "frame_486_at_306": frame486_306,
            "flat_5_8_at_306": flat306,
            "frame_vs_flat_delta": frame486_306 - flat306,
            "tax_486": split_tax(LAW_B_486),
        },
        {
            "Q_rung": 665,
            "frame_665_at_665": frame665_665,
            "flat_5_8_at_665": flat665,
            "frame_vs_flat_delta": frame665_665 - flat665,
            "tax_665": split_tax(LAW_B_665),
        },
    ]
    return {
        "operator": "rung-native Law B frames vs flat on rung lengths",
        "cases": cases,
        "verdict": (
            "frames_lose_to_flat58_on_same_length"
            if frame486_306 > flat306 and frame665_665 > flat665
            else "frames_win_some_rung_lengths"
        ),
        "note": "486/665 are chiral assemblies; flat 5_8 still density knee unstructured",
    }


def run_O6() -> Dict[str, Any]:
    """Chiral splits on numbers from twin table vs random."""
    splits = []
    # Twin-derived: 41+43 around 42; 5+7 around 6
    for mid, parts in [(42, (41, 43)), (6, (5, 7)), (312, (311, 313)), (30, (29, 31))]:
        a, b = parts
        led = ledger_bytes(a, b)
        splits.append(
            {
                "mid": mid,
                "pair": [a, b],
                "best_tax": led["best_tax"],
                "complement_dist": format(abs(frac_Q_alpha(a) + frac_Q_alpha(b) - 1), "f"),
                "best_fiber_41": led["best_fiber_41"],
            }
        )
    # Law B reference
    splits.append(
        {
            "mid": "306=7*41+19",
            "pair": [41, 306],
            "best_tax": ledger_bytes(41, 306)["best_tax"],
            "complement_dist": format(
                abs(frac_Q_alpha(41) + frac_Q_alpha(306) - 1), "f"
            ),
            "best_fiber_41": ledger_bytes(41, 306)["best_fiber_41"],
            "reference": "Law_B_chiral",
        }
    )
    splits.sort(key=lambda x: x["best_tax"])
    return {
        "operator": "chiral rectangle tax on twin pairs vs Law B",
        "splits": splits,
        "verdict": (
            "twin_pairs_low_tax"
            if splits[0]["mid"] in (42, 6) and splits[0]["best_tax"] <= 5
            else "twin_pairs_not_auto_low_tax"
        ),
    }


def run_all() -> Dict[str, Any]:
    o1 = run_O1_O2()
    o3 = run_O3()
    o4 = run_O4()
    o5 = run_O5()
    o6 = run_O6()

    survivors = []
    kills = []
    for oid, block in [("O1_O2", o1), ("O3", o3), ("O4", o4), ("O5", o5), ("O6", o6)]:
        v = block.get("verdict", "")
        if "null" in v or "coincidence" in v or "weak" in v or "not_auto" in v:
            kills.append({"id": oid, "verdict": v, "operator": block.get("operator")})
        else:
            survivors.append({"id": oid, "verdict": v, "operator": block.get("operator")})

    return {
        "stance": "Exploratory operators. Validate by bytes/tax; nulls are progress.",
        "O1_O2_twin_operator": o1,
        "O3_complement": o3,
        "O4_counterfactual_snap": o4,
        "O5_rung_native": o5,
        "O6_chiral_twin": o6,
        "survivors": survivors,
        "kills": kills,
        "synthesis": {
            "usable_operators": [s["operator"] for s in survivors],
            "deferred": [k["operator"] for k in kills],
        },
    }


def selftest() -> None:
    # Typed rule: (c=7,m=3) → center 42 → sheets 41,43
    t = typed_center_sheets(7, 3)
    assert t["mid"] == 42 and t["lo"]["Q"] == 41 and t["hi"]["Q"] == 43
    # Screenshot (c=5,m=3) → 30±1, not p(p-1)=20
    t5 = typed_center_sheets(5, 3)
    assert t5["mid"] == 30 and t5["lo"]["Q"] == 29
    # C1 seed layer
    t1 = typed_center_sheets(1, 1)
    assert t1["mid"] == 2 and t1["lo"]["Q"] == 1 and t1["hi"]["Q"] == 3
    out = run_O1_O2()
    assert out["sweep_n_mids"] >= 5
    assert out["odd_rungs_all_representable"] is True
    assert 306 in out["even_rungs_impossible_as_sheets"]
    print("organizing_principles selftest OK")


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: organizing_principles.py {selftest|run}")
        return 0
    if argv[0] == "selftest":
        selftest()
        return 0
    if argv[0] == "run":
        out = run_all()
        path = Path(__file__).resolve().parent / "organizing_principles_results.json"
        o1 = out["O1_O2_twin_operator"]
        summary = {
            "survivors": out["survivors"],
            "kills": out["kills"],
            "synthesis": out["synthesis"],
            "typed_center": {
                "verdict": o1["verdict"],
                "exemplar_hits": [
                    {
                        "c": r["c"],
                        "m": r["m"],
                        "mid": r["mid"],
                        "lo": r["lo"]["Q"],
                        "hi": r["hi"]["Q"],
                    }
                    for r in o1["exemplar_rung_hits"]
                ],
                "n_sweep_rung_hits": o1["n_sweep_rung_hits"],
                "naive_p_pm1_hits": o1["naive_p_pm1_hits"],
                "all_rungs_representable": o1.get("odd_rungs_all_representable"),
                "even_impossible": o1.get("even_rungs_impossible_as_sheets"),
                "structural_fact": o1.get("structural_fact"),
                "rungs_as_sheets": o1["rungs_as_typed_sheets"],
            },
            "O4_delta_f41": out["O4_counterfactual_snap"]["aggregate_delta_f41"],
            "O6_best": out["O6_chiral_twin"]["splits"][0],
        }
        print(json.dumps(summary, indent=2))
        path.write_text(json.dumps(out, indent=2))
        print(f"wrote {path}")
        return 0
    print(f"unknown: {argv[0]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
