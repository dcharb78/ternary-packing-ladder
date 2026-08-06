#!/usr/bin/env python3
"""Seed → amplify → harmonize process-language probes (exploratory).

Hypothesis framing for Phase 4 Collatz↔packing bridge. Every check must
reduce to settled circle identities (tax_rows, complements) or report null.

Probes:
  1. Seed catalog — {Qα} for surplus/deficit generators; side class
  2. Amplify fidelity — tax_rows == m−1−⌊m·{nα}⌋ on random (m,n)
  3. Harmonize scan — |φi+φj−1| vs row/col tax
  4. Half-flip audit — tax vs |φ−½| and vs |1−φ|; self-complement band
  5. Breathing cloud — BS = Σ({Qi α}−½); correlate with tax / pad delta
  6. Return-map toy — residual phase → next amplify; tax trajectory
  7. No-compensation — mid-phase mode fixed; do other modes cancel tax?

See COLLATZ_BRIDGE.md. Not theorems.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harmonic_multi import DEFAULT_CKPT, bitnet_phase_cloud, multi_mode_pad
from harmonic_tax import (
    floor_dec,
    frac,
    frac_Q_alpha,
    tax_cols_harmonic,
    tax_rows_harmonic,
)
from tax_graph import DEFICIT_PIECES, SURPLUS_RUNGS, bits

getcontext().prec = 120

SEED_QS: Tuple[int, ...] = (
    *SURPLUS_RUNGS,
    *DEFICIT_PIECES,
    665,
    640,
    2560,
    6912,
    1,
    60,
    101,
)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def classify_seed_side(phi: Decimal) -> str:
    """→0 / mid / →1 / ~½ — finer than harmonic_tax.classify_phase."""
    if abs(phi - Decimal("0.5")) <= Decimal("0.05"):
        return "near_half"
    if phi >= Decimal("0.9"):
        return "toward_1"
    if phi <= Decimal("0.1"):
        return "toward_0"
    return "mid"


# ---------------------------------------------------------------------------
# 1. Seed catalog
# ---------------------------------------------------------------------------

def seed_catalog(qs: Sequence[int] | None = None) -> List[Dict[str, Any]]:
    qs = list(qs) if qs is not None else list(SEED_QS)
    rows = []
    for Q in sorted(set(qs)):
        phi = frac_Q_alpha(Q)
        rows.append(
            {
                "Q": Q,
                "phi": format(phi, "f"),
                "side": classify_seed_side(phi),
                "dist_to_0": format(phi, "e"),
                "dist_to_1": format(1 - phi, "e"),
                "dist_to_half": format(abs(phi - Decimal("0.5")), "f"),
                "bits": bits(Q),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 2. Amplify fidelity
# ---------------------------------------------------------------------------

def amplify_fidelity(
    n_trials: int = 400,
    m_max: int = 200,
    n_max: int = 400,
    rng: random.Random | None = None,
) -> Dict[str, Any]:
    """Assert process 'amplify' = floor term in tax_rows identity."""
    rng = rng or random.Random(0)
    mismatches = 0
    samples = []
    for _ in range(n_trials):
        m = rng.randint(1, m_max)
        n = rng.randint(1, n_max)
        phi = frac_Q_alpha(n)
        fl = floor_dec(Decimal(m) * phi)
        tax = tax_rows_harmonic(m, n)
        expected = m - 1 - fl
        ok = tax == expected
        if not ok:
            mismatches += 1
        if len(samples) < 8:
            samples.append(
                {
                    "m": m,
                    "n": n,
                    "floor_m_phi": fl,
                    "tax_rows": tax,
                    "m_minus_1_minus_floor": expected,
                    "ok": ok,
                }
            )
    # Also dense small grid for certainty
    grid_fail = 0
    for m in range(1, 40):
        for n in range(1, 40):
            if tax_rows_harmonic(m, n) != m - 1 - floor_dec(
                Decimal(m) * frac_Q_alpha(n)
            ):
                grid_fail += 1
            # Cross-check additive chart when feasible
            if tax_rows_harmonic(m, n) != m * bits(n) - bits(m * n):
                grid_fail += 1
    return {
        "n_trials": n_trials,
        "mismatches": mismatches,
        "grid_1_39_failures": grid_fail,
        "pass": mismatches == 0 and grid_fail == 0,
        "reading": (
            "PASS = 'amplify' is faithful renaming of ⌊m·{nα}⌋ in tax_rows; "
            "FAIL = framing drifted from math."
        ),
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# 3. Harmonize scan
# ---------------------------------------------------------------------------

def harmonize_scan(
    qs: Sequence[int] | None = None,
) -> Dict[str, Any]:
    """Correlate |φi+φj−1| with best-axis tax at shape (Qi, Qj)."""
    qs = list(qs) if qs is not None else list(SEED_QS)
    qs = sorted(set(qs))
    pairs = []
    dists: List[float] = []
    best_taxes: List[float] = []
    for i, a in enumerate(qs):
        for b in qs[i:]:
            pa, pb = frac_Q_alpha(a), frac_Q_alpha(b)
            d1 = float(abs(pa + pb - 1))
            tr = tax_rows_harmonic(a, b)
            tc = tax_cols_harmonic(a, b)
            best = min(tr, tc)
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "dist_sum_to_1": format(abs(pa + pb - 1), "f"),
                    "tax_rows": tr,
                    "tax_cols": tc,
                    "best_tax": best,
                }
            )
            dists.append(d1)
            best_taxes.append(float(best))
    # Near-complement vs mid: mean best_tax
    near = [t for d, t in zip(dists, best_taxes) if d < 0.05]
    mid = [t for d, t in zip(dists, best_taxes) if 0.2 <= d <= 0.5]
    r = _pearson(dists, best_taxes)
    pairs.sort(key=lambda p: float(p["dist_sum_to_1"]))
    return {
        "n_pairs": len(pairs),
        "pearson_dist1_vs_best_tax": r,
        "mean_best_tax_near_complement_d1_lt_0.05": (
            sum(near) / len(near) if near else None
        ),
        "mean_best_tax_mid_d1_0.2_0.5": (
            sum(mid) / len(mid) if mid else None
        ),
        "n_near": len(near),
        "n_mid": len(mid),
        "top_near_pairs": pairs[:12],
        "hypothesis": (
            "Near φ+ψ≈1 should show systematically lower best-axis tax than "
            "mid-phase pairs at catalog sizes — partial pass expected from Law B."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Half-flip audit
# ---------------------------------------------------------------------------

def half_flip_audit(
    qs: Sequence[int] | None = None,
    m_probe: int = 32,
) -> Dict[str, Any]:
    """Tax vs |φ−½| (Hypothesis A: near-½ bad for surplus alone)
    and vs |1−φ| (surplus goodness); flag self-complement under squaring.
    """
    qs = list(qs) if qs is not None else list(
        range(1, 120)
    ) + [306, 665, 640, 2560, 6912, 4096]
    qs = sorted(set(qs))
    rows = []
    for Q in qs:
        phi = frac_Q_alpha(Q)
        tax_m = tax_rows_harmonic(m_probe, Q)
        # Self-complement: 2φ ≈ 1 → square shape (Q,Q) near complement
        d_self = abs(phi + phi - 1)
        rows.append(
            {
                "Q": Q,
                "phi": format(phi, "f"),
                "dist_half": float(abs(phi - Decimal("0.5"))),
                "dist_to_1": float(1 - phi),
                "tax_rows_m": tax_m,
                "self_complement_d1": float(d_self),
                "self_complement_band": d_self < 0.05,
            }
        )

    # Correlate tax_rows(m_probe, Q) with dist_to_half and dist_to_1
    taxes = [r["tax_rows_m"] for r in rows]
    d_half = [r["dist_half"] for r in rows]
    d1 = [r["dist_to_1"] for r in rows]
    self_band = [r for r in rows if r["self_complement_band"]]

    # Hypothesis A: near-½ → higher tax for fixed m (worse surplus packing)
    # Expect negative pearson(dist_half, tax): closer to ½ → higher tax
    # i.e. pearson(dist_half, tax) < 0
    return {
        "m_probe": m_probe,
        "n_Q": len(rows),
        "pearson_dist_half_vs_tax": _pearson(d_half, [float(t) for t in taxes]),
        "pearson_dist_to_1_vs_tax": _pearson(d1, [float(t) for t in taxes]),
        "reading_A": (
            "Hypothesis A (near-½ worst for surplus alone): expect "
            "pearson(dist_half, tax) < 0 (closer to ½ → higher tax)."
        ),
        "reading_B": (
            "Hypothesis B (near-½ enables self-complement under squaring): "
            "flag Q with |2φ−1|<0.05; BitNet 2560 is the measured case."
        ),
        "self_complement_band": [
            {"Q": r["Q"], "phi": r["phi"], "d1": r["self_complement_d1"]}
            for r in self_band
        ],
        "bitnet_2560": next((r for r in rows if r["Q"] == 2560), None),
        "sample_extremes": {
            "lowest_tax": sorted(rows, key=lambda r: r["tax_rows_m"])[:5],
            "highest_tax": sorted(rows, key=lambda r: -r["tax_rows_m"])[:5],
        },
    }


# ---------------------------------------------------------------------------
# 5. Breathing cloud
# ---------------------------------------------------------------------------

def breathing_score(qs: Sequence[int]) -> float:
    """Collatz-style BS = Σ({Qi α} − ½). Cloud summary only — not a certificate."""
    return float(sum(frac_Q_alpha(q) - Decimal("0.5") for q in qs))


def breathing_cloud_probe(
    ckpt: Optional[Path] = None,
    skip_ckpt: bool = False,
) -> Dict[str, Any]:
    """BS over mode sets; Pearson vs tax and pad improvement. Expect weak mean corr."""
    catalog_sets = [
        [5, 41, 306],
        [19, 53, 665],
        [306, 53],
        [41, 19],
        [640, 2560],
        [2560, 2560],
        [2560, 6912],
        [640, 2560, 6912],
        [5, 41, 19],
        [306, 53, 41],
    ]
    rows = []
    for dims in catalog_sets:
        bs = breathing_score(dims)
        if len(dims) == 2:
            m, n = dims
            best_tax = min(tax_rows_harmonic(m, n), tax_cols_harmonic(m, n))
            pad = multi_mode_pad(dims, max_pad=48, target="harmonize")
            pad_delta = float(Decimal(pad["score_before"]) - Decimal(pad["score_after"]))
        elif len(dims) == 3:
            from harmonic_multi import best_3d_tax

            best_tax = best_3d_tax(*dims)["best_tax"]
            pad = multi_mode_pad(dims, max_pad=24, target="harmonize")
            pad_delta = float(Decimal(pad["score_before"]) - Decimal(pad["score_after"]))
        else:
            best_tax = None
            pad_delta = None
        rows.append(
            {
                "dims": dims,
                "breathing_score": bs,
                "abs_BS": abs(bs),
                "best_tax": best_tax,
                "pad_score_delta": pad_delta,
            }
        )

    with_tax = [r for r in rows if r["best_tax"] is not None]
    r_tax = _pearson(
        [r["breathing_score"] for r in with_tax],
        [float(r["best_tax"]) for r in with_tax],
    )
    r_abs = _pearson(
        [r["abs_BS"] for r in with_tax],
        [float(r["best_tax"]) for r in with_tax],
    )
    r_pad = _pearson(
        [r["breathing_score"] for r in with_tax if r["pad_score_delta"] is not None],
        [r["pad_score_delta"] for r in with_tax if r["pad_score_delta"] is not None],
    )

    bitnet = None
    path = ckpt or DEFAULT_CKPT
    if not skip_ckpt and path.is_file():
        cloud = bitnet_phase_cloud(path)
        # Per-tensor: BS vs best-axis tax (not vs d1 — for 2 modes
        # BS = φm+φn−1 and d1=|BS| is tautological).
        bs_list = []
        tax_list = []
        pad_deltas = []
        for t in cloud["tensors"]:
            m, n = t["shape"]
            bs_list.append(breathing_score([m, n]))
            tax_list.append(
                float(min(tax_rows_harmonic(m, n), tax_cols_harmonic(m, n)))
            )
        # Unique mode triples only once each for pad cost
        uniq = cloud["unique_mode_lengths"]
        if len(uniq) >= 2:
            for a, b in ((uniq[0], uniq[1]), (uniq[1], uniq[-1])):
                pad = multi_mode_pad([a, b], max_pad=32, target="harmonize")
                pad_deltas.append(
                    {
                        "dims": [a, b],
                        "BS": breathing_score([a, b]),
                        "pad_delta": float(
                            Decimal(pad["score_before"]) - Decimal(pad["score_after"])
                        ),
                    }
                )
        bitnet = {
            "n_tensors": cloud["n_tensors"],
            "unique_modes": cloud["unique_mode_lengths"],
            "pearson_BS_vs_best_tax": _pearson(bs_list, tax_list),
            "pearson_absBS_vs_best_tax": _pearson(
                [abs(b) for b in bs_list], tax_list
            ),
            "mean_BS": sum(bs_list) / len(bs_list) if bs_list else None,
            "note": (
                "For 2-mode tensors BS=φm+φn−1; correlating BS with "
                "|φm+φn−1| would be tautological — use best-axis tax instead."
            ),
            "near_complement_lt_0.05": cloud["tensors_dist_sum_to_1_lt_0.05"],
            "unique_pair_pad_deltas": pad_deltas,
        }

    return {
        "prior": (
            "Collatz: expect weak mean correlation of BS with dynamics; "
            "look at tails / failure modes. Packing: measure vs tax and pad."
        ),
        "catalog_rows": rows,
        "pearson_BS_vs_best_tax": r_tax,
        "pearson_absBS_vs_best_tax": r_abs,
        "pearson_BS_vs_pad_delta": r_pad,
        "bitnet": bitnet,
        "verdict_hint": (
            "null/weak |r| → drop BS as packing predictor; keep as optional cloud label"
        ),
    }


# ---------------------------------------------------------------------------
# 6. Return-map toy
# ---------------------------------------------------------------------------

def return_map_toy(
    seed_Q: int = 53,
    modes: Sequence[int] = (8, 13, 41),
    iterations: int = 2,
) -> Dict[str, Any]:
    """Feed residual phase after amplify as new seed into next mode.

    Residual = {m · φ}; interpret as a new fractional seed (not a new Q).
    Tax trajectory on successive modes. Null reduction is informative.
    """
    phi = frac_Q_alpha(seed_Q)
    traj = []
    cur_phi = phi
    total_tax = 0
    for it in range(iterations):
        for m in modes:
            prod = Decimal(m) * cur_phi
            fl = floor_dec(prod)
            residual = frac(prod)
            # dissonance analogue: m−1−floor when treating cur_phi as {nα}
            tax = m - 1 - fl
            total_tax += tax
            traj.append(
                {
                    "iter": it,
                    "mode_m": m,
                    "phi_in": format(cur_phi, "f"),
                    "floor": fl,
                    "residual_phi": format(residual, "f"),
                    "tax_analogue": tax,
                }
            )
            cur_phi = residual
    # Compare to one-shot amplify of original seed against product of modes
    # (baseline: no return wrap)
    M = 1
    for m in modes:
        M *= m
    baseline = tax_rows_harmonic(M, seed_Q) if iterations >= 1 else None
    return {
        "seed_Q": seed_Q,
        "seed_phi": format(phi, "f"),
        "modes": list(modes),
        "iterations": iterations,
        "trajectory": traj,
        "sum_tax_analogue": total_tax,
        "baseline_tax_rows_product_mode": baseline,
        "reading": (
            "Does iterating residual→seed reduce cumulative tax analogue? "
            "Null/increase = Collatz-style negative control — do not elevate metaphor."
        ),
    }


# ---------------------------------------------------------------------------
# 7. No-compensation check
# ---------------------------------------------------------------------------

def no_compensation_check(
    mid_mode: int = 19,
    companions: Sequence[int] = (5, 41, 53, 306, 640, 2560),
) -> Dict[str, Any]:
    """Fix one mid-phase mode; vary the other. Does companion surplus cancel tax?

    Collatz: coarse compensation fails for streaks. Packing tax is algebraic —
    measure whether best_tax(mid, companion) drops when companion is surplus.
    """
    phi_mid = frac_Q_alpha(mid_mode)
    rows = []
    for c in companions:
        tr = tax_rows_harmonic(mid_mode, c)
        tc = tax_cols_harmonic(mid_mode, c)
        rows.append(
            {
                "mid": mid_mode,
                "companion": c,
                "phi_companion": format(frac_Q_alpha(c), "f"),
                "companion_side": classify_seed_side(frac_Q_alpha(c)),
                "tax_rows": tr,
                "tax_cols": tc,
                "best_tax": min(tr, tc),
                "sum_phis_dist_1": format(
                    abs(phi_mid + frac_Q_alpha(c) - 1), "f"
                ),
            }
        )
    surplus_best = [
        r["best_tax"] for r in rows if r["companion_side"] == "toward_1"
    ]
    mid_best = [r["best_tax"] for r in rows if r["companion_side"] == "mid"]
    deficit_best = [
        r["best_tax"] for r in rows if r["companion_side"] == "toward_0"
    ]
    half_best = [
        r["best_tax"] for r in rows if r["companion_side"] == "near_half"
    ]
    return {
        "mid_mode": mid_mode,
        "phi_mid": format(phi_mid, "f"),
        "pairs": rows,
        "mean_best_tax_with_surplus_companion": (
            sum(surplus_best) / len(surplus_best) if surplus_best else None
        ),
        "mean_best_tax_with_mid_companion": (
            sum(mid_best) / len(mid_best) if mid_best else None
        ),
        "mean_best_tax_with_deficit_companion": (
            sum(deficit_best) / len(deficit_best) if deficit_best else None
        ),
        "mean_best_tax_with_half_companion": (
            sum(half_best) / len(half_best) if half_best else None
        ),
        "hypothesis": (
            "If surplus companions systematically lower best_tax vs mid/deficit, "
            "packing allows algebraic 'compensation' unlike Collatz dynamics. "
            "If not, import the negative control."
        ),
    }


# ---------------------------------------------------------------------------
# CLI / selftest
# ---------------------------------------------------------------------------

def selftest() -> int:
    # Amplify fidelity must pass
    af = amplify_fidelity(n_trials=80, m_max=60, n_max=80, rng=random.Random(1))
    assert af["pass"], af

    # Seed sides
    cat = {r["Q"]: r["side"] for r in seed_catalog()}
    assert cat[5] == "toward_1"
    assert cat[53] == "toward_0"
    assert cat[2560] == "near_half"
    assert cat[19] == "mid"

    # Harmonize: 53+306 near complement
    hs = harmonize_scan([53, 306, 19, 41])
    near_pair = next(
        p for p in hs["top_near_pairs"] if {p["a"], p["b"]} == {53, 306}
    )
    assert float(near_pair["dist_sum_to_1"]) < 0.02

    # Half-flip: 2560 in self-complement band
    hf = half_flip_audit([2560, 5, 53])
    assert any(r["Q"] == 2560 for r in hf["self_complement_band"])

    # Breathing score deterministic
    assert isinstance(breathing_score([5, 41]), float)

    # Return map runs
    rm = return_map_toy(seed_Q=53, modes=(5, 8), iterations=1)
    assert len(rm["trajectory"]) == 2

    # No-compensation runs
    nc = no_compensation_check()
    assert len(nc["pairs"]) >= 3

    # Pad alias wiring (imported multi_mode_pad)
    p1 = multi_mode_pad([2560, 6912], max_pad=8, target="harmonize")
    p2 = multi_mode_pad([2560, 6912], max_pad=8, target="pairwise_complement")
    assert p1["padded"] == p2["padded"]
    p3 = multi_mode_pad([41, 19], max_pad=8, target="amplify_surplus")
    p4 = multi_mode_pad([41, 19], max_pad=8, target="max_surplus")
    assert p3["padded"] == p4["padded"]

    print("PROCESS_LANGUAGE_PROBES PASS")
    return 0


def run(skip_ckpt: bool = False, ckpt: Optional[Path] = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "stance": (
            "Exploratory Phase 4 probes. Process language is a hypothesis. "
            "Null results are successes when they match Collatz negative controls."
        ),
        "seed_catalog": seed_catalog(),
        "amplify_fidelity": amplify_fidelity(),
        "harmonize_scan": harmonize_scan(),
        "half_flip_audit": half_flip_audit(),
        "breathing_cloud": breathing_cloud_probe(ckpt=ckpt, skip_ckpt=skip_ckpt),
        "return_map_toy": {
            "53_via_8_13_41": return_map_toy(53, (8, 13, 41), iterations=2),
            "306_via_5_8": return_map_toy(306, (5, 8), iterations=2),
            "2560_via_2_2": return_map_toy(2560, (2, 2), iterations=2),
        },
        "no_compensation": no_compensation_check(),
    }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "cmd",
        nargs="?",
        default="run",
        choices=["run", "selftest"],
    )
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--skip-ckpt", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "process_language_results.json",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "selftest":
        return selftest()

    selftest()
    report = run(skip_ckpt=args.skip_ckpt, ckpt=args.ckpt)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PROCESS_LANGUAGE wrote {args.out}")

    af = report["amplify_fidelity"]
    print(f"amplify fidelity: {'PASS' if af['pass'] else 'FAIL'} "
          f"(mismatches={af['mismatches']})")

    hs = report["harmonize_scan"]
    print(
        f"harmonize: pearson(d1, best_tax)={hs['pearson_dist1_vs_best_tax']!s}  "
        f"mean_tax near={hs['mean_best_tax_near_complement_d1_lt_0.05']}  "
        f"mid={hs['mean_best_tax_mid_d1_0.2_0.5']}"
    )

    hf = report["half_flip_audit"]
    print(
        f"half-flip: pearson(|φ−½|,tax)={hf['pearson_dist_half_vs_tax']!s}  "
        f"pearson(|1−φ|,tax)={hf['pearson_dist_to_1_vs_tax']!s}  "
        f"self-complement band={[r['Q'] for r in hf['self_complement_band']]}"
    )

    bc = report["breathing_cloud"]
    print(
        f"breathing: pearson(BS,tax)={bc['pearson_BS_vs_best_tax']!s}  "
        f"pearson(|BS|,tax)={bc['pearson_absBS_vs_best_tax']!s}  "
        f"pearson(BS,padΔ)={bc['pearson_BS_vs_pad_delta']!s}"
    )
    if bc.get("bitnet"):
        bn = bc["bitnet"]
        print(
            f"  BitNet BS vs best_tax: r={bn['pearson_BS_vs_best_tax']!s}  "
            f"|BS| vs best_tax: r={bn['pearson_absBS_vs_best_tax']!s}  "
            f"mean_BS={bn['mean_BS']}"
        )
    else:
        print("  BitNet cloud: skipped")

    for name, rm in report["return_map_toy"].items():
        print(
            f"return-map {name}: sum_tax_analogue={rm['sum_tax_analogue']}  "
            f"baseline_product={rm['baseline_tax_rows_product_mode']}"
        )

    nc = report["no_compensation"]
    print(
        f"no-compensation (mid={nc['mid_mode']}): "
        f"mean best_tax surplus_comp={nc['mean_best_tax_with_surplus_companion']}  "
        f"mid={nc['mean_best_tax_with_mid_companion']}  "
        f"deficit={nc['mean_best_tax_with_deficit_companion']}  "
        f"half={nc['mean_best_tax_with_half_companion']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
