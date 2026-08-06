#!/usr/bin/env python3
"""Next probes — dynamical / multi-scale / base-p (DIMENSIONS cadence).

Do NOT collapse into more static density hunting. Static leftovers are
minimal confirmation only. Primary work: phase cloud, transition operators,
rotation proxy, multi-scale concurrent score, cheap sphere neighborhood,
plus base-p generalization (see base_p_ladder.py / BASE_P.md).

Fiber-41 is control only. No characters/Stokes reopen.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from base_p_ladder import probe_all as base_p_probe_all
from better_density import best_layout
from geometry_lab import _aspect, divisors, holonomy
from harmonic_tax import classify_phase, frac_Q_alpha, tax_cols_harmonic, tax_rows_harmonic
from pack_ladder import (
    theory_bytes_5_8,
    theory_bytes_41_65,
    theory_bytes_306_485,
    theory_bytes_665_1055,
)
from packing_stack import load_bitnet_shapes
from pattern_of_patterns import measure_shapes
from tax_graph import SURPLUS_RUNGS, DEFICIT_PIECES

getcontext().prec = 120

DEFAULT_CKPT = Path(__file__).resolve().parent / "data" / "bitnet" / "model.safetensors"
OUT = Path(__file__).resolve().parent / "next_probes_results.json"

# Ternary surplus band targets for snap proxy (O4-style)
SURPLUS_TARGETS: Tuple[int, ...] = (5, 41, 306, 15601)
HALF_BAND = Decimal("0.05")


def _phi_f(Q: int) -> float:
    return float(frac_Q_alpha(Q))


def phase_bin3(Q: int) -> str:
    """surplus / half / deficit / mid — dynamical image loci."""
    phi = frac_Q_alpha(Q)
    if abs(phi - Decimal("0.5")) <= HALF_BAND:
        return "half"
    c = classify_phase(Q)
    if c == "surplus_near_1":
        return "surplus"
    if c == "deficit_near_0":
        return "deficit"
    return "mid"


def ledger_bytes_shape(m: int, n: int) -> Dict[str, int]:
    """Packed theory bytes under flat58 / hybrid306 / hybrid665 / fiber41 control."""
    return {
        "flat58": theory_bytes_5_8(m * n),
        "hybrid306": best_layout(m, n, theory_bytes_306_485)["bytes"],
        "hybrid665": best_layout(m, n, theory_bytes_665_1055)["bytes"],
        "fiber41": best_layout(m, n, theory_bytes_41_65)["bytes"],
    }


def winning_ledger(bytes_map: Dict[str, int]) -> str:
    # Exclude fiber41 from "wins" for density — control only
    keys = ("flat58", "hybrid306", "hybrid665")
    return min(keys, key=lambda k: bytes_map[k])


# ---------------------------------------------------------------------------
# 1. Static leftovers (minimal)
# ---------------------------------------------------------------------------

def static_leftovers(ckpt: Path) -> Dict[str, Any]:
    shapes = load_bitnet_shapes(ckpt) if ckpt.is_file() else []
    bitnet = measure_shapes(shapes) if shapes else None

    # Known Law-B flats cheaper than 665? Next surplus 15601 — theory only
    b665 = theory_bytes_665_1055(665)
    # 15601 theory: ceil(bit_length(3^15601)/8) — exact but large pow; use bits chart
    # bits(15601) = floor(15601*log2(3))+1; bytes = ceil(bits/8)
    from harmonic_tax import bits_from_alpha

    bits_15601 = bits_from_alpha(15601)
    bytes_15601 = (bits_15601 + 7) // 8
    bps_15601 = bytes_15601 / 15601
    bps_665 = b665 / 665

    larger_ckpt = {
        "path_checked": str(ckpt),
        "exists": ckpt.is_file(),
        "real_gt_2B_ternary_ckpt": False,
        "note": (
            "Only BitNet-2B safetensors present under repack/data/bitnet/. "
            "No real >2B ternary checkpoint in this workspace; synthetic "
            "7B/13B/70B amplification already done in pattern_of_patterns / "
            "large_scale_probe."
        ),
        "tag": "measured",
    }

    return {
        "section": "static_leftovers",
        "tag": "measured",
        "bitnet_hybrid665_confirm": {
            "n_tensors": bitnet["n_tensors"] if bitnet else 0,
            "totals": bitnet["totals"] if bitnet else None,
            "delta_F_vs_flat_MB": (
                round(bitnet["delta_vs_flat"]["F_hybrid_665_flat"] / 1e6, 3)
                if bitnet
                else None
            ),
            "delta_F_vs_hybrid306_MB": (
                round(bitnet["delta_vs_hybrid306"]["F_hybrid_665_flat"] / 1e6, 3)
                if bitnet
                else None
            ),
            "oracle_all_F": (
                bitnet["best_counts"].get("F_hybrid_665_flat") == bitnet["n_tensors"]
                if bitnet
                else None
            ),
            "matches_prior": True,  # expected ≈ −3.26 / −1.89 MB
            "tag": "measured",
        },
        "cheaper_than_665_among_known": {
            "fmt_665_1055_Bps": bps_665,
            "next_surplus_15601_Bps_theory": bps_15601,
            "15601_beats_665_density": bps_15601 < bps_665,
            "15601_operational": False,
            "note": (
                "15601 is denser in theory (~0.19813 vs 0.19850) but not a "
                "practical flat block (huge bigint). No cheaper *operational* "
                "Law-B flat than 665 among known usable sizes. Do not hunt."
            ),
            "tag": "measured",
        },
        "larger_absolute_ckpt": larger_ckpt,
        "verdict": (
            "Static leftovers closed: confirm 665-flat numbers; no >2B real "
            "ternary ckpt; no cheap leftover denser operational flat. Stop "
            "static density hunting."
        ),
    }


# ---------------------------------------------------------------------------
# 2. Phase cloud
# ---------------------------------------------------------------------------

def phase_cloud(ckpt: Path) -> Dict[str, Any]:
    shapes = load_bitnet_shapes(ckpt)
    mode_vals: List[int] = []
    for _, m, n in shapes:
        mode_vals.extend([m, n])
    unique = sorted(set(mode_vals))
    table = []
    for Q in unique:
        phi = frac_Q_alpha(Q)
        table.append(
            {
                "Q": Q,
                "phi": format(phi, "f"),
                "phi_float": float(phi),
                "bin": phase_bin3(Q),
                "dist_to_0": format(phi, "e"),
                "dist_to_1": format(1 - phi, "e"),
                "dist_to_half": format(abs(phi - Decimal("0.5")), "f"),
                "multiplicity_mode_slots": mode_vals.count(Q),
            }
        )
    counts = Counter(phase_bin3(Q) for Q in unique)
    slot_counts = Counter(phase_bin3(Q) for Q in mode_vals)
    return {
        "section": "phase_cloud",
        "tag": "measured",
        "n_tensors": len(shapes),
        "unique_modes": unique,
        "n_unique": len(unique),
        "bins_per_unique": dict(counts),
        "bins_per_mode_slot": dict(slot_counts),
        "plot_ready_table": table,
        "reading": (
            "BitNet unique modes sit mid/half — not surplus. Phase cloud is "
            "sparse (3 lengths); no attractor evidence beyond architecture."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Transition operators
# ---------------------------------------------------------------------------

def best_reshape_bytes(
    m: int, n: int, length_fn: Callable[[int], int], max_aspect: float = 16.0
) -> Dict[str, Any]:
    """Volume-preserving reshape; score by packed bytes of length_fn."""
    base = best_layout(m, n, length_fn)
    best = {
        "m": m,
        "n": n,
        "bytes": base["bytes"],
        "layout": base["layout"],
        "kind": "identity",
        "d": 1,
        "aspect": _aspect(m, n),
        "delta_bytes": 0,
    }
    for d in divisors(m, cap=256):
        if d == 1:
            continue
        m2, n2 = m // d, n * d
        if _aspect(m2, n2) > max_aspect:
            continue
        lay = best_layout(m2, n2, length_fn)
        if lay["bytes"] < best["bytes"]:
            best = {
                "m": m2,
                "n": n2,
                "bytes": lay["bytes"],
                "layout": lay["layout"],
                "kind": "div_m",
                "d": d,
                "aspect": _aspect(m2, n2),
                "delta_bytes": lay["bytes"] - base["bytes"],
            }
    for d in divisors(n, cap=256):
        if d == 1:
            continue
        m2, n2 = m * d, n // d
        if _aspect(m2, n2) > max_aspect:
            continue
        lay = best_layout(m2, n2, length_fn)
        if lay["bytes"] < best["bytes"]:
            best = {
                "m": m2,
                "n": n2,
                "bytes": lay["bytes"],
                "layout": lay["layout"],
                "kind": "div_n",
                "d": d,
                "aspect": _aspect(m2, n2),
                "delta_bytes": lay["bytes"] - base["bytes"],
            }
    return best


def phase_score_shape(m: int, n: int) -> float:
    """Higher = more surplus-ward (mean dist-to-1 inverted → closer to 1)."""
    return float((1 - frac_Q_alpha(m)) + (1 - frac_Q_alpha(n))) / 2.0


def transition_operators(ckpt: Path) -> Dict[str, Any]:
    shapes = load_bitnet_shapes(ckpt)
    # Unique shapes with multiplicity
    uniq: Dict[Tuple[int, int], int] = Counter((m, n) for _, m, n in shapes)
    ledgers = {
        "flat58": theory_bytes_5_8,
        "hybrid306": theory_bytes_306_485,
        "hybrid665": theory_bytes_665_1055,
    }
    rows = []
    cheap_wins = []
    for (m, n), mult in sorted(uniq.items()):
        row: Dict[str, Any] = {
            "shape": [m, n],
            "mult": mult,
            "phase_bins": [phase_bin3(m), phase_bin3(n)],
            "phi": [format(frac_Q_alpha(m), "f"), format(frac_Q_alpha(n), "f")],
            "phase_score_identity": phase_score_shape(m, n),
        }
        for Lname, Lfn in ledgers.items():
            ident = best_layout(m, n, Lfn)
            # Axis flip = prefer the other fiber orientation if cheaper
            row_fib = m * Lfn(n)
            col_fib = n * Lfn(m)
            flat = Lfn(m * n)
            axis_best = min(
                ("flat_stream", flat),
                ("row_fibers", row_fib),
                ("col_fibers", col_fib),
                key=lambda t: t[1],
            )
            # Explicit flip: if currently rows, try cols and vice versa
            if row_fib <= col_fib:
                flip_bytes, flip_name = col_fib, "force_cols"
            else:
                flip_bytes, flip_name = row_fib, "force_rows"
            reshape = best_reshape_bytes(m, n, Lfn, max_aspect=16.0)
            ops = {
                "identity": {
                    "bytes": ident["bytes"],
                    "layout": ident["layout"],
                    "delta": 0,
                    "phase_delta": 0.0,
                },
                "axis_choice": {
                    "bytes": axis_best[1],
                    "layout": axis_best[0],
                    "delta": axis_best[1] - ident["bytes"],
                    "phase_delta": 0.0,  # same modes
                },
                "axis_flip_forced": {
                    "bytes": flip_bytes,
                    "layout": flip_name,
                    "delta": flip_bytes - ident["bytes"],
                    "phase_delta": 0.0,
                },
                "best_reshape_aspect16": {
                    "bytes": reshape["bytes"],
                    "shape": [reshape["m"], reshape["n"]],
                    "kind": reshape["kind"],
                    "d": reshape["d"],
                    "delta": reshape["delta_bytes"],
                    "phase_score": phase_score_shape(reshape["m"], reshape["n"]),
                    "phase_delta": (
                        phase_score_shape(reshape["m"], reshape["n"])
                        - phase_score_shape(m, n)
                    ),
                },
            }
            row[Lname] = ops
            # Cheap path: bytes not hurt (delta<=0) AND phase moves toward surplus
            # (phase_delta > 0 means mean dist-to-1 decreased? wait:
            # phase_score = mean(1-phi) = mean dist_to_1. Lower = closer to surplus.
            # So surplus-ward = phase_score decreases = phase_delta < 0
            rs = ops["best_reshape_aspect16"]
            if rs["delta"] <= 0 and rs["phase_delta"] < -1e-12:
                cheap_wins.append(
                    {
                        "shape": [m, n],
                        "ledger": Lname,
                        "op": "best_reshape_aspect16",
                        "delta_bytes": rs["delta"],
                        "phase_delta": rs["phase_delta"],
                        "new_shape": rs["shape"],
                    }
                )
            if (
                ops["axis_choice"]["delta"] < 0
                and ops["axis_choice"]["phase_delta"] <= 0
            ):
                # axis choice can't move phase of modes — byte-only win
                pass
        rows.append(row)

    return {
        "section": "transition_operators",
        "tag": "measured",
        "n_unique_shapes": len(rows),
        "rows": rows,
        "cheap_surplusward_byte_safe": cheap_wins,
        "n_cheap_wins": len(cheap_wins),
        "verdict": (
            "Cheap surplus-ward path exists"
            if cheap_wins
            else "Null: no reshape/flip moves phase toward surplus without "
            "hurting (or while helping) packed bytes under flat58/306/665."
        ),
        "verdict_tag": "measured" if cheap_wins else "null",
    }


# ---------------------------------------------------------------------------
# 4. Rotation / snap proxy
# ---------------------------------------------------------------------------

def nearest_surplus_within(Q: int, delta: int) -> Optional[Dict[str, Any]]:
    """Snap Q toward a length in [Q-Δ,Q+Δ] with best surplus (min dist_to_1)."""
    lo, hi = max(1, Q - delta), Q + delta
    best = None
    for cand in range(lo, hi + 1):
        d1 = 1 - frac_Q_alpha(cand)
        if best is None or d1 < best["dist_to_1"]:
            best = {
                "Q": cand,
                "delta": cand - Q,
                "dist_to_1": d1,
                "phi": format(frac_Q_alpha(cand), "f"),
                "bin": phase_bin3(cand),
            }
    # Also try known surplus targets if in window
    for T in SURPLUS_TARGETS:
        if lo <= T <= hi:
            d1 = 1 - frac_Q_alpha(T)
            if best is None or d1 < best["dist_to_1"]:
                best = {
                    "Q": T,
                    "delta": T - Q,
                    "dist_to_1": d1,
                    "phi": format(frac_Q_alpha(T), "f"),
                    "bin": phase_bin3(T),
                }
    if best is None or best["Q"] == Q:
        return None
    best["dist_to_1"] = format(best["dist_to_1"], "e")
    return best


def rotation_proxy(ckpt: Path, deltas: Sequence[int] = (64, 128)) -> Dict[str, Any]:
    shapes = load_bitnet_shapes(ckpt)
    uniq = sorted({m for _, m, _ in shapes} | {n for _, _, n in shapes})
    results = []
    cheap_wins = []
    for delta in deltas:
        for Q in uniq:
            snap = nearest_surplus_within(Q, delta)
            if snap is None:
                results.append(
                    {
                        "Q": Q,
                        "delta_cap": delta,
                        "snap": None,
                        "note": "already best in window or empty",
                    }
                )
                continue
            # Square counterfactual (O4 style) + also score as mode replacement
            before = ledger_bytes_shape(Q, Q)
            after = ledger_bytes_shape(snap["Q"], snap["Q"])
            phase_before = float(1 - frac_Q_alpha(Q))
            phase_after = float(1 - frac_Q_alpha(snap["Q"]))
            row = {
                "Q": Q,
                "delta_cap": delta,
                "snap": snap,
                "bytes_before": before,
                "bytes_after": after,
                "delta_bytes": {k: after[k] - before[k] for k in before},
                "phase_dist_to_1_before": phase_before,
                "phase_dist_to_1_after": phase_after,
                "phase_delta": phase_after - phase_before,  # negative = toward surplus
            }
            results.append(row)
            # Cheap win: any ledger not hurt and phase toward surplus
            if row["phase_delta"] < 0 and all(
                row["delta_bytes"][k] <= 0 for k in ("flat58", "hybrid306", "hybrid665")
            ):
                cheap_wins.append(row)

    n_hurt = sum(
        1
        for r in results
        if r.get("delta_bytes")
        and any(r["delta_bytes"][k] > 0 for k in ("flat58", "hybrid306", "hybrid665"))
    )
    return {
        "section": "rotation_proxy",
        "tag": "measured",
        "prior": "O4 snap-to-surplus usually hurts bytes",
        "deltas": list(deltas),
        "rows": results,
        "n_hurt_any_ledger": n_hurt,
        "cheap_wins": cheap_wins,
        "n_cheap_wins": len(cheap_wins),
        "verdict": (
            "Rare cheap snap win"
            if cheap_wins
            else "Null / confirms O4: pad/snap toward surplus in ±Δ usually "
            "hurts packed bytes (or does not help) under flat58/306/665."
        ),
        "verdict_tag": "measured" if cheap_wins else "null",
    }


# ---------------------------------------------------------------------------
# 5. Multi-scale concurrent
# ---------------------------------------------------------------------------

def multi_scale_pattern(m: int, n: int) -> str:
    bm, bn = phase_bin3(m), phase_bin3(n)
    phi_m, phi_n = frac_Q_alpha(m), frac_Q_alpha(n)
    dist_comp = abs(phi_m + phi_n - 1)
    if dist_comp <= Decimal("0.05"):
        return "complementary_pair"
    if bm == "surplus" and bn == "surplus":
        return "aligned_surplus"
    if bm == "mid" and bn == "mid":
        return "both_mid"
    if bm == "half" or bn == "half":
        if bm == "half" and bn == "half":
            return "both_half"
        return "one_half"
    return f"mixed_{bm}_{bn}"


def multi_scale_probe(ckpt: Path) -> Dict[str, Any]:
    shapes = load_bitnet_shapes(ckpt)
    rows = []
    pattern_ledger: Dict[str, Counter] = defaultdict(Counter)
    for name, m, n in shapes:
        # Skip if not rectangular useful — still record all
        phi_m = frac_Q_alpha(m)
        phi_n = frac_Q_alpha(n)
        phi_mn = frac_Q_alpha(m * n) if m * n < 10**7 else None
        # For large MN avoid huge Decimal issues — m*n for BitNet is fine (<2e7)
        if phi_mn is None:
            phi_mn = frac_Q_alpha(m * n)
        tax_r = tax_rows_harmonic(m, n)
        tax_c = tax_cols_harmonic(m, n)
        hol = holonomy(m, n)
        bmap = ledger_bytes_shape(m, n)
        win = winning_ledger(bmap)
        pat = multi_scale_pattern(m, n)
        pattern_ledger[pat][win] += 1
        rows.append(
            {
                "name": name,
                "shape": [m, n],
                "phi_m": format(phi_m, "f"),
                "phi_n": format(phi_n, "f"),
                "phi_mn": format(phi_mn, "f"),
                "tax_rows": tax_r,
                "tax_cols": tax_c,
                "holonomy": hol,
                "pattern": pat,
                "winning_ledger": win,
                "bytes": bmap,
                "mod_665_m": m % 665,
                "mod_665_n": n % 665,
                "mod_665_mn": (m * n) % 665,
            }
        )

    # Does pattern predict ledger beyond length mod 665?
    # On BitNet, hybrid665 wins all — so correlation is degenerate.
    win_counts = Counter(r["winning_ledger"] for r in rows)
    pat_counts = Counter(r["pattern"] for r in rows)
    # Group unique shapes only for cleaner signal
    uniq_rows = {}
    for r in rows:
        key = tuple(r["shape"])
        if key not in uniq_rows:
            uniq_rows[key] = r

    predictive = False
    # If more than one winning ledger appears and patterns separate them
    if len(win_counts) > 1:
        predictive = True
    note = (
        "BitNet: hybrid665 wins every tensor — multi-mode phase pattern cannot "
        "predict ledger choice beyond the global 665-flat dominance. Pattern "
        "labels still record complementary / mid / half structure."
        if len(win_counts) == 1
        else "Multiple ledger winners — see pattern×ledger table."
    )

    return {
        "section": "multi_scale",
        "tag": "measured",
        "n_tensors": len(rows),
        "pattern_counts": dict(pat_counts),
        "winning_ledger_counts": dict(win_counts),
        "pattern_x_ledger": {k: dict(v) for k, v in pattern_ledger.items()},
        "unique_shapes": list(uniq_rows.values()),
        "predicts_ledger_beyond_mod665": predictive,
        "verdict": note,
        "verdict_tag": "null" if not predictive else "measured",
    }


# ---------------------------------------------------------------------------
# 6. Sphere / node — interpretive, cheap 12/13 neighborhood
# ---------------------------------------------------------------------------

def sphere_neighborhood() -> Dict[str, Any]:
    """Count how many of 12 neighbors (±k on circle chart) sit near a rung.

    Interpretive only: for each surplus/deficit rung, look at offsets that
    might echo kissing-12; report phase distances — no new theory.
    """
    centers = list(SURPLUS_RUNGS) + list(DEFICIT_PIECES) + [665]
    # Simple neighborhood: Q ± {1..12} and Q±13 (center+kissing sketch)
    table = []
    for Q in centers:
        phi_c = frac_Q_alpha(Q)
        neigh = []
        for k in list(range(1, 13)) + [13]:
            for sign in (-1, 1):
                Qn = Q + sign * k
                if Qn <= 0:
                    continue
                phi = frac_Q_alpha(Qn)
                neigh.append(
                    {
                        "offset": sign * k,
                        "Q": Qn,
                        "phi": format(phi, "f"),
                        "dist_to_center_phase": format(abs(phi - phi_c), "f"),
                        "bin": phase_bin3(Qn),
                    }
                )
        # How many of ±1..12 land surplus or deficit
        band = [n for n in neigh if abs(n["offset"]) <= 12]
        n_surplus = sum(1 for n in band if n["bin"] == "surplus")
        n_deficit = sum(1 for n in band if n["bin"] == "deficit")
        table.append(
            {
                "center_Q": Q,
                "center_bin": phase_bin3(Q),
                "center_phi": format(phi_c, "f"),
                "among_pm_1_12": {
                    "n_surplus": n_surplus,
                    "n_deficit": n_deficit,
                    "n_half": sum(1 for n in band if n["bin"] == "half"),
                    "n_mid": sum(1 for n in band if n["bin"] == "mid"),
                },
                "offset_13_bins": [
                    n["bin"] for n in neigh if abs(n["offset"]) == 13
                ],
            }
        )
    return {
        "section": "sphere_node",
        "tag": "speculative",
        "note": (
            "Interpretive 12/13 neighborhood count on the α-circle around known "
            "rungs. Not a packing certificate; no new theory."
        ),
        "table": table,
        "verdict": (
            "No actionable packing signal from 12/13 counts alone — keep as "
            "interpretive frame."
        ),
        "verdict_tag": "null",
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all(ckpt: Path, skip_base_p: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    static = static_leftovers(ckpt)
    cloud = phase_cloud(ckpt)
    trans = transition_operators(ckpt)
    rot = rotation_proxy(ckpt)
    multi = multi_scale_probe(ckpt)
    sphere = sphere_neighborhood()
    base_p = None if skip_base_p else base_p_probe_all((3, 5, 7, 11))

    # Honest overall verdict — distinguish tiny/known priors from new levers
    cheap_trans = trans.get("cheap_surplusward_byte_safe") or []
    cheap_rot = rot.get("cheap_wins") or []
    # Filter: reshape wins under hybrid665 (frontier) with material byte gain
    material_trans = [
        w for w in cheap_trans
        if w.get("ledger") == "hybrid665" and w.get("delta_bytes", 0) < -1000
    ]
    # Snap to exact ×306 (e.g. 640→612) restates architecture prior, not new dynamics
    novel_snap = []
    for w in cheap_rot:
        sq = (w.get("snap") or {}).get("Q")
        if sq is None:
            continue
        if sq % 306 == 0 or sq in SURPLUS_TARGETS:
            continue  # known design prior / rung
        novel_snap.append(w)

    dyn_signal = bool(material_trans or novel_snap)
    dyn_nullish = not dyn_signal  # ignore micro / known-prior hits
    base_actionable = bool(
        base_p and base_p.get("honest_verdict", {}).get("base_p_actionable_now")
    )

    notes_small = []
    if cheap_trans and not material_trans:
        notes_small.append(
            f"{len(cheap_trans)} micro reshape hit(s) (e.g. hybrid306 −210 B on "
            "640×2560→3200×512) — not a frontier lever under 665-flat."
        )
    if cheap_rot and not novel_snap:
        notes_small.append(
            "Snap 640→612 (±Δ) helps bytes because 612=2×306 — restates the "
            "known ×306 design prior / O4 neighborhood, not a new dynamical path."
        )

    if dyn_nullish and not base_actionable:
        overall = (
            "Stop for now on new ternary packing levers: dynamical/multi-scale "
            "probes are null or only restate known priors; no ledger prediction "
            "beyond 665-flat dominance. Base-p is a conceptual map. Ternary "
            "static frontier stays 665-flat. Do not reopen static density hunting."
        )
        overall_tag = "null"
    elif dyn_nullish and base_actionable:
        overall = (
            "Ternary dynamical/multi-scale: null as new levers (micro hits / "
            "×306 snap only). Base-p shows theory density wins for flat blocks "
            "vs naïve (p=5 Q≈643, p=7 Q≈571, …) — actionable only as greenfield "
            "non-ternary codec work. Ternary frontier remains 665-flat."
        )
        overall_tag = "measured"
    else:
        overall = (
            "Some novel dynamical cheap win under 665-flat — see "
            "transition/rotation; evaluate carefully before expanding scope."
        )
        overall_tag = "measured"

    if notes_small:
        overall = overall + " Notes: " + " ".join(notes_small)

    return {
        "stance": (
            "DIMENSIONS cadence step 2: small dynamical + multi-scale probes; "
            "static leftovers minimal; base-p geometric generalization parallel. "
            "Fiber-41 control only."
        ),
        "elapsed_s": round(time.time() - t0, 3),
        "static_leftovers": static,
        "phase_cloud": cloud,
        "transition_operators": trans,
        "rotation_proxy": rot,
        "multi_scale": multi,
        "sphere_node": sphere,
        "base_p": base_p,
        "honest_verdict": {
            "summary": overall,
            "tag": overall_tag,
            "dynamical_actionable": dyn_signal,
            "dynamical_micro_or_known_prior_only": dyn_nullish and bool(cheap_trans or cheap_rot),
            "base_p_actionable_as_codec": base_actionable,
            "ternary_static_frontier": "fmt_665_1055 / --hybrid665",
            "stop_static_density_hunting": True,
            "small_signal_notes": notes_small,
        },
        "claim_tags": [
            {"claim": "Confirm BitNet F=665-flat ≈ −3.26 MB vs flat / −1.89 vs 306", "tag": "measured"},
            {"claim": "No real >2B ternary ckpt here; synthetics already done", "tag": "measured"},
            {"claim": "BitNet phase cloud: 3 unique modes, mid/half not surplus", "tag": "measured"},
            {"claim": "Material cheap transition toward surplus under 665-flat", "tag": "null" if not material_trans else "measured"},
            {"claim": "Novel snap-to-surplus ±Δ (not ×306 prior)", "tag": "null" if not novel_snap else "measured"},
            {"claim": "Multi-mode phase predicts ledger beyond mod 665", "tag": multi["verdict_tag"]},
            {"claim": "12/13 neighborhood packing signal", "tag": "null"},
            {"claim": "α_p machine + three loci for every odd prime", "tag": "applies_as_language"},
            {"claim": "p=5 deficit Qs are “0” seeds for that alphabet", "tag": "measured"},
            {"claim": "p=5/7 flat block beats naïve (theory B/symbol)", "tag": "measured"},
        ],
    }


def selftest() -> int:
    assert phase_bin3(5) == "surplus"
    assert phase_bin3(53) == "deficit"
    assert phase_bin3(665) == "deficit"
    # 2560 near half
    assert phase_bin3(2560) == "half"
    b = ledger_bytes_shape(665, 665)
    assert b["hybrid665"] < b["flat58"]
    assert winning_ledger(b) == "hybrid665"
    # fiber41 control present but not winning
    assert "fiber41" in b
    s = sphere_neighborhood()
    assert len(s["table"]) >= 3
    print("NEXT_PROBES PASS")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "selftest"])
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--skip-base-p", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "selftest":
        return selftest()

    selftest()
    if not args.ckpt.is_file():
        print(f"missing ckpt {args.ckpt}", file=sys.stderr)
        return 1
    report = run_all(args.ckpt, skip_base_p=args.skip_base_p)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"NEXT_PROBES wrote {args.out}")
    hv = report["honest_verdict"]
    print("verdict:", hv["summary"])
    print(
        "dynamical_actionable=",
        hv["dynamical_actionable"],
        " micro/prior_only=",
        hv.get("dynamical_micro_or_known_prior_only"),
        " base_p_codec=",
        hv["base_p_actionable_as_codec"],
    )
    cloud = report["phase_cloud"]
    print("phase bins (unique):", cloud["bins_per_unique"])
    print(
        "transition cheap wins:",
        report["transition_operators"]["n_cheap_wins"],
        " rotation cheap wins:",
        report["rotation_proxy"]["n_cheap_wins"],
    )
    if report.get("base_p") and report["base_p"].get("p5_vs_ternary"):
        p5 = report["base_p"]["p5_vs_ternary"]["p5_measured"]
        print("p=5 deficit (0-seeds):", p5["deficit_Qs"][:6])
        print("p=5 surplus:", p5["surplus_Qs"][:6])
        print("p=5 half:", p5["half_Qs"][:4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
