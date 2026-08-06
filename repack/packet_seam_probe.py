#!/usr/bin/env python3
"""Packet / seam probe — test 0→1 prime-context framing against packing.

Five dimensions (each killed or kept by a metric):

  P1 Pattern    (⌊n/p⌋, n mod p) vs CF phase as packing-event predictors
  P2 Recursion  k-cycles of p; CF ladder as scale-transport (not 2p / p²)
  P3 Nesting    Law-C digit nest ≠ UFRF 0→1 preservation unless joint win
  P4 Hierarchy  C0/C1/odd-prime layers ↔ packing ledgers (no flatten 0=1)
  P5 Filter     tag each claim: applies_operationally | applies_as_language
                | does_not_apply | false_identification

Practical metrics:
  - twin-seam vs interior tax / bytes
  - BitNet redesign-to-nearest-306-multiple (not pad)
  - density_signal: A/B/C/D counterfactuals, 61/306 theory, proximity, scale
  - synthetic multiples of 306 / 665
"""

from __future__ import annotations

import json
import math
import struct
import sys
import time
from collections import defaultdict
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harmonic_tax import frac_Q_alpha, max_zero_tax_m
from ledger_packer import pack_tensor
from pack_ladder import (
    theory_bytes_5_8,
    theory_bytes_41_65,
    theory_bytes_306_485,
)
from packing_stack import load_bitnet_shapes
from tax_graph import (
    DEFICIT_PIECES,
    SURPLUS_RUNGS,
    bits,
    split_tax,
)

getcontext().prec = 120

DEFAULT_CKPT = Path(__file__).resolve().parent / "data" / "bitnet" / "model.safetensors"
OUT = Path(__file__).resolve().parent / "packet_seam_results.json"

SURPLUS = set(SURPLUS_RUNGS) | {15601}
DEFICIT = set(DEFICIT_PIECES) | {665}
ALL_RUNGS = SURPLUS | DEFICIT
CF_LADDER = (5, 41, 306, 15601)  # surplus scale-transport
PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
COMPOSITES = (4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 306)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def packet_chart(n: int, p: int) -> Dict[str, Any]:
    """Retain both coordinates: (⌊n/p⌋, n mod p). At n=p → (1, 0)."""
    if p <= 0:
        raise ValueError("p must be > 0")
    cycle, local = divmod(n, p)
    return {
        "n": n,
        "p": p,
        "cycle": cycle,
        "local": local,
        "at_seam": local == 0 and n > 0,
        "typed_rechart": n == p,  # completion 1 in parent + seed 0 in child
        "last_interior": local == p - 1,
        "first_interior": local == 1,
    }


def twin_seam_triple(k: int, p: int) -> Dict[str, Any]:
    """Around contextual zero kp: (k-1,p-1), (k,0), (k,1)."""
    if k < 1 or p < 1:
        raise ValueError("k,p >= 1")
    kp = k * p
    return {
        "k": k,
        "p": p,
        "seam": kp,
        "prev_sheet": packet_chart(kp - 1, p),
        "zero": packet_chart(kp, p),
        "next_sheet": packet_chart(kp + 1, p),
        "prev_n": kp - 1,
        "next_n": kp + 1,
    }


def tag_length(Q: int) -> Dict[str, Any]:
    if Q <= 0:
        return {"Q": Q, "phi": None, "dist1": None, "max_m_zt": 0, "rung": False}
    phi = frac_Q_alpha(Q)
    dist1 = min(phi, 1 - phi) if phi <= Decimal("0.5") else (1 - phi)
    # prefer distance-to-1 for surplus reading
    d1 = 1 - phi
    return {
        "Q": Q,
        "phi": format(phi, "f"),
        "dist_to_1": format(d1, "f"),
        "dist_to_0_or_1": format(min(phi, d1), "f"),
        "max_m_zt": max_zero_tax_m(Q),
        "rung_surplus": Q in SURPLUS,
        "rung_deficit": Q in DEFICIT,
        "rung": Q in ALL_RUNGS,
        "bytes_5_8": theory_bytes_5_8(Q),
        "bytes_41": theory_bytes_41_65(Q),
        "bytes_306": theory_bytes_306_485(Q) if Q >= 306 or Q % 306 == 0 else None,
    }


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3, 5, 7):
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    for d in range(3, r + 1, 2):
        if n % d == 0:
            return False
    return True


# ---------------------------------------------------------------------------
# P1 Pattern — predict packing events
# ---------------------------------------------------------------------------

def packing_events(Q_lo: int = 1, Q_hi: int = 800) -> List[Dict[str, Any]]:
    """Enumerate packing 'events' in a window."""
    events = []
    for Q in range(Q_lo, Q_hi + 1):
        t = tag_length(Q)
        # event types
        kinds = []
        if t["rung"]:
            kinds.append("rung_hit")
        if t["max_m_zt"] >= 8:  # strong zero-tax partner capacity
            kinds.append("strong_zt")
        # ledger switch candidate: exact quantum
        if Q % 306 == 0:
            kinds.append("ledger_306")
        if Q % 665 == 0:
            kinds.append("ledger_665")
        if Q % 41 == 0 and Q >= 41:
            kinds.append("fiber_41_align")
        if not kinds:
            continue
        events.append({"Q": Q, "kinds": kinds, **{k: t[k] for k in ("phi", "dist_to_1", "max_m_zt", "rung")}})
    return events


def pattern_predictivity(events: Sequence[Dict[str, Any]], contexts: Sequence[int]) -> Dict[str, Any]:
    """Does seam (local=0) or twin sheets (±1) concentrate events vs CF surplus band?"""
    if not events:
        return {"n_events": 0}

    Qs = [e["Q"] for e in events]
    Q_hi = max(Qs)
    universe = list(range(1, Q_hi + 1))

    # CF surplus predictor: {Qα} >= 0.9
    surplus_band = []
    for Q in universe:
        phi = frac_Q_alpha(Q)
        if phi >= Decimal("0.9"):
            surplus_band.append(Q)
    surplus_set = set(surplus_band)
    event_set = set(Qs)

    def precision_recall(pred: set) -> Dict[str, float]:
        if not pred:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pred": 0}
        tp = len(pred & event_set)
        prec = tp / len(pred)
        rec = tp / len(event_set) if event_set else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "n_pred": len(pred), "tp": tp}

    # Seam predictor union over contexts: Q ≡ 0 (mod p)
    seam_pred: set = set()
    twin_pred: set = set()
    for p in contexts:
        for Q in universe:
            ch = packet_chart(Q, p)
            if ch["at_seam"]:
                seam_pred.add(Q)
            if ch["last_interior"] or ch["first_interior"]:
                twin_pred.add(Q)

    # Chance: random density matched to surplus_band size
    chance_rate = len(event_set) / len(universe)
    results = {
        "n_events": len(event_set),
        "universe": len(universe),
        "base_rate": chance_rate,
        "cf_surplus_band_ge_0.9": precision_recall(surplus_set),
        "seam_any_context": precision_recall(seam_pred),
        "twin_sheets_any_context": precision_recall(twin_pred),
        "contexts_used": list(contexts),
    }
    # Per-context seam precision on rung_hit only
    rung_Qs = {e["Q"] for e in events if "rung_hit" in e["kinds"]}
    per_p = []
    for p in contexts:
        seams = {Q for Q in universe if Q % p == 0}
        tp = len(seams & rung_Qs)
        per_p.append(
            {
                "p": p,
                "n_seams": len(seams),
                "rung_hits_on_seam": tp,
                "precision_rung": tp / len(seams) if seams else 0.0,
                "note": "seam≡0 only predicts rung when p itself is a rung quantum",
            }
        )
    results["per_context_seam_vs_rung"] = per_p
    results["winner"] = max(
        (
            ("cf_surplus_band", results["cf_surplus_band_ge_0.9"]["f1"]),
            ("seam_any", results["seam_any_context"]["f1"]),
            ("twin_sheets", results["twin_sheets_any_context"]["f1"]),
        ),
        key=lambda x: x[1],
    )
    return results


# ---------------------------------------------------------------------------
# P2 Recursion — scale transport
# ---------------------------------------------------------------------------

def recursion_scale_transport() -> Dict[str, Any]:
    """Local bound p is not global; next child needs α/CF, not 2p or p²."""
    rows = []
    for i, p in enumerate(CF_LADDER[:-1]):
        child = CF_LADDER[i + 1]
        candidates = {
            "2p": 2 * p,
            "p2": p * p,
            "p_plus_1_times_something": None,
            "cf_next": child,
        }
        # distance of naive extrapolations to actual CF child
        d2 = abs(2 * p - child)
        dsq = abs(p * p - child)
        # also check typed center sheets near child
        rows.append(
            {
                "parent": p,
                "cf_child": child,
                "naive_2p": 2 * p,
                "naive_p2": p * p,
                "err_2p": d2,
                "err_p2": dsq,
                "phi_parent": format(frac_Q_alpha(p), "f"),
                "phi_child": format(frac_Q_alpha(child), "f"),
                "bits_parent": bits(p) if p <= 2000 else None,
                "bits_child": bits(child) if child <= 2000 else None,
                "ratio_child_parent": format(Decimal(child) / Decimal(p), "f"),
            }
        )
    # k-cycle rechart: for p=5, lengths 5,10,15,...,40 and ask if recursive
    # seam alone produces next rung 41 — it does not.
    p = 5
    k_seams = [k * p for k in range(1, 12)]
    return {
        "cf_ladder": list(CF_LADDER),
        "transport_rows": rows,
        "verdict": (
            "next_child_needs_alpha_CF_not_2p_or_p2"
            if all(r["err_2p"] > 0 and r["err_p2"] > 0 for r in rows)
            else "unexpected_naive_hit"
        ),
        "k_cycle_seams_of_5": k_seams,
        "rung_41_is_seam_of_5": 41 % 5 == 1,  # local=1, not seam
        "rung_41_chart_in_p5": packet_chart(41, 5),
        "rung_306_chart_in_p41": packet_chart(306, 41),
        "odd_sheet_vs_even_306": {
            "typed_center_explains_odd": True,
            "seam_rechart_alone_explains_306": False,
            "note": (
                "Recursive rechart at kp gives twins kp±1 (odd when p odd) — "
                "same parity fact as typed sheets. Even 306 is CF surplus rung, "
                "not a typed sheet; k-cycle of odd p never yields even sheet."
            ),
        },
    }


# ---------------------------------------------------------------------------
# P3 Nesting — false identification check
# ---------------------------------------------------------------------------

def nesting_identification() -> Dict[str, Any]:
    """Is Law-C digit nesting the packing analogue of preserving every 0→1 step?"""
    # Compare flat digit path vs one-level nest on 306 multiples (theory + prior timing note)
    rows = []
    for n in (306, 612, 1224, 665, 1330):
        b58 = theory_bytes_5_8(n)
        b41 = theory_bytes_41_65(n)
        b306 = theory_bytes_306_485(n)
        # "preserving every 0→1" strawman: nest 5 inside 41 inside 306 whenever divisible
        nested_claim = "would_force_hierarchy_always"
        rows.append(
            {
                "n": n,
                "flat_5_8": b58,
                "fiber_41": b41,
                "rung_306": b306,
                "best_flat_family": min(b58, b41, b306),
                "306_beats_5_8": b306 < b58,
                "nest_strawman": nested_claim,
            }
        )
    return {
        "claim_under_test": (
            "UFRF 'preserve every 0→1 step' ≡ Law-C hierarchical digit nesting"
        ),
        "rows": rows,
        "prior_pipeline": {
            "associator_flat_wins_pct": "~92-98",
            "nest_decode_vs_5_8": "~8x slower",
            "recommendation": "keep_never_nested_rule",
        },
        "verdict_tag": "false_identification",
        "correct_reading": (
            "Preserving 0→1 steps is a *typed chart* discipline (retain cycle+local). "
            "Digit nesting is a *container composition* choice. Same language of "
            "hierarchy; different object. Only keep digit nesting when bytes+speed win."
        ),
    }


# ---------------------------------------------------------------------------
# P4 Hierarchy map
# ---------------------------------------------------------------------------

def hierarchy_map() -> Dict[str, Any]:
    """Typed layers vs packing ledgers — refuse 0=1 flatten."""
    layers = [
        {
            "layer": "C0",
            "context": 0,
            "ufrf": "signed source/mirror; sheets ±1; div/mod chart not appropriate",
            "packing_owner": None,
            "operator": "none — not a trit length",
            "flatten_risk": "calling seed 'completion' would erase source typing",
        },
        {
            "layer": "C1",
            "context": 1,
            "ufrf": "seed/completion grammar 1→{1,3}; center 2",
            "packing_owner": "fmt_5_8 atom (rung 5 appears later as C≥3 sheet)",
            "operator": "smallest practical block; not identified with context 1",
            "flatten_risk": "equating C1 with fmt_5_8 conflates seed grammar with CF rung 5",
        },
        {
            "layer": "C_odd_prime",
            "context": "3,5,7,…",
            "ufrf": "field-like interior; sheets 2mc±1; twin seam kp±1",
            "packing_owner": "odd rungs 5,19,41,53,665 as design-time sheets / complements",
            "operator": "sheet/complement tools OK; fiber-41 when length aligns",
            "flatten_risk": "treating every odd length as sheet without (c,m) typing",
        },
        {
            "layer": "even_CF_rung",
            "context": "306 (composite)",
            "ufrf": "not a typed sheet (sheets always odd)",
            "packing_owner": "fmt_306_485 rung-block ledger",
            "operator": "frame_only — never typed-sheet search",
            "flatten_risk": "forcing 306 into twin-sheet language (0=1 style collapse)",
        },
    ]
    rechart = {
        "at_n_equals_p": {
            "parent_chart": "completion cycle=1",
            "child_chart": "local seed=0",
            "not": "0=1",
            "packing_analogue": (
                "exact block boundary: remainder 0 in fmt_p; next block starts fresh. "
                "Two ledgers meet; they are not the same symbol."
            ),
        }
    }
    return {"layers": layers, "typed_rechart": rechart}


# ---------------------------------------------------------------------------
# Practical metrics
# ---------------------------------------------------------------------------

def twin_vs_interior_metric(
    contexts: Sequence[int] = (3, 5, 7, 11, 13, 17, 19, 41),
    k_max: int = 20,
) -> Dict[str, Any]:
    """Compare twin sheets kp±1 vs interior positions on surplus / zt / bytes."""
    twin_stats = []
    interior_stats = []
    for p in contexts:
        for k in range(1, k_max + 1):
            trip = twin_seam_triple(k, p)
            for label, n in (("prev", trip["prev_n"]), ("next", trip["next_n"])):
                twin_stats.append(tag_length(n))
            # interiors: locals 2..p-2 at this cycle (if p>3)
            if p > 3:
                for loc in range(2, min(p - 1, 6)):  # sample a few interiors
                    n = (k - 1) * p + loc if k >= 1 else loc
                    if n <= 0:
                        continue
                    # only interiors of current packet k: n = (k-1)*p + loc with 1<loc<p
                    n = k * p - p + loc  # in packet k-? use cycle k's packet: kp is seam
                    # packet for cycle index (k): positions (k-1)*p+1 .. kp-1
                    n = (k - 1) * p + loc
                    if 1 < loc < p - 1 and n > 0:
                        interior_stats.append(tag_length(n))

    def agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0}
        phis = [Decimal(r["phi"]) for r in rows if r["phi"] is not None]
        d1s = [Decimal(r["dist_to_1"]) for r in rows if r.get("dist_to_1")]
        zt = [r["max_m_zt"] for r in rows]
        rung_hits = sum(1 for r in rows if r["rung"])
        mean_phi = sum(phis) / len(phis)
        mean_d1 = sum(d1s) / len(d1s)
        return {
            "n": len(rows),
            "mean_phi": format(mean_phi, "f"),
            "mean_dist_to_1": format(mean_d1, "f"),
            "mean_max_m_zt": sum(zt) / len(zt),
            "rung_hit_rate": rung_hits / len(rows),
            "rung_hits": rung_hits,
        }

    tw = agg(twin_stats)
    inn = agg(interior_stats)
    # Also: do twins land on packing rungs more than interiors?
    return {
        "twins": tw,
        "interiors_sampled": inn,
        "twins_higher_rung_rate": tw.get("rung_hit_rate", 0) > inn.get("rung_hit_rate", 0),
        "twins_closer_to_1": (
            Decimal(tw["mean_dist_to_1"]) < Decimal(inn["mean_dist_to_1"])
            if tw.get("mean_dist_to_1") and inn.get("mean_dist_to_1")
            else None
        ),
        "note": (
            "Twins are always odd when p odd — necessary for typed sheets / odd rungs, "
            "not sufficient for surplus or byte wins."
        ),
    }


def nearest_multiple(x: int, q: int, mode: str = "nearest") -> int:
    if x <= 0:
        return q
    if mode == "nearest":
        down = (x // q) * q
        up = down + q
        if down <= 0:
            return up
        return up if abs(up - x) < abs(x - down) else (down if abs(x - down) <= abs(up - x) else up)
    if mode == "up":
        return ((x + q - 1) // q) * q
    return max(q, (x // q) * q)


def dist_to_306_multiple(x: int, q: int = 306) -> Dict[str, Any]:
    """Signed / absolute distance from x to nearest multiple of q."""
    if x <= 0:
        return {"x": x, "down": q, "up": q, "nearest": q, "abs_dist": q, "signed_dist": q}
    down = (x // q) * q
    up = down + q if down < x or x % q else down
    if x % q == 0:
        return {"x": x, "down": x, "up": x, "nearest": x, "abs_dist": 0, "signed_dist": 0, "residue": 0}
    # prefer down on ties (same as nearest_multiple)
    if abs(up - x) < abs(x - down):
        nearest, signed = up, up - x
    else:
        nearest, signed = down, down - x
    return {
        "x": x,
        "down": down,
        "up": up,
        "nearest": nearest,
        "abs_dist": abs(signed),
        "signed_dist": signed,
        "residue": x % q,
    }


def hybrid_306_58_bytes(n: int) -> int:
    """fmt_306_485 on divisible prefix + fmt_5_8 on remainder (fixed trit count)."""
    if n <= 0:
        return 0
    full, rem = divmod(n, 306)
    return full * 61 + theory_bytes_5_8(rem)


def hybrid_tensor_bytes(m: int, n: int) -> Dict[str, Any]:
    """Best of flat-stream / row-fiber / col-fiber hybrid packing."""
    mn = m * n
    opts = [
        {"layout": "flat_stream", "bytes": hybrid_306_58_bytes(mn)},
        {"layout": "row_fibers", "bytes": m * hybrid_306_58_bytes(n)},
        {"layout": "col_fibers", "bytes": n * hybrid_306_58_bytes(m)},
    ]
    best = min(opts, key=lambda o: o["bytes"])
    return {"best": best, "all": opts, "flat_5_8": theory_bytes_5_8(mn)}


def _pack_summary(m: int, n: int) -> Dict[str, Any]:
    d = pack_tensor(m, n)
    dec = d["decision"]
    return {
        "selected": dec["bytes"],
        "flat_5_8": d["candidates"]["flat_5_8"],
        "path": dec["path"],
        "ledger": dec["ledger"],
    }


def bitnet_306_redesign(
    ckpt: Path,
    quantum: int = 306,
    mode: str = "nearest",
) -> Dict[str, Any]:
    """Counterfactual: redesign each mode to a multiple of quantum (no pad).

    Payload count changes with the redesign (train-time width change), unlike
    post-hoc zero-pad which keeps MN and adds waste.

    mode: 'nearest' | 'down' | 'up'
    """
    if not ckpt.is_file():
        return {"skipped": True, "reason": f"missing ckpt {ckpt}"}

    shapes = load_bitnet_shapes(ckpt)
    # unique modes
    modes = sorted({m for _, m, n in shapes} | {n for _, m, n in shapes})
    mode_map = {m: nearest_multiple(m, quantum, mode) for m in modes}

    before_total = 0
    after_total = 0
    before_flat = 0
    after_flat = 0
    path_before: Dict[str, int] = defaultdict(int)
    path_after: Dict[str, int] = defaultdict(int)
    per_unique = []
    seen = set()
    trit_before = 0
    trit_after = 0

    for name, m, n in shapes:
        key = (m, n)
        b = _pack_summary(m, n)
        m2, n2 = mode_map[m], mode_map[n]
        a = _pack_summary(m2, n2)
        before_total += b["selected"]
        after_total += a["selected"]
        before_flat += b["flat_5_8"]
        after_flat += a["flat_5_8"]
        path_before[b["path"]] += 1
        path_after[a["path"]] += 1
        trit_before += m * n
        trit_after += m2 * n2
        if key not in seen:
            seen.add(key)
            per_unique.append(
                {
                    "shape": [m, n],
                    "redesign": [m2, n2],
                    "delta_mode": [m2 - m, n2 - n],
                    "before": b,
                    "after": a,
                    "delta_selected": a["selected"] - b["selected"],
                    "trit_delta": m2 * n2 - m * n,
                }
            )

    dens_b = before_total / trit_before if trit_before else None
    dens_a = after_total / trit_after if trit_after else None

    return {
        "skipped": False,
        "quantum": quantum,
        "redesign_mode": mode,
        "n_tensors": len(shapes),
        "n_unique_shapes": len(seen),
        "mode_map": {str(k): v for k, v in mode_map.items()},
        "mode_map_sample": {str(k): v for k, v in list(mode_map.items())[:12]},
        "totals": {
            "selected_before": before_total,
            "selected_after": after_total,
            "delta_selected": after_total - before_total,
            "flat58_before": before_flat,
            "flat58_after": after_flat,
            "trits_before": trit_before,
            "trits_after": trit_after,
            "trit_delta": trit_after - trit_before,
            "bytes_per_trit_before": dens_b,
            "bytes_per_trit_after": dens_a,
            "density_delta": (dens_a - dens_b) if dens_b and dens_a else None,
            "density_delta_pct_of_before": (
                100.0 * (dens_a - dens_b) / dens_b if dens_b and dens_a else None
            ),
        },
        "path_counts_before": dict(path_before),
        "path_counts_after": dict(path_after),
        "frame_path_after": path_after.get("frame", 0),
        "per_unique": per_unique,
        "verdict": (
            "density_improved"
            if dens_a is not None and dens_b is not None and dens_a < dens_b
            else "density_not_improved"
        ),
    }


def bitnet_306_pad_up(ckpt: Path, quantum: int = 306) -> Dict[str, Any]:
    """Counterfactual C: pad each mode UP to next ≥ multiple of quantum.

    Packs padded length; useful density = packed_bytes / original_trits (expect hurt).
    """
    if not ckpt.is_file():
        return {"skipped": True, "reason": f"missing ckpt {ckpt}"}

    shapes = load_bitnet_shapes(ckpt)
    modes = sorted({m for _, m, n in shapes} | {n for _, m, n in shapes})
    mode_map = {m: nearest_multiple(m, quantum, "up") for m in modes}

    orig_bytes = 0
    pad_bytes = 0
    orig_flat = 0
    trit_orig = 0
    trit_pad = 0
    path_pad: Dict[str, int] = defaultdict(int)
    per_unique = []
    seen = set()

    for name, m, n in shapes:
        b = _pack_summary(m, n)
        m2, n2 = mode_map[m], mode_map[n]
        a = _pack_summary(m2, n2)
        orig_bytes += b["selected"]
        pad_bytes += a["selected"]
        orig_flat += b["flat_5_8"]
        trit_orig += m * n
        trit_pad += m2 * n2
        path_pad[a["path"]] += 1
        key = (m, n)
        if key not in seen:
            seen.add(key)
            per_unique.append(
                {
                    "shape": [m, n],
                    "padded": [m2, n2],
                    "pad_delta_mode": [m2 - m, n2 - n],
                    "before": b,
                    "after_pad_pack": a,
                    "delta_selected": a["selected"] - b["selected"],
                    "trit_pad_delta": m2 * n2 - m * n,
                }
            )

    dens_orig = orig_bytes / trit_orig if trit_orig else None
    dens_pad_on_pad = pad_bytes / trit_pad if trit_pad else None
    dens_useful = pad_bytes / trit_orig if trit_orig else None  # hurt metric

    return {
        "skipped": False,
        "quantum": quantum,
        "counterfactual": "C_pad_up",
        "n_tensors": len(shapes),
        "mode_map": {str(k): v for k, v in mode_map.items()},
        "totals": {
            "selected_before": orig_bytes,
            "selected_after_pad": pad_bytes,
            "delta_selected": pad_bytes - orig_bytes,
            "flat58_before": orig_flat,
            "trits_original": trit_orig,
            "trits_padded": trit_pad,
            "trit_pad_delta": trit_pad - trit_orig,
            "bytes_per_trit_before": dens_orig,
            "bytes_per_padded_trit": dens_pad_on_pad,
            "bytes_per_useful_trit_after_pad": dens_useful,
            "useful_density_delta": (dens_useful - dens_orig) if dens_useful and dens_orig else None,
        },
        "path_counts_after_pad": dict(path_pad),
        "frame_path_after": path_pad.get("frame", 0),
        "per_unique": per_unique,
        "verdict": (
            "pad_hurts_useful_density"
            if dens_useful is not None and dens_orig is not None and dens_useful > dens_orig
            else "pad_did_not_hurt"
        ),
    }


def bitnet_hybrid_306_58(ckpt: Path) -> Dict[str, Any]:
    """Counterfactual D: keep trit count; 306-prefix + 5_8 rem (best layout)."""
    if not ckpt.is_file():
        return {"skipped": True, "reason": f"missing ckpt {ckpt}"}

    shapes = load_bitnet_shapes(ckpt)
    flat_total = 0
    hybrid_total = 0
    selected_packer = 0
    pure306_total = 0
    trit_total = 0
    per_unique = []
    seen = set()
    layout_counts: Dict[str, int] = defaultdict(int)

    for name, m, n in shapes:
        mn = m * n
        flat = theory_bytes_5_8(mn)
        hy = hybrid_tensor_bytes(m, n)
        pure = theory_bytes_306_485(mn)  # rem via bigint container, not 5_8
        pk = _pack_summary(m, n)
        flat_total += flat
        hybrid_total += hy["best"]["bytes"]
        pure306_total += pure
        selected_packer += pk["selected"]
        trit_total += mn
        layout_counts[hy["best"]["layout"]] += 1
        key = (m, n)
        if key not in seen:
            seen.add(key)
            per_unique.append(
                {
                    "shape": [m, n],
                    "flat_5_8": flat,
                    "hybrid_best": hy["best"],
                    "hybrid_all": hy["all"],
                    "theory_306_with_bigint_rem": pure,
                    "packer_selected": pk,
                    "hybrid_minus_flat": hy["best"]["bytes"] - flat,
                }
            )

    dens_flat = flat_total / trit_total if trit_total else None
    dens_hy = hybrid_total / trit_total if trit_total else None
    dens_pure = pure306_total / trit_total if trit_total else None

    return {
        "skipped": False,
        "counterfactual": "D_hybrid_306_prefix_5_8_rem",
        "n_tensors": len(shapes),
        "n_unique_shapes": len(seen),
        "layout_counts": dict(layout_counts),
        "totals": {
            "flat_5_8": flat_total,
            "hybrid_306_58": hybrid_total,
            "theory_306_bigint_rem": pure306_total,
            "packer_selected": selected_packer,
            "trits": trit_total,
            "bytes_per_trit_flat_5_8": dens_flat,
            "bytes_per_trit_hybrid": dens_hy,
            "bytes_per_trit_306_bigint_rem": dens_pure,
            "hybrid_minus_flat": hybrid_total - flat_total,
            "hybrid_density_delta": (dens_hy - dens_flat) if dens_hy and dens_flat else None,
        },
        "per_unique": per_unique,
        "verdict": (
            "hybrid_beats_flat"
            if hybrid_total < flat_total
            else "hybrid_does_not_beat_flat"
        ),
    }


def exact_density_theory() -> Dict[str, Any]:
    """Exact B/trit on block multiples: 61/306 vs 62/306 vs n/5."""
    n = 306
    b58 = theory_bytes_5_8(n)  # ceil → 62
    b306 = theory_bytes_306_485(n)  # 61
    # exact n/5 when n≡0 (mod 5): 306 is not ÷5, so also show k*306 with k*306÷5
    # 306*5=1530 is ÷5; density of flat on exact ÷5 streams is exactly 0.2
    n5 = 1530  # 5×306
    return {
        "block": n,
        "fmt_306_485_bytes": b306,
        "fmt_5_8_ceil_bytes": b58,
        "B_per_trit": {
            "fmt_306_485": b306 / n,  # 61/306
            "fmt_5_8_on_exact_306_block": b58 / n,  # 62/306
            "fmt_5_8_exact_multiple_of_5": 1.0 / 5,  # 0.2
            "ratio_61_over_306": 61 / 306,
            "ratio_62_over_306": 62 / 306,
        },
        "delta_vs_0p2": {
            "absolute_B_per_trit": 0.2 - (61 / 306),
            "pct_of_0p2": 100.0 * (0.2 - 61 / 306) / 0.2,
            "bytes_saved_per_306_trits_vs_0p2": 0.2 * 306 - 61,  # 0.2
            "bytes_saved_per_306_trits_vs_ceil5_8": b58 - b306,  # 1
            "pct_vs_ceil_5_8_block": 100.0 * (b58 - b306) / b58,
        },
        "why_reported_0p199346": {
            "value": 61 / 306,
            "matches_reported": abs((61 / 306) - 0.19934640522875818) < 1e-15,
            "explanation": (
                "After nearest×306 redesign every mode is ÷306, packer takes "
                "fmt_306_485 everywhere → density collapses exactly to 61/306. "
                "Baseline 0.200 is flat 5_8 on BitNet lengths that are ÷5 "
                "(exact n/5), not the 62/306 ceil-on-block figure."
            ),
        },
        "cross_check_5x306": {
            "n": n5,
            "bytes_5_8": theory_bytes_5_8(n5),
            "bytes_306": theory_bytes_306_485(n5),
            "B_per_trit_5_8": theory_bytes_5_8(n5) / n5,
            "B_per_trit_306": theory_bytes_306_485(n5) / n5,
            "saved_bytes": theory_bytes_5_8(n5) - theory_bytes_306_485(n5),
        },
    }


def scale_density_projection() -> Dict[str, Any]:
    """Project MB saved if fully 306-aligned ternary weights use fmt_306_485."""
    # vs exact 0.2 (5_8 on ÷5 lengths): save 0.2 B per 306 trits = trit_count/1530
    # vs ceil 5_8 on exact 306 blocks (62 B): save 1 B per 306 = trit_count/306
    classes = {
        "BitNet_2B_measured_trits": 2_084_044_800,  # from prior probe
        "class_2B_approx": 2_000_000_000,
        "class_7B_approx": 7_000_000_000,
        "class_70B_approx": 70_000_000_000,
    }
    rows = []
    for label, trits in classes.items():
        # assume fully ÷306
        b306 = trits * 61 // 306
        b58_exact02 = trits // 5  # if also ÷5
        b58_ceil_block = trits * 62 // 306  # if packed as 306-blocks with 5_8 ceil
        save_vs_02 = b58_exact02 - b306
        save_vs_ceil = b58_ceil_block - b306
        rows.append(
            {
                "label": label,
                "trits": trits,
                "bytes_fmt_306_485": b306,
                "bytes_5_8_at_0p2": b58_exact02,
                "bytes_5_8_ceil_per_306": b58_ceil_block,
                "saved_vs_0p2_bytes": save_vs_02,
                "saved_vs_0p2_MB": save_vs_02 / (1024 * 1024),
                "saved_vs_0p2_pct": 100.0 * save_vs_02 / b58_exact02,
                "saved_vs_ceil_block_bytes": save_vs_ceil,
                "saved_vs_ceil_block_MB": save_vs_ceil / (1024 * 1024),
                "saved_vs_ceil_block_pct": 100.0 * save_vs_ceil / b58_ceil_block,
                "formula": (
                    "save_vs_0p2 ≈ trit_count/1530 B (= trit_count×0.2/306); "
                    "save_vs_ceil ≈ trit_count/306 B (= 1 B per 306-block)"
                ),
            }
        )
    return {
        "theory": {
            "B_per_trit_306": 61 / 306,
            "B_per_trit_5_8_exact": 0.2,
            "pct_of_5_8_size": 100.0 * (0.2 - 61 / 306) / 0.2,  # ~0.327%
            "pct_vs_ceil_62": 100.0 * 1 / 62,  # ~1.613%
        },
        "projections": rows,
    }


def mode_proximity_to_306(ckpt: Path, quantum: int = 306) -> Dict[str, Any]:
    """How close are BitNet unique mode lengths to multiples of 306?"""
    if not ckpt.is_file():
        return {"skipped": True, "reason": f"missing ckpt {ckpt}"}

    shapes = load_bitnet_shapes(ckpt)
    modes = sorted({m for _, m, n in shapes} | {n for _, m, n in shapes})
    # residue frequency under uniform null
    # E[min(r, q-r)] for r uniform in 0..q-1
    expected_abs = sum(min(r, quantum - r) for r in range(quantum)) / quantum
    # also E for r in 1..q-1 (never exact) — same almost
    mode_rows = []
    for m in modes:
        d = dist_to_306_multiple(m, quantum)
        # how often this mode appears as a matrix dim
        n_as_m = sum(1 for _, a, b in shapes if a == m)
        n_as_n = sum(1 for _, a, b in shapes if b == m)
        mode_rows.append({**d, "n_tensors_as_row": n_as_m, "n_tensors_as_col": n_as_n})

    abs_dists = [r["abs_dist"] for r in mode_rows]
    mean_abs = sum(abs_dists) / len(abs_dists) if abs_dists else None
    # trit-weighted: weight each mode by sum of trits touching it
    trit_w = []
    for r in mode_rows:
        m = r["x"]
        w = sum(a * b for _, a, b in shapes if a == m or b == m)
        trit_w.append((r["abs_dist"], w))
    wsum = sum(w for _, w in trit_w)
    mean_abs_trit = sum(d * w for d, w in trit_w) / wsum if wsum else None

    # random null comparison: fraction of modes closer than expected
    closer_than_E = sum(1 for d in abs_dists if d < expected_abs)
    # Monte-ish: probability a random residue has abs_dist <= observed median
    median_abs = sorted(abs_dists)[len(abs_dists) // 2] if abs_dists else None

    return {
        "skipped": False,
        "quantum": quantum,
        "n_unique_modes": len(modes),
        "modes": mode_rows,
        "stats": {
            "mean_abs_dist": mean_abs,
            "mean_abs_dist_trit_weighted": mean_abs_trit,
            "min_abs_dist": min(abs_dists) if abs_dists else None,
            "max_abs_dist": max(abs_dists) if abs_dists else None,
            "median_abs_dist": median_abs,
            "n_exact_multiples": sum(1 for d in abs_dists if d == 0),
            "null_uniform_E_abs_dist": expected_abs,
            "n_modes_closer_than_null_E": closer_than_E,
            "frac_modes_closer_than_null_E": closer_than_E / len(abs_dists) if abs_dists else None,
            "max_possible_abs_dist": quantum // 2,
        },
        "verdict": (
            "no_systematic_seam_proximity"
            if mean_abs is not None and mean_abs >= expected_abs * 0.5
            else "closer_than_null"
        ),
        "note": (
            "Only 3 unique BitNet modes (640, 2560, 6912). Small-N proximity "
            "test — treat as descriptive, not a geometric discovery."
        ),
    }


def density_signal(ckpt: Path, quantum: int = 306) -> Dict[str, Any]:
    """Dissect the ~0.33% BitNet density delta: confounders, theory, scale, proximity."""
    theory = exact_density_theory()
    nearest = bitnet_306_redesign(ckpt, quantum, "nearest")
    down = bitnet_306_redesign(ckpt, quantum, "down")
    up = bitnet_306_redesign(ckpt, quantum, "up")
    pad = bitnet_306_pad_up(ckpt, quantum)
    hybrid = bitnet_hybrid_306_58(ckpt)
    prox = mode_proximity_to_306(ckpt, quantum)
    scale = scale_density_projection()

    # Confounder split for nearest: absolute byte drop from trit shrink vs density
    confound = None
    if not nearest.get("skipped"):
        t = nearest["totals"]
        # Counterfactual: same after-trits packed at before density (0.2)
        bytes_if_same_density = t["trits_after"] * t["bytes_per_trit_before"]
        byte_drop_from_fewer_trits = t["selected_before"] - bytes_if_same_density
        byte_drop_from_density = bytes_if_same_density - t["selected_after"]
        confound = {
            "absolute_byte_delta": t["delta_selected"],
            "byte_drop_from_fewer_trits": byte_drop_from_fewer_trits,
            "byte_drop_from_true_density": byte_drop_from_density,
            "frac_drop_from_trits": (
                byte_drop_from_fewer_trits / abs(t["delta_selected"])
                if t["delta_selected"]
                else None
            ),
            "frac_drop_from_density": (
                byte_drop_from_density / abs(t["delta_selected"])
                if t["delta_selected"]
                else None
            ),
            "note": (
                "Most absolute byte drop is fewer trits (e.g. 2560→2448). "
                "True packing signal is density_delta → exactly 61/306 − 0.2."
            ),
        }

    # Honest geometric verdict
    dens_match = False
    if not nearest.get("skipped") and nearest["totals"].get("bytes_per_trit_after"):
        dens_match = abs(nearest["totals"]["bytes_per_trit_after"] - 61 / 306) < 1e-12

    verdict = {
        "signal_class": "known_rung_block_density",
        "is_new_geometry": False,
        "matches_61_over_306": dens_match,
        "train_time_align_worth_it": (
            "optional_greenfield_prior_only — ~0.33% vs 5_8 (or ~1.61% vs "
            "ceil-on-306-blocks); capacity/accuracy tradeoff dominates; "
            "no evidence BitNet modes sit systematically on 306 seams"
        ),
        "summary": (
            "The 0.200→0.199346 figure is exactly 0.2→61/306. Redesign enables "
            "the already-measured −1 B/306 rung-block win; it does not reveal a "
            "new packet/seam geometric effect. Mode distances to ×306 are not "
            "closer than a uniform-residue null (descriptive, N=3 modes)."
        ),
    }

    return {
        "theory": theory,
        "confounder_split_nearest": confound,
        "A_redesign_down": {
            k: down[k]
            for k in (
                "skipped",
                "redesign_mode",
                "mode_map",
                "totals",
                "frame_path_after",
                "verdict",
                "n_tensors",
            )
            if k in down
        },
        "B_redesign_up": {
            k: up[k]
            for k in (
                "skipped",
                "redesign_mode",
                "mode_map",
                "totals",
                "frame_path_after",
                "verdict",
                "n_tensors",
            )
            if k in up
        },
        "nearest_for_ref": {
            k: nearest[k]
            for k in (
                "skipped",
                "redesign_mode",
                "mode_map",
                "totals",
                "frame_path_after",
                "verdict",
                "n_tensors",
            )
            if k in nearest
        },
        "C_pad_up": {
            k: pad[k]
            for k in (
                "skipped",
                "counterfactual",
                "mode_map",
                "totals",
                "frame_path_after",
                "verdict",
                "n_tensors",
            )
            if k in pad
        },
        "D_hybrid": {
            k: hybrid[k]
            for k in (
                "skipped",
                "counterfactual",
                "totals",
                "layout_counts",
                "verdict",
                "n_tensors",
                "per_unique",
            )
            if k in hybrid
        },
        "mode_proximity": prox,
        "scale_projection": scale,
        "verdict": verdict,
    }


def synthetic_multiples() -> Dict[str, Any]:
    rows = []
    for m, n in (
        (1, 306),
        (2, 306),
        (4, 306),
        (1, 612),
        (41, 306),
        (306, 306),
        (1, 665),
        (2, 665),
        (5, 665),
        (2560, 2560),
    ):
        d = pack_tensor(m, n)
        dec = d["decision"]
        rows.append(
            {
                "shape": [m, n],
                "path": dec["path"],
                "ledger": dec["ledger"],
                "bytes": dec["bytes"],
                "flat_5_8": d["candidates"]["flat_5_8"],
                "delta_vs_5_8": dec["bytes"] - d["candidates"]["flat_5_8"],
                "reason": dec["reason"],
            }
        )
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Claim board + filter tags
# ---------------------------------------------------------------------------

def claim_board(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    pat = results["pattern"]
    rec = results["recursion"]
    nest = results["nesting"]
    twin = results["twin_vs_interior"]
    bn = results.get("bitnet_306_redesign", {})

    winner = pat.get("winner", ("?", 0))
    claims = [
        {
            "id": "P1_seam_predicts_events",
            "claim": "(⌊n/p⌋, n mod p) seam/twin chart predicts packing events better than CF phase",
            "evidence": {
                "winner": winner,
                "cf_f1": pat.get("cf_surplus_band_ge_0.9", {}).get("f1"),
                "seam_f1": pat.get("seam_any_context", {}).get("f1"),
                "twin_f1": pat.get("twin_sheets_any_context", {}).get("f1"),
            },
            "tag": (
                "does_not_apply"
                if winner[0] == "cf_surplus_band"
                else "applies_as_language"
            ),
            "note": "CF surplus band wins event prediction; seam is the *definition* of ledger quantum, not an independent predictor.",
        },
        {
            "id": "P1_seam_is_ledger_quantum",
            "claim": "local=0 for p∈{306,665} is exactly when frame ledger may fire",
            "evidence": "by construction in ledger_packer FRAME_SPECS",
            "tag": "applies_operationally",
            "note": "Already in packer. Packet chart language restates the divisibility gate.",
        },
        {
            "id": "P2_local_bound_not_global",
            "claim": "p bounds one cycle only; next scale needs α/CF transport",
            "evidence": rec.get("verdict"),
            "tag": "applies_as_language",
            "note": "Falsifies 2p/p² as child law. Does not change packer; protects against wrong rung extrapolation.",
        },
        {
            "id": "P2_recursion_explains_odd_even",
            "claim": "Recursive rechart explains odd-sheet vs even-306 better than typed center",
            "evidence": rec.get("odd_sheet_vs_even_306"),
            "tag": "does_not_apply",
            "note": "Rechart gives same parity fact (odd p → odd twins). Typed center already owns odd sheets; 306 needs CF.",
        },
        {
            "id": "P3_nesting_eq_preserve_01",
            "claim": "Law-C digit nesting ≡ preserve every 0→1 step",
            "evidence": nest.get("verdict_tag"),
            "tag": "false_identification",
            "note": nest.get("correct_reading"),
        },
        {
            "id": "P4_hierarchy_no_flatten",
            "claim": "Keep C0/C1/odd-prime/even-CF as separate owners of operators",
            "evidence": "hierarchy_map layers",
            "tag": "applies_as_language",
            "note": "Clarifies odd/even packer rule; no new byte path.",
        },
        {
            "id": "P4_typed_rechart_not_0eq1",
            "claim": "At n=p, completion∩seed is typed rechart not 0=1",
            "evidence": "block boundary = remainder 0; two ledgers meet",
            "tag": "applies_as_language",
            "note": "Prevents collapsing fmt_5_8 into fmt_306_485 at seams.",
        },
        {
            "id": "M_twin_vs_interior",
            "claim": "Twin sheets kp±1 beat interiors on rung/surplus metrics",
            "evidence": {
                "twins": twin.get("twins"),
                "interiors": twin.get("interiors_sampled"),
                "twins_higher_rung": twin.get("twins_higher_rung_rate"),
                "twins_closer_to_1": twin.get("twins_closer_to_1"),
            },
            "tag": (
                "applies_as_language"
                if twin.get("twins_higher_rung_rate")
                else "does_not_apply"
            ),
            "note": "Necessary for odd-rung incidence language; not a byte selector.",
        },
        {
            "id": "M_bitnet_306_redesign",
            "claim": "Redesign BitNet modes to nearest ×306 improves packing density",
            "evidence": bn.get("totals") if not bn.get("skipped") else bn,
            "tag": (
                "applies_operationally"
                if bn.get("verdict") == "density_improved"
                else "does_not_apply"
            ),
            "note": (
                "Greenfield only: nearest×306 can shrink widths (e.g. 2560→2448) and "
                "cuts trit count. Density gain is the known fmt_306_485 gap (~0.33% vs "
                "5_8), not a new geometric miracle. Prefer picking 2448/2754 at design "
                "time; never post-hoc pad."
            ),
        },
        {
            "id": "M_density_signal_61_306",
            "claim": "0.200→0.199346 is a new geometric seam signal beyond known −1 B/306",
            "evidence": (results.get("density_signal") or {}).get("verdict"),
            "tag": "does_not_apply",
            "note": (
                "Exact restatement of 61/306 vs 0.2. Confounder split: absolute byte "
                "drop is mostly fewer trits; density delta matches nesting/rung test. "
                "Mode proximity to ×306 is not above uniform null (N=3 modes)."
            ),
        },
    ]
    return claims


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(ckpt: Path = DEFAULT_CKPT) -> Dict[str, Any]:
    t0 = time.time()
    events = packing_events(1, 800)
    # contexts: primes + packing rungs + composites
    contexts = list(PRIMES[:10]) + [41, 53, 306, 665]
    pattern = pattern_predictivity(events, contexts)
    recursion = recursion_scale_transport()
    nesting = nesting_identification()
    hierarchy = hierarchy_map()
    twin = twin_vs_interior_metric()
    synth = synthetic_multiples()
    bitnet = bitnet_306_redesign(ckpt, mode="nearest")
    densig = density_signal(ckpt)

    out: Dict[str, Any] = {
        "stance": (
            "0.33% = known 61/306 rung-block, not seam discovery. "
            "Keep: design-time ×306 and optional hybrid prefix. "
            "No new default packer rules; no pad; no theory reopen. "

            "Test 0→1 packet / prime-context framing against packing metrics. "
            "Tag claims; do not reopen theory without a metric win."
        ),
        "pattern": pattern,
        "recursion": recursion,
        "nesting": nesting,
        "hierarchy": hierarchy,
        "twin_vs_interior": twin,
        "synthetic_multiples": synth,
        "bitnet_306_redesign": bitnet,
        "density_signal": densig,
        "elapsed_s": time.time() - t0,
    }
    out["claims"] = claim_board(out)

    # Summary counts
    tags = defaultdict(int)
    for c in out["claims"]:
        tags[c["tag"]] += 1
    out["tag_counts"] = dict(tags)
    dens = (out.get("bitnet_306_redesign") or {}).get("totals") or {}
    dens_note = dens.get("density_delta")
    ds_verdict = (densig or {}).get("verdict") or {}
    out["overall_verdict"] = {
        "summary": "packet_language_clarifies_gates_nesting_is_false_id_306_redesign_is_tiny_density_win",
        "pattern_winner": out["pattern"].get("winner"),
        "nesting": "false_identification",
        "bitnet_306_redesign_density_delta": dens_note,
        "density_signal_class": ds_verdict.get("signal_class"),
        "density_signal_is_new_geometry": ds_verdict.get("is_new_geometry"),
        "operational_moves": [
            c["id"] for c in out["claims"] if c["tag"] == "applies_operationally"
        ],
        "do_not_reopen_theory": True,
    }
    return out


def selftest() -> None:
    ch = packet_chart(5, 5)
    assert ch["cycle"] == 1 and ch["local"] == 0 and ch["typed_rechart"]
    trip = twin_seam_triple(1, 5)
    assert trip["prev_n"] == 4 and trip["next_n"] == 6
    trip7 = twin_seam_triple(6, 7)  # 42 seam → 41,43
    assert trip7["prev_n"] == 41 and trip7["next_n"] == 43
    assert nearest_multiple(2560, 306) in (2448, 2754)
    assert nearest_multiple(2560, 306, "down") == 2448
    assert nearest_multiple(2560, 306, "up") == 2754
    assert abs((61 / 306) - 0.19934640522875818) < 1e-15
    assert hybrid_306_58_bytes(306) == 61
    assert hybrid_306_58_bytes(307) == 61 + theory_bytes_5_8(1)
    assert hybrid_306_58_bytes(310) == 61 + theory_bytes_5_8(4)
    th = exact_density_theory()
    assert th["B_per_trit"]["fmt_306_485"] == 61 / 306
    assert th["delta_vs_0p2"]["bytes_saved_per_306_trits_vs_ceil5_8"] == 1
    d = pack_tensor(1, 306)
    assert d["decision"]["path"] == "frame"
    print("packet_seam_probe selftest OK")


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "selftest":
        selftest()
        return 0
    if argv[0] == "run":
        ckpt = DEFAULT_CKPT
        if "--ckpt" in argv:
            ckpt = Path(argv[argv.index("--ckpt") + 1])
        out = run(ckpt)
        OUT.write_text(json.dumps(out, indent=2, default=str))
        ds = out.get("density_signal") or {}
        print(json.dumps({
            "overall_verdict": out["overall_verdict"],
            "tag_counts": out["tag_counts"],
            "pattern_winner": out["pattern"].get("winner"),
            "recursion": out["recursion"].get("verdict"),
            "nesting_tag": out["nesting"].get("verdict_tag"),
            "bitnet": {
                "verdict": out["bitnet_306_redesign"].get("verdict"),
                "totals": out["bitnet_306_redesign"].get("totals"),
                "frame_path_after": out["bitnet_306_redesign"].get("frame_path_after"),
            },
            "density_signal": {
                "verdict": ds.get("verdict"),
                "confounder_split_nearest": ds.get("confounder_split_nearest"),
                "theory_B_per_trit": (ds.get("theory") or {}).get("B_per_trit"),
                "A_down_totals": (ds.get("A_redesign_down") or {}).get("totals"),
                "B_up_totals": (ds.get("B_redesign_up") or {}).get("totals"),
                "C_pad_totals": (ds.get("C_pad_up") or {}).get("totals"),
                "D_hybrid_totals": (ds.get("D_hybrid") or {}).get("totals"),
                "mode_proximity_stats": (ds.get("mode_proximity") or {}).get("stats"),
                "scale_theory": (ds.get("scale_projection") or {}).get("theory"),
                "scale_projections": [
                    {
                        "label": r["label"],
                        "saved_vs_0p2_MB": r["saved_vs_0p2_MB"],
                        "saved_vs_ceil_block_MB": r["saved_vs_ceil_block_MB"],
                        "saved_vs_0p2_pct": r["saved_vs_0p2_pct"],
                    }
                    for r in ((ds.get("scale_projection") or {}).get("projections") or [])
                ],
            },
            "elapsed_s": out["elapsed_s"],
            "wrote": str(OUT),
        }, indent=2, default=str))
        return 0
    print("usage: packet_seam_probe.py [selftest|run] [--ckpt PATH]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
