#!/usr/bin/env python3
"""Surviving packing stack — harden, measure, diagnose.

Pipeline (deterministic):
  reshape (deployability filter) → pick axis → pad fiber toward surplus → pack flat

Compared against flat fmt_5_8 / flat 486 / fiber-41 baselines.
When the stack loses, emit a failure analysis (do not silently accept).

Also: systematic associator map (nested vs flat tax) over phase triples.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from frame_formats import theory_bytes_486
from geometry_lab import (
    _aspect,
    associator,
    best_tax,
    divisors,
    fiber_pad_strategies,
    holonomy,
)
from harmonic_tax import frac_Q_alpha, tax_cols_harmonic, tax_rows_harmonic
from pack_ladder import theory_bytes_41_65, theory_bytes_5_8
from pad_to_tax0 import pad_toward_surplus_phase
from scale_probe import LLM_SHAPES, axis_bits, layout_bytes

getcontext().prec = 120

DEFAULT_CKPT = Path(__file__).resolve().parent / "data" / "bitnet" / "model.safetensors"


# ---------------------------------------------------------------------------
# Deployability filter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeployConstraints:
    max_aspect: float = 16.0
    align: int = 64  # both dims multiple of align (0 = off)
    min_dim: int = 32
    max_dim: int = 2_000_000

    def ok(self, m: int, n: int) -> bool:
        if m < self.min_dim or n < self.min_dim:
            return False
        if m > self.max_dim or n > self.max_dim:
            return False
        if _aspect(m, n) > self.max_aspect + 1e-12:
            return False
        if self.align > 0 and (m % self.align or n % self.align):
            return False
        return True


def ledger_bytes(m: int, n: int) -> Dict[str, int]:
    """Byte ledgers for shape (m,n). Includes pad-induced trit growth via callers."""
    mn = m * n
    ax = axis_bits(m, n)
    lay = layout_bytes(m, n)
    if lay["row_41_65"] <= lay["col_41_65"]:
        best_41 = lay["row_41_65"]
        axis_41 = "rows"
    else:
        best_41 = lay["col_41_65"]
        axis_41 = "cols"
    if lay["row_5_8"] <= lay["col_5_8"]:
        best_fiber_58 = lay["row_5_8"]
        axis_58 = "rows"
    else:
        best_fiber_58 = lay["col_5_8"]
        axis_58 = "cols"
    return {
        "trits": mn,
        "flat_5_8": lay["flat_5_8"],
        "flat_486": lay["flat_486"],
        "flat_41_65": lay["flat_41_65"],
        "best_fiber_41": best_41,
        "axis_41": axis_41,
        "best_fiber_5_8": best_fiber_58,
        "axis_5_8": axis_58,
        "axis_bits_saved": ax["bits_saved_vs_worse"],
        "best_tax": best_tax(m, n),
        "holonomy": holonomy(m, n),
    }


def reshape_under_constraints(
    m: int,
    n: int,
    cons: DeployConstraints,
    d_cap: int = 256,
    objective: str = "fiber41",
) -> Dict[str, Any]:
    """Best volume-preserving reshape under deployability constraints.

    objective:
      'tax'     — minimize best_tax (legacy; can worsen fiber-41 bytes)
      'fiber41' — minimize best_fiber_41 bytes (default after measurement)
      'lex'     — fiber41 first, then tax, then aspect

    Tie-break: smaller aspect, then closer to square.
    If no legal reshape passes, soft-relax align then report.
    """
    base_tax = best_tax(m, n)
    base_led = ledger_bytes(m, n)
    base_f41 = base_led["best_fiber_41"]
    candidates: List[Dict[str, Any]] = []

    def add(m2: int, n2: int, d: int, kind: str) -> None:
        if m2 <= 0 or n2 <= 0:
            return
        legal = cons.ok(m2, n2)
        led = ledger_bytes(m2, n2)
        t = led["best_tax"]
        f41 = led["best_fiber_41"]
        candidates.append(
            {
                "m": m2,
                "n": n2,
                "d": d,
                "kind": kind,
                "best_tax": t,
                "best_fiber_41": f41,
                "delta_tax": t - base_tax,
                "delta_fiber41": f41 - base_f41,
                "aspect": _aspect(m2, n2),
                "legal": legal,
            }
        )

    add(m, n, 1, "identity")
    for d in divisors(m, cap=d_cap):
        if d == 1:
            continue
        add(m // d, n * d, d, "div_m")
    for d in divisors(n, cap=d_cap):
        if d == 1:
            continue
        add(m * d, n // d, d, "div_n")

    legal = [c for c in candidates if c["legal"]]
    relaxed_align = False
    if not legal:
        soft = DeployConstraints(
            max_aspect=cons.max_aspect,
            align=0,
            min_dim=cons.min_dim,
            max_dim=cons.max_dim,
        )
        legal = [c for c in candidates if soft.ok(c["m"], c["n"])]
        for c in legal:
            c["legal"] = True
            c["align_relaxed"] = True
        relaxed_align = True

    if not legal:
        best = candidates[0]
        return {
            "baseline": {
                "m": m,
                "n": n,
                "best_tax": base_tax,
                "best_fiber_41": base_f41,
                "aspect": _aspect(m, n),
            },
            "chosen": best,
            "n_legal": 0,
            "n_candidates": len(candidates),
            "align_relaxed": False,
            "failed_constraints": True,
            "objective": objective,
        }

    def sort_key(c: Dict[str, Any]) -> Tuple:
        if objective == "tax":
            primary = (c["best_tax"],)
        elif objective == "fiber41":
            primary = (c["best_fiber_41"],)
        elif objective == "lex":
            primary = (c["best_fiber_41"], c["best_tax"])
        else:
            raise ValueError(f"unknown reshape objective {objective}")
        return primary + (
            c["aspect"],
            abs(math.log(max(c["m"], 1) / max(c["n"], 1))),
            c["d"],
        )

    legal.sort(key=sort_key)
    best = legal[0]
    return {
        "baseline": {
            "m": m,
            "n": n,
            "best_tax": base_tax,
            "best_fiber_41": base_f41,
            "aspect": _aspect(m, n),
        },
        "chosen": best,
        "n_legal": len(legal),
        "n_improved_tax": sum(1 for c in legal if c["delta_tax"] < 0),
        "n_improved_fiber41": sum(1 for c in legal if c["delta_fiber41"] < 0),
        "n_improved_legal": sum(1 for c in legal if c["delta_fiber41"] < 0),
        "n_candidates": len(candidates),
        "align_relaxed": relaxed_align,
        "failed_constraints": False,
        "top_legal": legal[:5],
        "objective": objective,
    }


def apply_stack(
    m: int,
    n: int,
    cons: DeployConstraints,
    max_pad: int = 64,
    allow_pad: bool = True,
    pad_gate: str = "fiber41",
    reshape_objective: str = "fiber41",
) -> Dict[str, Any]:
    """Deterministic stack on one rectangle.

    pad_gate:
      'always'   — always fiber-pad when allow_pad
      'fiber41'  — only keep pad if best_fiber_41 bytes improve vs post-reshape
      'flat58'   — only keep pad if flat_5_8 bytes improve vs post-reshape
      'never'    — synonym for allow_pad=False

    reshape_objective: 'tax' | 'fiber41' | 'lex' (default fiber41 after measurement)
    """
    if pad_gate == "never":
        allow_pad = False
    before = ledger_bytes(m, n)
    rs = reshape_under_constraints(m, n, cons, objective=reshape_objective)
    m1, n1 = rs["chosen"]["m"], rs["chosen"]["n"]
    mid = ledger_bytes(m1, n1)

    pad_info: Dict[str, Any] = {
        "applied": False,
        "kept": False,
        "pads": [0, 0],
        "total_pad_trits_added": 0,
        "pad_gate": pad_gate,
    }
    m2, n2 = m1, n1
    if allow_pad:
        fp = fiber_pad_strategies(m1, n1, max_pad=max_pad)
        m_p, n_p = fp["shapes"]["fiber"]
        after_pad = ledger_bytes(m_p, n_p)
        keep = False
        if pad_gate == "always":
            keep = True
        elif pad_gate == "fiber41":
            keep = after_pad["best_fiber_41"] < mid["best_fiber_41"]
        elif pad_gate == "flat58":
            keep = after_pad["flat_5_8"] < mid["flat_5_8"]
        else:
            raise ValueError(f"unknown pad_gate {pad_gate}")
        pad_info = {
            "applied": True,
            "kept": keep,
            "orient": fp["orient"],
            "fiber": fp["fiber"],
            "strategy_taxes": fp["taxes"],
            "strategy_pads": fp["pads"],
            "best_efficiency": fp["best_efficiency"],
            "pads": [
                (m_p - m1 if fp["orient"] == "cols" else 0),
                (n_p - n1 if fp["orient"] == "rows" else 0),
            ],
            "total_pad_trits_added": (m_p * n_p - m1 * n1) if keep else 0,
            "candidate_shape": [m_p, n_p],
            "candidate_fiber41": after_pad["best_fiber_41"],
            "post_reshape_fiber41": mid["best_fiber_41"],
            "pad_gate": pad_gate,
            "reject_reason": None
            if keep
            else "pad_failed_byte_gate",
        }
        if keep:
            m2, n2 = m_p, n_p

    after = ledger_bytes(m2, n2)

    def delta(key: str) -> int:
        return after[key] - before[key]

    deltas = {
        "delta_flat_5_8": delta("flat_5_8"),
        "delta_flat_486": delta("flat_486"),
        "delta_best_fiber_41": delta("best_fiber_41"),
        "delta_trits": after["trits"] - before["trits"],
        "delta_best_tax": after["best_tax"] - before["best_tax"],
        "delta_axis_bits_saved": after["axis_bits_saved"] - before["axis_bits_saved"],
        "delta_fiber41_vs_reshape_only": after["best_fiber_41"] - mid["best_fiber_41"],
    }

    reasons: List[str] = []
    if deltas["delta_flat_5_8"] >= 0 and deltas["delta_best_fiber_41"] >= 0:
        reasons.append("no_byte_win_vs_flat5_8_or_fiber41")
    if deltas["delta_flat_5_8"] > 0:
        reasons.append("worse_than_original_flat_5_8")
    if deltas["delta_trits"] > 0:
        reasons.append("pad_trit_growth")
    if pad_info.get("reject_reason"):
        reasons.append("pad_rejected_by_byte_gate")
    if rs["chosen"]["kind"] == "identity" and rs.get("n_improved_legal", 0) == 0:
        reasons.append("reshape_found_no_legal_improvement")
    if rs.get("align_relaxed"):
        reasons.append("align_constraint_relaxed")
    if rs.get("failed_constraints"):
        reasons.append("failed_all_deploy_constraints")
    if before["best_tax"] == 0:
        reasons.append("already_zero_tax")
    vs_flat58 = after["best_fiber_41"] - before["flat_5_8"]
    if vs_flat58 > 0:
        reasons.append("stack_fiber41_still_loses_to_original_flat_5_8")
    if mid["best_fiber_41"] < before["best_fiber_41"] and after["best_fiber_41"] >= before["best_fiber_41"]:
        reasons.append("reshape_helped_but_pad_undid_fiber41_win")

    return {
        "input": [m, n],
        "reshape": rs,
        "pad": pad_info,
        "output": [m2, n2],
        "before": before,
        "after_reshape": mid,
        "after": after,
        "deltas": deltas,
        "stack_fiber41_minus_orig_flat58": vs_flat58,
        "diagnosis": reasons,
        "constraints": asdict(cons),
    }


# ---------------------------------------------------------------------------
# Associator quantification
# ---------------------------------------------------------------------------

def phase_bin(phi: Decimal, edges: Sequence[float] = (0.1, 0.4, 0.6, 0.9)) -> str:
    f = float(phi)
    if f < edges[0]:
        return "deficit"
    if f < edges[1]:
        return "low_mid"
    if f < edges[2]:
        return "near_half"
    if f < edges[3]:
        return "high_mid"
    return "surplus"


def run_associator_map(
    mode_pool: Sequence[int],
    max_triples: int = 200,
) -> Dict[str, Any]:
    """Map nested−flat tax by phase-bin triple of (m,n,p)."""
    pool = sorted(set(int(x) for x in mode_pool if x > 0))
    rows = []
    by_bins: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(pool):
        for j, n in enumerate(pool):
            if j == i:
                continue
            for k, p in enumerate(pool):
                if k == i or k == j:
                    continue
                if m * n > 2_000_000 or n * p > 2_000_000:
                    continue
                a = associator(m, n, p)
                bm = phase_bin(frac_Q_alpha(m))
                bn = phase_bin(frac_Q_alpha(n))
                bp = phase_bin(frac_Q_alpha(p))
                key = f"{bm}|{bn}|{bp}"
                by_bins[key].append(a["associator"])
                rows.append(
                    {
                        **a,
                        "bins": key,
                        "phi_m": format(frac_Q_alpha(m), "f"),
                        "phi_n": format(frac_Q_alpha(n), "f"),
                        "phi_p": format(frac_Q_alpha(p), "f"),
                    }
                )
                if len(rows) >= max_triples:
                    break
            if len(rows) >= max_triples:
                break
        if len(rows) >= max_triples:
            break

    summary_bins = []
    for key, vals in sorted(by_bins.items(), key=lambda kv: -len(kv[1])):
        summary_bins.append(
            {
                "bins": key,
                "n": len(vals),
                "mean_associator": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "frac_nested_worse": sum(1 for v in vals if v > 0) / len(vals),
                "frac_flat_worse": sum(1 for v in vals if v < 0) / len(vals),
                "frac_tie": sum(1 for v in vals if v == 0) / len(vals),
            }
        )
    assoc_vals = [r["associator"] for r in rows]
    # When is nesting acceptable? look for bins with frac_flat_worse high or mean<=0
    nesting_ok = [b for b in summary_bins if b["frac_nested_worse"] < 0.5]
    return {
        "n_triples": len(rows),
        "mean_associator": sum(assoc_vals) / len(assoc_vals) if assoc_vals else 0,
        "frac_nested_worse": sum(1 for v in assoc_vals if v > 0) / len(assoc_vals)
        if assoc_vals
        else 0,
        "frac_tie": sum(1 for v in assoc_vals if v == 0) / len(assoc_vals)
        if assoc_vals
        else 0,
        "by_phase_bins": summary_bins,
        "nesting_acceptable_bins": nesting_ok,
        "rule": (
            "prefer_flat_almost_always"
            if (not nesting_ok)
            or all(b["n"] < 3 for b in nesting_ok)
            else "nesting_ok_in_some_phase_bins"
        ),
        "samples_head": rows[:15],
    }


# ---------------------------------------------------------------------------
# Suite + BitNet measurement
# ---------------------------------------------------------------------------

def load_bitnet_shapes(path: Path) -> List[Tuple[str, int, int]]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode())
    out = []
    for name in sorted(hdr):
        if name == "__metadata__":
            continue
        meta = hdr[name]
        if meta.get("dtype") != "U8" or not name.endswith(".weight"):
            continue
        shape = meta["shape"]
        m, n = int(shape[0]) * 4, int(shape[1])
        out.append((name, m, n))
    return out


def aggregate_stack_runs(runs: Sequence[Dict[str, Any]], label: str) -> Dict[str, Any]:
    n = len(runs)
    if n == 0:
        return {"label": label, "n": 0}

    def sum_key(path: Sequence[str]) -> int:
        total = 0
        for r in runs:
            cur: Any = r
            for p in path:
                cur = cur[p]
            total += int(cur)
        return total

    before_flat58 = sum_key(("before", "flat_5_8"))
    after_flat58 = sum_key(("after", "flat_5_8"))
    before_f41 = sum_key(("before", "best_fiber_41"))
    after_f41 = sum_key(("after", "best_fiber_41"))
    before_tax = sum_key(("before", "best_tax"))
    after_tax = sum_key(("after", "best_tax"))
    trit_delta = sum_key(("deltas", "delta_trits"))

    diag_counts: Dict[str, int] = defaultdict(int)
    for r in runs:
        for d in r["diagnosis"]:
            diag_counts[d] += 1

    n_byte_win_58 = sum(1 for r in runs if r["deltas"]["delta_flat_5_8"] < 0)
    n_byte_win_41 = sum(1 for r in runs if r["deltas"]["delta_best_fiber_41"] < 0)
    n_tax_win = sum(1 for r in runs if r["deltas"]["delta_best_tax"] < 0)
    n_reshape = sum(1 for r in runs if r["reshape"]["chosen"]["kind"] != "identity")
    n_pad_kept = sum(1 for r in runs if r["pad"].get("kept"))
    n_pad_rejected = sum(
        1 for r in runs if r["pad"].get("applied") and not r["pad"].get("kept")
    )
    n_still_lose_to_58 = sum(
        1 for r in runs if r["stack_fiber41_minus_orig_flat58"] > 0
    )

    # Failure analysis narrative hooks
    analysis: List[str] = []
    if after_flat58 >= before_flat58:
        analysis.append(
            "Stack does not beat original flat fmt_5_8 in aggregate bytes "
            f"(Δ={after_flat58 - before_flat58})."
        )
    if after_f41 < before_f41 and after_flat58 >= before_flat58:
        analysis.append(
            "Stack improves fiber-41 family but still loses to flat 5_8 — "
            "same Pareto knee as prior probes; geometry helps the wrong ledger."
        )
    if trit_delta > 0:
        analysis.append(
            f"Fiber pad added {trit_delta} trits total; pad growth can erase tax wins "
            "on the flat-5_8 ledger (~1.6 bit/trit)."
        )
    if n_still_lose_to_58 == n:
        analysis.append(
            "Every tensor's post-stack fiber-41 still exceeds original flat 5_8 — "
            "stack cannot displace 5_8 without a denser block format on the new shape."
        )
    if n_reshape == 0:
        analysis.append(
            "No tensor reshaped under constraints — deployability filter or shape "
            "set left nothing movable (check align/aspect)."
        )
    if diag_counts.get("reshape_found_no_legal_improvement", 0) > n // 2:
        analysis.append(
            "Majority had no legal reshape improvement — BitNet dims may already "
            "be near aspect bounds or align grid blocks surplus-seeking reshapes."
        )

    return {
        "label": label,
        "n": n,
        "bytes": {
            "before_flat_5_8": before_flat58,
            "after_flat_5_8": after_flat58,
            "delta_flat_5_8": after_flat58 - before_flat58,
            "before_best_fiber_41": before_f41,
            "after_best_fiber_41": after_f41,
            "delta_best_fiber_41": after_f41 - before_f41,
            "before_flat_486": sum_key(("before", "flat_486")),
            "after_flat_486": sum_key(("after", "flat_486")),
        },
        "tax": {
            "before_sum_best_tax": before_tax,
            "after_sum_best_tax": after_tax,
            "delta": after_tax - before_tax,
        },
        "trits_delta": trit_delta,
        "counts": {
            "n_byte_win_flat58": n_byte_win_58,
            "n_byte_win_fiber41": n_byte_win_41,
            "n_tax_win": n_tax_win,
            "n_reshaped": n_reshape,
            "n_pad_kept": n_pad_kept,
            "n_pad_rejected": n_pad_rejected,
            "n_stack_fiber41_loses_to_orig_flat58": n_still_lose_to_58,
        },
        "diagnosis_counts": dict(diag_counts),
        "failure_analysis": analysis,
        "verdict": (
            "stack_beats_flat_5_8"
            if after_flat58 < before_flat58
            else "stack_helps_fiber41_only"
            if after_f41 < before_f41
            else "stack_no_aggregate_byte_win"
        ),
    }


def run_suite_shapes(
    shapes: Sequence[Tuple[str, int, int]],
    cons: DeployConstraints,
    max_pad: int = 64,
    pad_gate: str = "fiber41",
    reshape_objective: str = "fiber41",
) -> Dict[str, Any]:
    runs = []
    for name, m, n in shapes:
        r = apply_stack(
            m,
            n,
            cons,
            max_pad=max_pad,
            pad_gate=pad_gate,
            reshape_objective=reshape_objective,
        )
        r["name"] = name
        runs.append(r)
    return {
        "aggregate": aggregate_stack_runs(runs, "suite"),
        "per_tensor": runs,
        "reshape_objective": reshape_objective,
    }


def run_bitnet(
    path: Path,
    cons: DeployConstraints,
    max_pad: int = 64,
    max_tensors: Optional[int] = None,
    pad_gate: str = "fiber41",
    reshape_objective: str = "fiber41",
) -> Dict[str, Any]:
    shapes = load_bitnet_shapes(path)
    if max_tensors is not None:
        shapes = shapes[:max_tensors]
    uniq: Dict[Tuple[int, int], int] = defaultdict(int)
    for _, m, n in shapes:
        uniq[(m, n)] += 1
    uniq_runs = []
    for (m, n), cnt in sorted(uniq.items()):
        r = apply_stack(
            m,
            n,
            cons,
            max_pad=max_pad,
            pad_gate=pad_gate,
            reshape_objective=reshape_objective,
        )
        r["name"] = f"unique_{m}x{n}"
        r["n_tensors"] = cnt
        uniq_runs.append(r)

    expanded = []
    for r in uniq_runs:
        for _ in range(r["n_tensors"]):
            expanded.append(r)

    return {
        "path": str(path),
        "n_tensors": len(shapes),
        "unique_shapes": [
            {"shape": [m, n], "count": c} for (m, n), c in sorted(uniq.items())
        ],
        "per_unique": uniq_runs,
        "aggregate": aggregate_stack_runs(expanded, "bitnet_weighted"),
        "pad_gate": pad_gate,
        "reshape_objective": reshape_objective,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def selftest() -> None:
    cons = DeployConstraints(max_aspect=16, align=64, min_dim=32)
    r = apply_stack(2560, 2560, cons, max_pad=32, pad_gate="fiber41", reshape_objective="fiber41")
    assert "deltas" in r
    r2 = apply_stack(2560, 2560, cons, max_pad=32, pad_gate="never", reshape_objective="fiber41")
    assert r2["deltas"]["delta_trits"] == 0
    r_tax = apply_stack(2560, 6912, cons, max_pad=0, pad_gate="never", reshape_objective="tax")
    r_f41 = apply_stack(2560, 6912, cons, max_pad=0, pad_gate="never", reshape_objective="fiber41")
    # fiber41 objective should not worsen fiber41 vs identity more than tax obj
    assert r_f41["deltas"]["delta_best_fiber_41"] <= r_tax["deltas"]["delta_best_fiber_41"]
    a = run_associator_map([5, 8, 41, 53, 128, 256, 640, 2560], max_triples=30)
    assert a["n_triples"] > 0
    print("packing_stack selftest OK")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["selftest", "run", "associator", "bitnet", "all"])
    p.add_argument("--max-pad", type=int, default=64)
    p.add_argument("--max-aspect", type=float, default=16.0)
    p.add_argument("--align", type=int, default=64)
    p.add_argument("--min-dim", type=int, default=32)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--max-tensors", type=int, default=None)
    p.add_argument("--skip-ckpt", action="store_true")
    p.add_argument("--pad-gate", choices=["always", "fiber41", "flat58", "never"], default="fiber41")
    p.add_argument(
        "--reshape-objective",
        choices=["tax", "fiber41", "lex"],
        default="fiber41",
    )
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.cmd == "selftest":
        selftest()
        return 0

    cons = DeployConstraints(
        max_aspect=args.max_aspect, align=args.align, min_dim=args.min_dim
    )
    out: Dict[str, Any] = {
        "stance": (
            "Harden surviving stack; measure bytes; diagnose losses. "
            "Default reshape_objective=fiber41; pad_gate=fiber41. "
            "Tax wins that grow trits or lose to flat 5_8 are not practical wins."
        ),
        "constraints": asdict(cons),
        "max_pad": args.max_pad,
        "pad_gate": args.pad_gate,
        "reshape_objective": args.reshape_objective,
    }

    if args.cmd in ("associator", "all"):
        pool = sorted(
            {m for _, m, n in LLM_SHAPES}
            | {n for _, m, n in LLM_SHAPES}
            | {5, 8, 19, 41, 53, 60, 101, 306, 640, 665, 2560, 6912}
        )
        out["associator"] = run_associator_map(pool, max_triples=250)

    if args.cmd in ("run", "all"):
        out["llm_suite"] = run_suite_shapes(
            LLM_SHAPES,
            cons,
            max_pad=args.max_pad,
            pad_gate=args.pad_gate,
            reshape_objective=args.reshape_objective,
        )
        # Compare tax-minimizing reshape (legacy) vs fiber41
        out["llm_suite_obj_tax"] = run_suite_shapes(
            LLM_SHAPES,
            cons,
            max_pad=args.max_pad,
            pad_gate="never",
            reshape_objective="tax",
        )
        out["llm_suite_obj_fiber41"] = run_suite_shapes(
            LLM_SHAPES,
            cons,
            max_pad=args.max_pad,
            pad_gate="never",
            reshape_objective="fiber41",
        )

    if args.cmd in ("bitnet", "all") and not args.skip_ckpt:
        if not args.ckpt.is_file():
            out["bitnet"] = {"error": f"missing checkpoint {args.ckpt}"}
        else:
            t0 = time.time()
            out["bitnet"] = run_bitnet(
                args.ckpt,
                cons,
                max_pad=args.max_pad,
                max_tensors=args.max_tensors,
                pad_gate=args.pad_gate,
                reshape_objective=args.reshape_objective,
            )
            out["bitnet"]["elapsed_s"] = time.time() - t0
            out["bitnet_obj_tax"] = run_bitnet(
                args.ckpt,
                cons,
                max_pad=args.max_pad,
                max_tensors=args.max_tensors,
                pad_gate="never",
                reshape_objective="tax",
            )
            out["bitnet_obj_fiber41"] = run_bitnet(
                args.ckpt,
                cons,
                max_pad=args.max_pad,
                max_tensors=args.max_tensors,
                pad_gate="never",
                reshape_objective="fiber41",
            )

    if args.cmd in ("run", "all"):
        suite_nopad = run_suite_shapes(
            LLM_SHAPES,
            cons,
            max_pad=args.max_pad,
            pad_gate="never",
            reshape_objective=args.reshape_objective,
        )
        out["llm_suite_no_pad"] = suite_nopad
        cons_align0 = DeployConstraints(
            max_aspect=args.max_aspect, align=0, min_dim=args.min_dim
        )
        out["llm_suite_align0"] = run_suite_shapes(
            LLM_SHAPES,
            cons_align0,
            max_pad=args.max_pad,
            pad_gate=args.pad_gate,
            reshape_objective=args.reshape_objective,
        )
        out["llm_suite_pad_always"] = run_suite_shapes(
            LLM_SHAPES,
            cons,
            max_pad=args.max_pad,
            pad_gate="always",
            reshape_objective=args.reshape_objective,
        )

    path = Path(__file__).resolve().parent / "packing_stack_results.json"
    summary: Dict[str, Any] = {
        "constraints": out["constraints"],
        "reshape_objective": out.get("reshape_objective"),
        "associator_rule": out.get("associator", {}).get("rule"),
        "associator_frac_nested_worse": out.get("associator", {}).get(
            "frac_nested_worse"
        ),
    }
    if "llm_suite" in out:
        summary["llm_suite"] = out["llm_suite"]["aggregate"]
        summary["llm_suite_no_pad"] = out["llm_suite_no_pad"]["aggregate"]
        summary["llm_suite_pad_always"] = out["llm_suite_pad_always"]["aggregate"]
        summary["llm_obj_tax_Δ41"] = out["llm_suite_obj_tax"]["aggregate"]["bytes"][
            "delta_best_fiber_41"
        ]
        summary["llm_obj_fiber41_Δ41"] = out["llm_suite_obj_fiber41"]["aggregate"][
            "bytes"
        ]["delta_best_fiber_41"]
    if "bitnet" in out and "aggregate" in out.get("bitnet", {}):
        summary["bitnet"] = out["bitnet"]["aggregate"]
        summary["bitnet_unique"] = [
            {
                "name": r["name"],
                "out": r["output"],
                "deltas": r["deltas"],
                "pad_kept": r["pad"].get("kept"),
                "n": r.get("n_tensors"),
            }
            for r in out["bitnet"]["per_unique"]
        ]
    if "bitnet_obj_tax" in out:
        summary["bitnet_obj_tax_Δ41"] = out["bitnet_obj_tax"]["aggregate"]["bytes"][
            "delta_best_fiber_41"
        ]
        summary["bitnet_obj_fiber41_Δ41"] = out["bitnet_obj_fiber41"]["aggregate"][
            "bytes"
        ]["delta_best_fiber_41"]
        summary["bitnet_obj_compare_unique"] = [
            {
                "name": a["name"],
                "tax_out": a["output"],
                "tax_Δ41": a["deltas"]["delta_best_fiber_41"],
                "f41_out": b["output"],
                "f41_Δ41": b["deltas"]["delta_best_fiber_41"],
            }
            for a, b in zip(
                out["bitnet_obj_tax"]["per_unique"],
                out["bitnet_obj_fiber41"]["per_unique"],
            )
        ]

    print(json.dumps(summary, indent=2))
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
