#!/usr/bin/env python3
"""Deterministic three-ledger packer — use the geometry, measure bytes.

Decision tree (practical focus):

  if a mode/stream length is an exact frame quantum (306, 665, …):
      use rung-frame ledger   # even frame lengths: frame operator only
  else if reshape (aspect ≤ 16) improves fiber-41 bytes:
      reshape → best-axis fiber-41
  else:
      flat fmt_5_8

Odd/even split (codified):
  - Odd rungs (5,19,41,53,665,…) may use sheet/complement tools at design time.
  - Even frame lengths (esp. 306) → frame ledger only; never force typed sheets.

Design-time surplus×align list is documented, never used as post-hoc pad.

Also: controlled nesting test on 306/665 multiples (bytes + decode timing).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from frame_formats import (
    PARTS_486,
    PARTS_665,
    pack_frame,
    theory_bytes_486,
    theory_bytes_665,
    unpack_frame,
)
from hierarchical_digits import (
    unpack_306_485_hierarchical,
)
from pack_ladder import (
    pack_306_485,
    pack_41_65,
    pack_5_8,
    theory_bytes_306_485,
    theory_bytes_41_65,
    theory_bytes_5_8,
    unpack_306_485,
    unpack_41_65,
    unpack_5_8,
)
from packing_stack import (
    DeployConstraints,
    _aspect,
    ledger_bytes,
    load_bitnet_shapes,
    reshape_under_constraints,
)
from tax_graph import LAW_B_486, LAW_B_665

DEFAULT_CKPT = Path(__file__).resolve().parent / "data" / "bitnet" / "model.safetensors"

# Frame quanta (even 306 is frame-only; 665 is odd but also a Law B frame)
# Rung-block / frame quanta.
# Even 306 → fmt_306_485 (certified rung container) is the density ledger;
# 486-frame is chiral assembly (tax 1) — usually denser-decode, not denser-bytes.
FRAME_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "fmt_306_485",
        "quantum": 306,
        "parts": None,
        "parity": "even",
        "family": "frame_only",
        "bytes_fn": theory_bytes_306_485,
        "role": "rung_block_density",
    },
    {
        "name": "fmt_486_frame",
        "quantum": 306,
        "parts": LAW_B_486,
        "parity": "even",
        "family": "frame_only",
        "bytes_fn": theory_bytes_486,
        "role": "chiral_assembly",
    },
    {
        "name": "fmt_665_frame",
        "quantum": 665,
        "parts": LAW_B_665,
        "parity": "odd",
        "family": "frame_or_sheet",
        "bytes_fn": theory_bytes_665,
        "role": "chiral_assembly_tax0",
    },
)

ODD_RUNGS = (5, 19, 41, 53, 665, 15601)
EVEN_FRAME_LENGTHS = (306,)  # never search typed sheets for these

# Design-time only (architecture_prior); never pad into these post-hoc
DESIGN_TIME_SURPLUS_ALIGN64 = (576, 1600, 2624, 3648)


@dataclass(frozen=True)
class PackDecision:
    ledger: str
    bytes: int
    shape: Tuple[int, int]
    reason: str
    candidates: Dict[str, int]


def frame_bytes_for_length(n: int) -> List[Dict[str, Any]]:
    """All frame/rung-block ledgers whose quantum divides n."""
    hits = []
    for spec in FRAME_SPECS:
        q = spec["quantum"]
        if n > 0 and n % q == 0:
            hits.append(
                {
                    "ledger": spec["name"],
                    "bytes": int(spec["bytes_fn"](n)),
                    "quantum": q,
                    "role": spec["role"],
                    "parity_rule": (
                        "frame_only_even_quantum"
                        if q in EVEN_FRAME_LENGTHS
                        else "frame_ok_odd_quantum"
                    ),
                }
            )
    hits.sort(key=lambda h: h["bytes"])
    return hits


def evaluate_frame_options(m: int, n: int) -> Dict[str, Any]:
    """Frame/rung-block eligibility on flat MN, row fibers, col fibers."""
    options = []
    mn = m * n
    for label, length, mult in (
        ("flat_stream", mn, 1),
        ("row_fibers", n, m),
        ("col_fibers", m, n),
    ):
        for hit in frame_bytes_for_length(length):
            options.append(
                {
                    "layout": label,
                    "ledger": hit["ledger"],
                    "fiber_len": length,
                    "n_fibers": mult,
                    "quantum": hit["quantum"],
                    "role": hit["role"],
                    "bytes": hit["bytes"] * mult,
                    "parity_rule": hit["parity_rule"],
                }
            )
    options.sort(key=lambda o: o["bytes"])
    return {
        "eligible": len(options) > 0,
        "best": options[0] if options else None,
        "all": options[:12],
    }


def evaluate_hybrid_306(m: int, n: int) -> Dict[str, Any]:
    """306-prefix + rem (theory_bytes_306_485) on flat/row/col — no exact-÷306 gate.

    Captures the known 61/306 density when floor(length/306)≥1. Rem uses
    bigint container (same as fmt_306_485 remainder path).
    """
    options = []
    mn = m * n
    for label, length, mult in (
        ("flat_stream", mn, 1),
        ("row_fibers", n, m),
        ("col_fibers", m, n),
    ):
        if length < 306:
            continue
        options.append(
            {
                "layout": label,
                "ledger": "hybrid_306_485",
                "fiber_len": length,
                "n_fibers": mult,
                "quantum": 306,
                "full_blocks": length // 306,
                "rem": length % 306,
                "bytes": theory_bytes_306_485(length) * mult,
                "role": "rung_block_hybrid_prefix",
            }
        )
    options.sort(key=lambda o: o["bytes"])
    return {
        "eligible": len(options) > 0,
        "best": options[0] if options else None,
        "all": options[:12],
    }


def pack_tensor(
    m: int,
    n: int,
    cons: Optional[DeployConstraints] = None,
    allow_hybrid: bool = False,
) -> Dict[str, Any]:
    """Single deterministic three-ledger decision for shape (m, n).

    allow_hybrid: if True, also consider theory_bytes_306_485 on best layout
    without requiring length % 306 == 0 (captures ~0.33% on BitNet-class).
    """
    if cons is None:
        cons = DeployConstraints(max_aspect=16.0, align=64, min_dim=32)

    base = ledger_bytes(m, n)
    flat58 = base["flat_5_8"]

    # --- Branch A: frame if eligible (exact quantum) ---
    frames = evaluate_frame_options(m, n)
    hybrid = evaluate_hybrid_306(m, n) if allow_hybrid else {
        "eligible": False,
        "best": None,
        "all": [],
    }

    # --- Branch B: reshape + fiber-41 ---
    rs = reshape_under_constraints(m, n, cons, objective="fiber41")
    m2, n2 = rs["chosen"]["m"], rs["chosen"]["n"]
    after_rs = ledger_bytes(m2, n2)
    fiber41_bytes = after_rs["best_fiber_41"]
    reshape_helped = fiber41_bytes < base["best_fiber_41"]
    aspect_ok = _aspect(m2, n2) <= cons.max_aspect + 1e-12

    candidates = {
        "flat_5_8": flat58,
        "fiber41_original": base["best_fiber_41"],
        "fiber41_reshaped": fiber41_bytes,
    }
    if frames["best"]:
        candidates["frame_best"] = frames["best"]["bytes"]
    if hybrid["best"]:
        candidates["hybrid_306"] = hybrid["best"]["bytes"]

    # Pick best among frame / hybrid that beat flat (prefer denser)
    special_best = None
    special_path = None
    special_reason = None
    for kind, bundle in (("frame", frames), ("hybrid", hybrid)):
        if not bundle.get("best"):
            continue
        fb = bundle["best"]["bytes"]
        if fb < flat58 and (special_best is None or fb < special_best):
            special_best = fb
            special_path = kind
            if kind == "frame":
                special_reason = (
                    f"frame:{bundle['best']['layout']}"
                    f":Q={bundle['best']['quantum']}"
                    f":{bundle['best'].get('parity_rule', '')}"
                )
            else:
                special_reason = (
                    f"hybrid:{bundle['best']['layout']}"
                    f":blocks={bundle['best']['full_blocks']}"
                    f":rem={bundle['best']['rem']}"
                )

    # Decision tree
    if special_best is not None:
        decision = PackDecision(
            ledger=(
                frames["best"]["ledger"]
                if special_path == "frame"
                else "hybrid_306_485"
            ),
            bytes=special_best,
            shape=(m, n),
            reason=special_reason or special_path,
            candidates=candidates,
        )
        path = special_path
    elif aspect_ok and fiber41_bytes < flat58:
        decision = PackDecision(
            ledger="fiber_41_65",
            bytes=fiber41_bytes,
            shape=(m2, n2),
            reason=(
                "reshape+fiber41"
                if rs["chosen"]["kind"] != "identity"
                else "axis_fiber41"
            ),
            candidates=candidates,
        )
        path = "fiber41"
    else:
        decision = PackDecision(
            ledger="fmt_5_8",
            bytes=flat58,
            shape=(m, n),
            reason="flat58_pareto_knee",
            candidates=candidates,
        )
        path = "flat58"

    oracle = min(candidates.values())
    return {
        "input": [m, n],
        "decision": {
            "ledger": decision.ledger,
            "bytes": decision.bytes,
            "shape": list(decision.shape),
            "reason": decision.reason,
            "path": path,
        },
        "candidates": candidates,
        "oracle_min_bytes": oracle,
        "selected_minus_oracle": decision.bytes - oracle,
        "reshape": rs,
        "frames": frames,
        "hybrid": hybrid,
        "allow_hybrid": allow_hybrid,
        "odd_even": {
            "odd_rungs_sheet_ok": list(ODD_RUNGS),
            "even_frame_only": list(EVEN_FRAME_LENGTHS),
            "design_time_surplus_align64": list(DESIGN_TIME_SURPLUS_ALIGN64),
            "note": "never post-hoc pad into design-time list",
        },
    }


# ---------------------------------------------------------------------------
# BitNet measurement
# ---------------------------------------------------------------------------

def measure_bitnet(
    path: Path,
    max_tensors: Optional[int] = None,
    allow_hybrid: bool = False,
) -> Dict[str, Any]:
    shapes = load_bitnet_shapes(path)
    if max_tensors is not None:
        shapes = shapes[:max_tensors]

    uniq: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for name, m, n in shapes:
        uniq[(m, n)].append(name)

    cons = DeployConstraints(max_aspect=16.0, align=64, min_dim=32)
    per_unique = []
    totals = {
        "n_tensors": 0,
        "trits": 0,
        "bytes_selected": 0,
        "bytes_flat58": 0,
        "bytes_fiber41_orig": 0,
        "bytes_oracle": 0,
        "bytes_shipped_u8": 0,  # 2-bit
    }
    path_counts: Dict[str, int] = defaultdict(int)

    for (m, n), names in sorted(uniq.items()):
        r = pack_tensor(m, n, cons, allow_hybrid=allow_hybrid)
        cnt = len(names)
        flat58 = r["candidates"]["flat_5_8"]
        f41 = r["candidates"]["fiber41_original"]
        sel = r["decision"]["bytes"]
        oracle = r["oracle_min_bytes"]
        trits = m * n
        per_unique.append(
            {
                "shape": [m, n],
                "n_tensors": cnt,
                "decision": r["decision"],
                "candidates": r["candidates"],
                "oracle_min_bytes": oracle,
                "frames_eligible": r["frames"]["eligible"],
                "weighted_delta_vs_flat58": (sel - flat58) * cnt,
                "weighted_delta_vs_fiber41": (sel - f41) * cnt,
            }
        )
        totals["n_tensors"] += cnt
        totals["trits"] += trits * cnt
        totals["bytes_selected"] += sel * cnt
        totals["bytes_flat58"] += flat58 * cnt
        totals["bytes_fiber41_orig"] += f41 * cnt
        totals["bytes_oracle"] += oracle * cnt
        totals["bytes_shipped_u8"] += ((trits + 3) // 4) * cnt
        path_counts[r["decision"]["path"]] += cnt

    return {
        "path": str(path),
        "n_unique_shapes": len(uniq),
        "per_unique": per_unique,
        "totals": totals,
        "path_counts": dict(path_counts),
        "delta_selected_vs_flat58": totals["bytes_selected"] - totals["bytes_flat58"],
        "delta_selected_vs_fiber41": totals["bytes_selected"] - totals["bytes_fiber41_orig"],
        "delta_selected_vs_oracle": totals["bytes_selected"] - totals["bytes_oracle"],
        "delta_selected_vs_u8": totals["bytes_selected"] - totals["bytes_shipped_u8"],
        "verdict": (
            "beats_flat58"
            if totals["bytes_selected"] < totals["bytes_flat58"]
            else "matches_flat58_knee"
            if totals["bytes_selected"] == totals["bytes_flat58"]
            else "worse_than_flat58"
        ),
        "failure_analysis": _analyze_bitnet(totals, path_counts, per_unique),
    }


def _analyze_bitnet(
    totals: Dict[str, int],
    path_counts: Dict[str, int],
    per_unique: List[Dict[str, Any]],
) -> List[str]:
    lines = []
    d58 = totals["bytes_selected"] - totals["bytes_flat58"]
    if d58 >= 0:
        lines.append(
            f"Selected ledger does not beat flat fmt_5_8 (Δ={d58} B). "
            "Same Pareto knee as prior probes on unstructured BitNet."
        )
    if path_counts.get("frame", 0) == 0 and path_counts.get("hybrid", 0) == 0:
        lines.append(
            "No tensor used frame/hybrid ledger — BitNet lengths are not "
            "exact multiples of 306/665 (enable --hybrid for prefix path)."
        )
    if path_counts.get("hybrid", 0):
        lines.append(
            f"Hybrid 306-prefix used on {path_counts['hybrid']} tensors "
            "(known 61/306 density; not a new geometric effect)."
        )
    if path_counts.get("fiber41", 0) and d58 >= 0:
        lines.append(
            "Some tensors prefer fiber-41 after reshape, but in aggregate "
            "flat 5_8 still wins — fiber savings do not overcome 5_8 density."
        )
    for u in per_unique:
        if u["frames_eligible"]:
            lines.append(
                f"Shape {u['shape']} was frame-eligible; decision="
                f"{u['decision']['path']} reason={u['decision']['reason']}"
            )
    if not lines:
        lines.append("No anomalies; decision tree behaved as designed.")
    return lines


# ---------------------------------------------------------------------------
# Controlled nesting test
# ---------------------------------------------------------------------------

def nesting_test(
    lengths: Sequence[int] = (306, 612, 1224, 665, 1330),
    repeats: int = 40,
) -> Dict[str, Any]:
    """Flat vs one-level frame/nested decode on frame-aligned lengths.

    Measures exact theory bytes and wall-clock unpack for:
      - fmt_5_8 flat
      - fmt_41_65 (fiber-style blocks)
      - fmt_306_485 nested Law-C digit vs flat digit (on 306 multiples)
      - 486-frame / 665-frame chiral pack
    """
    rng = np.random.default_rng(42)
    rows = []
    for n in lengths:
        w = rng.integers(-1, 2, size=n, dtype=np.int8)
        row: Dict[str, Any] = {"n": n, "bytes": {}, "decode_us": {}, "rt_ok": {}}

        # 5_8
        b58 = pack_5_8(w)
        row["bytes"]["fmt_5_8"] = len(b58)
        row["bytes"]["th_5_8"] = theory_bytes_5_8(n)
        t0 = time.perf_counter()
        for _ in range(repeats):
            out = unpack_5_8(b58, n)
        row["decode_us"]["fmt_5_8"] = (time.perf_counter() - t0) * 1e6 / repeats
        row["rt_ok"]["fmt_5_8"] = bool(np.array_equal(out, w))

        # 41_65
        b41 = pack_41_65(w)
        row["bytes"]["fmt_41_65"] = len(b41)
        row["bytes"]["th_41_65"] = theory_bytes_41_65(n)
        t0 = time.perf_counter()
        for _ in range(repeats):
            out = unpack_41_65(b41, n)
        row["decode_us"]["fmt_41_65"] = (time.perf_counter() - t0) * 1e6 / repeats
        row["rt_ok"]["fmt_41_65"] = bool(np.array_equal(out, w))

        # 306 path
        if n % 306 == 0:
            b306 = pack_306_485(w)
            row["bytes"]["fmt_306_485"] = len(b306)
            t0 = time.perf_counter()
            for _ in range(repeats):
                out_flat = unpack_306_485_hierarchical(b306, n, mode="flat")
            row["decode_us"]["306_flat_digit"] = (
                (time.perf_counter() - t0) * 1e6 / repeats
            )
            t0 = time.perf_counter()
            for _ in range(repeats):
                out_nest = unpack_306_485_hierarchical(b306, n, mode="nested")
            row["decode_us"]["306_nested_LawC"] = (
                (time.perf_counter() - t0) * 1e6 / repeats
            )
            row["rt_ok"]["306_flat"] = bool(np.array_equal(out_flat, w))
            row["rt_ok"]["306_nested"] = bool(np.array_equal(out_nest, w))

            bf = pack_frame(w, PARTS_486)
            row["bytes"]["fmt_486_frame"] = len(bf)
            row["bytes"]["th_486_frame"] = theory_bytes_486(n)
            t0 = time.perf_counter()
            for _ in range(repeats):
                out_f = unpack_frame(bf, n, PARTS_486)
            row["decode_us"]["486_frame"] = (time.perf_counter() - t0) * 1e6 / repeats
            row["rt_ok"]["486_frame"] = bool(np.array_equal(out_f, w))

        if n % 665 == 0:
            bf = pack_frame(w, PARTS_665)
            row["bytes"]["fmt_665_frame"] = len(bf)
            row["bytes"]["th_665_frame"] = theory_bytes_665(n)
            t0 = time.perf_counter()
            for _ in range(repeats):
                out_f = unpack_frame(bf, n, PARTS_665)
            row["decode_us"]["665_frame"] = (time.perf_counter() - t0) * 1e6 / repeats
            row["rt_ok"]["665_frame"] = bool(np.array_equal(out_f, w))

        # Joint objective for nesting recommendation
        nest_verdict = "n_a"
        if n % 306 == 0:
            b_rung = row["bytes"]["fmt_306_485"]
            b_frame = row["bytes"]["fmt_486_frame"]
            b_flat = row["bytes"]["fmt_5_8"]
            t_nest = row["decode_us"]["306_nested_LawC"]
            t_flat_digit = row["decode_us"]["306_flat_digit"]
            t_58 = row["decode_us"]["fmt_5_8"]
            row["bytes_win_306_vs_58"] = b_rung - b_flat
            if b_rung < b_flat and t_nest <= t_58 * 1.05:
                nest_verdict = "rung306_bytes_and_nested_decode_ok"
            elif b_rung < b_flat:
                nest_verdict = (
                    "rung306_wins_bytes_but_decode_slower_than_5_8"
                    f"_nested={t_nest:.1f}us_vs_58={t_58:.1f}us"
                )
            elif b_frame < b_flat:
                nest_verdict = "chiral_486_frame_unexpected_byte_win"
            elif t_nest < t_flat_digit * 0.95:
                nest_verdict = "nested_faster_than_flat_digit_only"
            else:
                nest_verdict = "keep_never_nested_prefer_flat58_or_rung_block"
        elif n % 665 == 0:
            b_frame = row["bytes"]["fmt_665_frame"]
            b_flat = row["bytes"]["fmt_5_8"]
            t_f = row["decode_us"]["665_frame"]
            t_58 = row["decode_us"]["fmt_5_8"]
            if b_frame < b_flat and t_f <= t_58 * 1.05:
                nest_verdict = "665_frame_wins_joint"
            elif b_frame <= b_flat:
                nest_verdict = "665_frame_ties_or_beats_bytes_decode_slower"
            else:
                nest_verdict = "665_frame_no_win"
        row["nesting_verdict"] = nest_verdict
        rows.append(row)

    # Allow nest only if joint win; separately note rung-block byte wins
    keep_nest = any(
        "nested_decode_ok" in r["nesting_verdict"]
        or r["nesting_verdict"] == "665_frame_wins_joint"
        for r in rows
    )
    rung_block_byte_wins = any(
        r.get("bytes_win_306_vs_58", 0) < 0 for r in rows if "bytes_win_306_vs_58" in r
    )
    return {
        "repeats": repeats,
        "rows": rows,
        "recommendation": (
            "allow_one_level_nest_with_rung_block"
            if keep_nest
            else "keep_never_nested_rule"
        ),
        "rung_block_byte_wins_vs_5_8": rung_block_byte_wins,
        "note": (
            "fmt_306_485 often saves ~1 B per 306 trits vs 5_8 but decode is "
            "slower. Nesting (Law C digits) does not beat 5_8 decode speed. "
            "Use rung block for size-critical offline storage; keep 5_8 for "
            "decode-critical paths unless nest wins joint."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def selftest() -> None:
    f = evaluate_frame_options(306, 1)
    assert f["eligible"]
    assert f["best"]["quantum"] == 306
    assert f["best"]["ledger"] == "fmt_306_485"  # density, not 486-frame
    r = pack_tensor(2560, 2560)
    assert r["decision"]["path"] in ("flat58", "fiber41")
    # 1×306 stream should prefer fmt_306_485 over flat58 (61 < 62)
    r306 = pack_tensor(1, 306)
    assert r306["frames"]["eligible"]
    assert r306["decision"]["ledger"] == "fmt_306_485"
    assert r306["decision"]["bytes"] == 61
    # Hybrid on non-multiple (e.g. 2560) beats flat when enabled
    r_off = pack_tensor(1, 2560, allow_hybrid=False)
    assert r_off["decision"]["path"] == "flat58"
    r_hy = pack_tensor(1, 2560, allow_hybrid=True)
    assert r_hy["decision"]["path"] == "hybrid"
    assert r_hy["decision"]["bytes"] < r_off["decision"]["bytes"]
    h = evaluate_hybrid_306(1, 2560)
    assert h["best"]["bytes"] == theory_bytes_306_485(2560)
    nt = nesting_test(lengths=(306, 665), repeats=5)
    assert nt["rows"][0]["rt_ok"].get("fmt_5_8")
    print("ledger_packer selftest OK")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "cmd",
        choices=["selftest", "pack", "bitnet", "nest", "all"],
    )
    p.add_argument("--m", type=int, default=2560)
    p.add_argument("--n", type=int, default=2560)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--max-tensors", type=int, default=None)
    p.add_argument("--skip-ckpt", action="store_true")
    p.add_argument("--nest-repeats", type=int, default=40)
    p.add_argument(
        "--hybrid",
        action="store_true",
        help="Allow 306-prefix hybrid without exact ÷306 (known ~0.33%% density)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "selftest":
        selftest()
        return 0

    out: Dict[str, Any] = {
        "stance": (
            "Practical three-ledger packer. Measure bytes; diagnose non-wins. "
            "Odd rungs → sheets ok at design time; even 306 → frame only. "
            "Optional --hybrid: 306-prefix without exact gate (61/306 effect)."
        ),
        "design_time_surplus_align64": list(DESIGN_TIME_SURPLUS_ALIGN64),
        "odd_rungs_sheet_ok": list(ODD_RUNGS),
        "even_frame_only": list(EVEN_FRAME_LENGTHS),
        "allow_hybrid": args.hybrid,
    }

    if args.cmd in ("pack", "all"):
        out["example_pack"] = pack_tensor(args.m, args.n, allow_hybrid=args.hybrid)

    if args.cmd in ("nest", "all"):
        out["nesting"] = nesting_test(repeats=args.nest_repeats)

    if args.cmd in ("bitnet", "all") and not args.skip_ckpt:
        if not args.ckpt.is_file():
            out["bitnet"] = {"error": f"missing {args.ckpt}"}
        else:
            t0 = time.time()
            out["bitnet"] = measure_bitnet(
                args.ckpt,
                max_tensors=args.max_tensors,
                allow_hybrid=args.hybrid,
            )
            out["bitnet"]["elapsed_s"] = time.time() - t0
            out["bitnet"]["allow_hybrid"] = args.hybrid

    path = Path(__file__).resolve().parent / "ledger_packer_results.json"
    summary: Dict[str, Any] = {
        "design_time_surplus_align64": out["design_time_surplus_align64"],
        "odd_even": {
            "odd_rungs_sheet_ok": out["odd_rungs_sheet_ok"],
            "even_frame_only": out["even_frame_only"],
        },
    }
    if "example_pack" in out:
        summary["example_pack"] = out["example_pack"]["decision"]
        summary["example_candidates"] = out["example_pack"]["candidates"]
    if "nesting" in out:
        summary["nesting_recommendation"] = out["nesting"]["recommendation"]
        summary["nesting_rows"] = [
            {
                "n": r["n"],
                "bytes": r["bytes"],
                "decode_us": {k: round(v, 2) for k, v in r["decode_us"].items()},
                "verdict": r["nesting_verdict"],
            }
            for r in out["nesting"]["rows"]
        ]
    if "bitnet" in out and "totals" in out.get("bitnet", {}):
        b = out["bitnet"]
        summary["bitnet"] = {
            "verdict": b["verdict"],
            "n_tensors": b["totals"]["n_tensors"],
            "bytes_selected": b["totals"]["bytes_selected"],
            "bytes_flat58": b["totals"]["bytes_flat58"],
            "bytes_fiber41_orig": b["totals"]["bytes_fiber41_orig"],
            "bytes_oracle": b["totals"]["bytes_oracle"],
            "bytes_shipped_u8": b["totals"]["bytes_shipped_u8"],
            "delta_vs_flat58": b["delta_selected_vs_flat58"],
            "delta_vs_fiber41": b["delta_selected_vs_fiber41"],
            "delta_vs_u8": b["delta_selected_vs_u8"],
            "path_counts": b["path_counts"],
            "failure_analysis": b["failure_analysis"],
            "per_unique": b["per_unique"],
        }

    print(json.dumps(summary, indent=2))
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
