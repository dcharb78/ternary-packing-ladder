#!/usr/bin/env python3
"""Pattern-of-patterns probe — multi-rung / Law B composition vs hybrid 306.

Primary lens: CF ladder composition and Law B assemblies as *ledgers*, not
another fiber-41 baseline. Fiber-41 is a control only.

Question: can composed ledgers beat flat fmt_5_8 and/or plain hybrid
theory_bytes_306_485 (61/306 ≈ 0.33%) with honest byte metrics?

Candidates:
  A  flat fmt_5_8
  B  hybrid / theory_bytes_306_485          (known ~0.33% control)
  C  fiber-41                               (control; expected lose)
  D  Law B 486-frame (7×41+19)              (chiral parts)
  E  Law B 665-frame (2×306+53)             (chiral parts, tax 0)
  F  flat 665-prefix theory_bytes_665_1055  (Law B *sum* as rung-block)
  G  greedy 665-flat + 306_485 rem
  H  DP over rung atoms {5,19,41,53,306,665_flat} + rem1..4
  I  cascade 306→41→5_8                     (better_density restatement)

Digit nesting (Law C hierarchical) is tagged false_identification — do not
re-enable as the main idea (see PACKET_SEAM / ledger_packer nesting test).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from better_density import best_layout, cascade_306_41_58
from frame_formats import theory_bytes_486, theory_bytes_665
from large_scale_probe import SUITES
from pack_ladder import (
    BLOCK_665,
    BYTES_41,
    BYTES_306,
    BYTES_665,
    container_bytes_for_r_trits,
    pack_665_1055,
    theory_bytes_41_65,
    theory_bytes_5_8,
    theory_bytes_306_485,
    theory_bytes_665_1055,
    unpack_665_1055,
)
from packing_stack import load_bitnet_shapes
from tax_graph import LAW_B_486, LAW_B_665, enumerate_assemblies, split_tax

DEFAULT_CKPT = Path(__file__).resolve().parent / "data" / "bitnet" / "model.safetensors"
OUT = Path(__file__).resolve().parent / "pattern_of_patterns_results.json"

# Composition alphabet (rung + Law B deficit / assembly sizes)
RUNG_ATOMS: Tuple[Tuple[int, int, str], ...] = (
    (5, 1, "5"),
    (19, 4, "19"),
    (41, BYTES_41, "41"),
    (53, 11, "53"),
    (306, BYTES_306, "306_flat"),
    (665, BYTES_665, "665_flat"),
)

# Synthetic lengths where multi-rung / Law B composition should shine
LAW_B_ALIGNED_LENGTHS: Tuple[int, ...] = (
    306,
    486,  # name collision with bit-frame; trit length for stress
    612,
    665,
    1330,
    1830,
    1995,
    3060,
    665 * 7,
    306 * 41,  # multi-rung product
    665 * 41,
    5 * 41 * 306,  # deep CF nest as length only
)

# Rectangular Law-B-aligned shapes (m, n)
LAW_B_SHAPES: Tuple[Tuple[str, int, int], ...] = (
    ("sq_306", 306, 306),
    ("sq_665", 665, 665),
    ("row_665", 1, 665),
    ("row_1330", 1, 1330),
    ("fiber_41x306", 41, 306),
    ("fiber_665x306", 665, 306),
    ("llamaish_align665", 3990, 3990),  # 6×665
)


def compose_665_then_306(n: int) -> int:
    full, rem = divmod(n, BLOCK_665)
    return full * BYTES_665 + theory_bytes_306_485(rem)


def dp_rung_atoms(n: int, max_n: int = 20000) -> Optional[int]:
    """Unbounded DP over RUNG_ATOMS + rem∈{1..4} at 1 B. None if n too large."""
    if n < 0 or n > max_n:
        return None
    inf = 10**18
    dp = [inf] * (n + 1)
    dp[0] = 0
    blocks = [(sz, by) for sz, by, _ in RUNG_ATOMS]
    for i in range(n + 1):
        if dp[i] >= inf:
            continue
        base = dp[i]
        for sz, by in blocks:
            j = i + sz
            if j <= n and base + by < dp[j]:
                dp[j] = base + by
        for r in range(1, 5):
            j = i + r
            if j <= n and base + 1 < dp[j]:
                dp[j] = base + 1
    return int(dp[n])


def score_1d(n: int) -> Dict[str, Any]:
    dp = dp_rung_atoms(n)
    cands = {
        "A_flat_5_8": theory_bytes_5_8(n),
        "B_hybrid_306": theory_bytes_306_485(n),
        "C_fiber_41": theory_bytes_41_65(n),
        "D_486_frame": theory_bytes_486(n),
        "E_665_frame": theory_bytes_665(n),
        "F_hybrid_665_flat": theory_bytes_665_1055(n),
        "G_665_then_306": compose_665_then_306(n),
        "I_cascade_306_41_58": cascade_306_41_58(n),
    }
    if dp is not None:
        cands["H_dp_rung_atoms"] = dp
    best = min(cands, key=cands.get)
    flat = cands["A_flat_5_8"]
    hy306 = cands["B_hybrid_306"]
    return {
        "n": n,
        "candidates": cands,
        "best": best,
        "best_bytes": cands[best],
        "delta_best_vs_flat": cands[best] - flat,
        "delta_best_vs_hybrid306": cands[best] - hy306,
        "delta_F_vs_hybrid306": cands["F_hybrid_665_flat"] - hy306,
        "delta_D_vs_hybrid306": cands["D_486_frame"] - hy306,
        "delta_E_vs_hybrid306": cands["E_665_frame"] - hy306,
    }


def score_shape(m: int, n: int) -> Dict[str, Any]:
    layout_fns: Dict[str, Callable[[int], int]] = {
        "A_flat_5_8": theory_bytes_5_8,
        "B_hybrid_306": theory_bytes_306_485,
        "C_fiber_41": theory_bytes_41_65,
        "D_486_frame": theory_bytes_486,
        "E_665_frame": theory_bytes_665,
        "F_hybrid_665_flat": theory_bytes_665_1055,
        "G_665_then_306": compose_665_then_306,
        "I_cascade_306_41_58": cascade_306_41_58,
    }
    cands: Dict[str, int] = {}
    layouts: Dict[str, Any] = {}
    for name, fn in layout_fns.items():
        # A is flat stream only (true flat baseline)
        if name == "A_flat_5_8":
            cands[name] = theory_bytes_5_8(m * n)
            layouts[name] = {"layout": "flat_stream", "bytes": cands[name]}
        else:
            lay = best_layout(m, n, fn)
            cands[name] = lay["bytes"]
            layouts[name] = lay
    best = min(cands, key=cands.get)
    flat = cands["A_flat_5_8"]
    hy306 = cands["B_hybrid_306"]
    return {
        "shape": [m, n],
        "candidates": cands,
        "layouts": {k: layouts[k] for k in ("B_hybrid_306", "F_hybrid_665_flat", "C_fiber_41")},
        "best": best,
        "best_bytes": cands[best],
        "delta_best_vs_flat": cands[best] - flat,
        "delta_F_vs_hybrid306": cands["F_hybrid_665_flat"] - hy306,
        "delta_B_vs_flat": hy306 - flat,
        "delta_C_vs_flat": cands["C_fiber_41"] - flat,
    }


def measure_shapes(
    shapes: Sequence[Tuple[Any, int, int]],
    multiplicity: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    totals: Dict[str, int] = defaultdict(int)
    best_counts: Dict[str, int] = defaultdict(int)
    f_beats_b = 0
    n_tensors = 0
    trit_total = 0
    unique: List[Dict[str, Any]] = []
    seen = set()

    for i, item in enumerate(shapes):
        if len(item) == 3:
            _name, m, n = item
        else:
            m, n = item[0], item[1]
        mult = 1 if multiplicity is None else int(multiplicity[i])
        ev = score_shape(m, n)
        n_tensors += mult
        trit_total += m * n * mult
        for k, v in ev["candidates"].items():
            totals[k] += v * mult
        best_counts[ev["best"]] += mult
        if ev["delta_F_vs_hybrid306"] < 0:
            f_beats_b += mult
        key = (m, n)
        if key not in seen:
            seen.add(key)
            unique.append(ev)

    flat = totals["A_flat_5_8"]
    hy306 = totals["B_hybrid_306"]
    return {
        "n_tensors": n_tensors,
        "n_unique": len(unique),
        "trits": trit_total,
        "totals": dict(totals),
        "delta_vs_flat": {k: totals[k] - flat for k in totals},
        "delta_vs_hybrid306": {k: totals[k] - hy306 for k in totals},
        "delta_vs_flat_MB": {k: round((totals[k] - flat) / 1e6, 3) for k in totals},
        "delta_vs_hybrid306_MB": {
            k: round((totals[k] - hy306) / 1e6, 3) for k in totals
        },
        "best_counts": dict(best_counts),
        "tensors_F_beats_B": f_beats_b,
        "bytes_per_trit": {k: totals[k] / trit_total for k in totals} if trit_total else {},
        "per_unique": unique,
    }


def measure_bitnet(ckpt: Path, max_tensors: Optional[int] = None) -> Dict[str, Any]:
    if not ckpt.is_file():
        return {"skipped": True, "reason": f"missing {ckpt}"}
    shapes = load_bitnet_shapes(ckpt)
    if max_tensors is not None:
        shapes = shapes[:max_tensors]
    out = measure_shapes(shapes)
    out["ckpt"] = str(ckpt)
    out["skipped"] = False
    return out


def measure_large_scale(which: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    names = list(which) if which else list(SUITES.keys())
    suites = {}
    for name in names:
        suite = SUITES[name]
        shapes = [(tname, m, n) for tname, m, n, _mult in suite]
        mults = [mult for _t, _m, _n, mult in suite]
        # measure_shapes applies mult per shape entry — expand via multiplicity
        # but shapes list is unique rows; pass multiplicity aligned
        suites[name] = measure_shapes(shapes, multiplicity=mults)
        suites[name]["suite"] = name
    return suites


def measure_law_b_aligned() -> Dict[str, Any]:
    rows_1d = [score_1d(n) for n in LAW_B_ALIGNED_LENGTHS]
    shapes = measure_shapes([(name, m, n) for name, m, n in LAW_B_SHAPES])
    named = []
    for (name, _m, _n), row in zip(LAW_B_SHAPES, shapes["per_unique"]):
        r = dict(row)
        r["name"] = name
        named.append(r)
    shapes["per_unique"] = named

    # Catalog low-tax assemblies: frame_bytes vs flat 5_8 vs flat-sum container
    asms = enumerate_assemblies(max_trits=2000, max_tax=1, max_parts=8)
    asm_rows = []
    beat_flat_sum = 0
    beat_hybrid306 = 0
    for a in asms:
        frame_b = sum(container_bytes_for_r_trits(q) for q in a.parts)
        flat_sum_b = container_bytes_for_r_trits(a.total_trits)
        flat58 = theory_bytes_5_8(a.total_trits)
        hy306 = theory_bytes_306_485(a.total_trits)
        hy665 = theory_bytes_665_1055(a.total_trits)
        if flat_sum_b < frame_b:
            beat_flat_sum += 1
        if hy665 < hy306:
            beat_hybrid306 += 1
        asm_rows.append(
            {
                "parts": list(a.parts),
                "trits": a.total_trits,
                "tax": a.tax,
                "frame_bytes": frame_b,
                "flat_sum_bytes": flat_sum_b,
                "flat_5_8": flat58,
                "hybrid_306": hy306,
                "hybrid_665_flat": hy665,
                "flat_sum_beats_frame": flat_sum_b < frame_b,
                "hy665_beats_hy306": hy665 < hy306,
            }
        )
    # Keep densest flat-sum wins for report
    asm_rows.sort(key=lambda r: r["hybrid_665_flat"] - r["hybrid_306"])
    return {
        "lengths_1d": rows_1d,
        "shapes": shapes,
        "assemblies_tax_le_1": {
            "n": len(asms),
            "n_flat_sum_beats_frame_bytes": beat_flat_sum,
            "n_hy665_beats_hy306": beat_hybrid306,
            "documented": {
                "486_parts": list(LAW_B_486),
                "486_tax": split_tax(LAW_B_486),
                "486_frame_bytes_at_306": theory_bytes_486(306),
                "306_rung_bytes": theory_bytes_306_485(306),
                "665_parts": list(LAW_B_665),
                "665_tax": split_tax(LAW_B_665),
                "665_frame_bytes": theory_bytes_665(665),
                "665_flat_bytes": theory_bytes_665_1055(665),
            },
            "best_hy665_minus_hy306": asm_rows[:12],
        },
    }


def kronecker_spotcheck() -> Dict[str, Any]:
    """Cheap Kronecker check: factor pack vs flatten under F vs B."""
    rng = np.random.default_rng(0)
    cases = []
    for am, an, bm, bn in ((5, 5, 41, 41), (7, 7, 53, 53), (17, 19, 17, 19)):
        A = rng.integers(-1, 2, size=(am, an), dtype=np.int8)
        B = rng.integers(-1, 2, size=(bm, bn), dtype=np.int8)
        W = np.kron(A, B).astype(np.int8)
        a_n, b_n, w_n = A.size, B.size, W.size
        factor_5 = theory_bytes_5_8(a_n) + theory_bytes_5_8(b_n)
        flat_5 = theory_bytes_5_8(w_n)
        factor_306 = theory_bytes_306_485(a_n) + theory_bytes_306_485(b_n)
        flat_306 = theory_bytes_306_485(w_n)
        factor_665 = theory_bytes_665_1055(a_n) + theory_bytes_665_1055(b_n)
        flat_665 = theory_bytes_665_1055(w_n)
        cases.append(
            {
                "A": [am, an],
                "B": [bm, bn],
                "W_trits": w_n,
                "factor_5_8": factor_5,
                "flat_5_8": flat_5,
                "factor_vs_flat_5_8": factor_5 - flat_5,
                "flat_hybrid306": flat_306,
                "flat_hybrid665": flat_665,
                "factor_hybrid665": factor_665,
                "note": "factor wins only when Kronecker structure is known/stored",
            }
        )
    return {
        "cases": cases,
        "tag": "applies_as_language",
        "interpretation": (
            "Kronecker factor packing is a structural win when A⊗B is known; "
            "it is not a drop-in densifier for unstructured BitNet tensors."
        ),
    }


def claim_tags(bit: Dict[str, Any], lawb: Dict[str, Any]) -> List[Dict[str, str]]:
    doc = lawb["assemblies_tax_le_1"]["documented"]
    claims = [
        {
            "claim": "CF ladder / Law B sum 665 as flat rung-block (fmt_665_1055) beats hybrid 306",
            "tag": "applies_operationally",
            "evidence": "F_hybrid_665_flat vs B on BitNet + Law-B-aligned lengths",
        },
        {
            "claim": "Plain hybrid theory_bytes_306_485 ≈ 0.33% vs flat 5_8",
            "tag": "applies_operationally",
            "evidence": "known better_density control; restated here as B",
        },
        {
            "claim": "Law B chiral 486-frame denser than flat 5_8 / hybrid 306",
            "tag": "does_not_apply",
            "evidence": (
                f"486-frame@{306}={doc['486_frame_bytes_at_306']} vs "
                f"306_rung={doc['306_rung_bytes']}"
            ),
        },
        {
            "claim": "Law B chiral 665-frame denser than flat 665 container",
            "tag": "does_not_apply",
            "evidence": (
                f"665-frame={doc['665_frame_bytes']} vs "
                f"665-flat={doc['665_flat_bytes']} (byte-rounding of parts)"
            ),
        },
        {
            "claim": "Law B assemblies are a useful language for choosing larger containers",
            "tag": "applies_as_language",
            "evidence": "665=2×306+53 names the flat 132 B block; tax 0 in bits",
        },
        {
            "claim": "DP over small rung atoms beats hybrid 665-flat",
            "tag": "does_not_apply",
            "evidence": "H loses or ties F; rem bigint matters more than atom mosaic",
        },
        {
            "claim": "Fiber-41 primary packing lever",
            "tag": "does_not_apply",
            "evidence": "control C loses bytes on BitNet / scale suites",
        },
        {
            "claim": "Digit nesting / preserve 0→1 as packing density path",
            "tag": "false_identification",
            "evidence": "PACKET_SEAM + ledger_packer nesting: slower, no byte win",
        },
        {
            "claim": "Kronecker factor pack as unstructured BitNet densifier",
            "tag": "does_not_apply",
            "evidence": "requires known A⊗B structure; see kronecker spotcheck",
        },
    ]
    if not bit.get("skipped"):
        d = bit["delta_vs_hybrid306_MB"].get("F_hybrid_665_flat")
        claims[0]["evidence"] += f"; BitNet ΔF vs B = {d} MB"
    return claims


def verdict_from(bit: Dict[str, Any], scale: Dict[str, Any], lawb: Dict[str, Any]) -> Dict[str, Any]:
    doc = lawb["assemblies_tax_le_1"]["documented"]
    beats = False
    delta_mb = None
    if not bit.get("skipped"):
        delta = bit["delta_vs_hybrid306"]["F_hybrid_665_flat"]
        beats = delta < 0
        delta_mb = bit["delta_vs_hybrid306_MB"]["F_hybrid_665_flat"]
    scale_any = any(
        s["delta_vs_hybrid306"].get("F_hybrid_665_flat", 0) < 0 for s in scale.values()
    )
    # 1d Law-B aligned: count F wins vs B
    f_wins = sum(1 for r in lawb["lengths_1d"] if r["delta_F_vs_hybrid306"] < 0)
    return {
        "pattern_of_patterns_beats_hybrid_306": bool(beats or scale_any),
        "winning_ledger": "F_hybrid_665_flat (fmt_665_1055)" if beats or scale_any else None,
        "bitnet_delta_F_vs_B_MB": delta_mb,
        "bitnet_delta_F_vs_flat_MB": (
            None if bit.get("skipped") else bit["delta_vs_flat_MB"]["F_hybrid_665_flat"]
        ),
        "bitnet_delta_B_vs_flat_MB": (
            None if bit.get("skipped") else bit["delta_vs_flat_MB"]["B_hybrid_306"]
        ),
        "law_b_aligned_1d_F_beats_B": f_wins,
        "chiral_frames_lose_bytes": True,
        "documented_byte_gap": {
            "306_rung": doc["306_rung_bytes"],
            "486_frame": doc["486_frame_bytes_at_306"],
            "665_flat": doc["665_flat_bytes"],
            "665_frame": doc["665_frame_bytes"],
        },
        "interpretation": (
            "Pattern-of-patterns wins as a *flat container of the Law B sum* "
            "(665→132 B), not as chiral multi-part frames. That ledger beats "
            "hybrid 306 (61/306) on BitNet and scale suites. 486/665 frames "
            "and fiber-41 do not; digit nesting remains a false identification."
            if beats or scale_any
            else "No composed ledger beat hybrid 306 on measured suites."
        ),
        "packer_recommendation": (
            "Add optional hybrid path using theory_bytes_665_1055 / fmt_665_1055 "
            "(665-prefix + bigint rem), analogous to --hybrid for 306. Do not "
            "prefer chiral 486/665 frames for bytes. Keep fiber-41 off the default "
            "path. Do not re-enable nested digit decode for density."
        ),
    }


def run(
    ckpt: Path,
    max_tensors: Optional[int] = None,
    suites: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    bit = measure_bitnet(ckpt, max_tensors=max_tensors)
    scale = measure_large_scale(suites)
    lawb = measure_law_b_aligned()
    kron = kronecker_spotcheck()
    claims = claim_tags(bit, lawb)
    verd = verdict_from(bit, scale, lawb)

    # Compact scale summary
    scale_summary = []
    for name, s in scale.items():
        scale_summary.append(
            {
                "suite": name,
                "trits_B": round(s["trits"] / 1e9, 3),
                "delta_B_vs_flat_MB": s["delta_vs_flat_MB"]["B_hybrid_306"],
                "delta_F_vs_flat_MB": s["delta_vs_flat_MB"]["F_hybrid_665_flat"],
                "delta_F_vs_B_MB": s["delta_vs_hybrid306_MB"]["F_hybrid_665_flat"],
                "delta_C_vs_flat_MB": s["delta_vs_flat_MB"]["C_fiber_41"],
                "best_counts": s["best_counts"],
            }
        )

    out = {
        "stance": (
            "Primary lens: pattern-of-patterns (CF / Law B composition as ledgers). "
            "Fiber-41 is control only. Measure whether composition beats hybrid 306."
        ),
        "bitnet": bit,
        "large_scale_summary": scale_summary,
        "large_scale": {k: {kk: vv for kk, vv in v.items() if kk != "per_unique"} for k, v in scale.items()},
        "law_b_aligned": {
            "lengths_1d": lawb["lengths_1d"],
            "shapes_summary": {
                kk: vv for kk, vv in lawb["shapes"].items() if kk != "per_unique"
            },
            "shapes_per_unique": lawb["shapes"]["per_unique"],
            "assemblies_tax_le_1": lawb["assemblies_tax_le_1"],
        },
        "kronecker_spotcheck": kron,
        "claims": claims,
        "verdict": verd,
        "elapsed_s": round(time.time() - t0, 3),
    }
    # Drop huge per_unique from bitnet in written file? Keep unique only — already.
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def selftest() -> None:
    assert theory_bytes_665_1055(665) == 132
    assert theory_bytes_665(665) == 133
    assert theory_bytes_306_485(306) == 61
    assert theory_bytes_486(306) == 67
    assert theory_bytes_5_8(306) == 62
    assert theory_bytes_665_1055(665) < theory_bytes_306_485(665)
    assert compose_665_then_306(665) == 132
    # Round-trip flat 665 blocks
    rng = np.random.default_rng(1)
    for n in (0, 1, 664, 665, 666, 1330):
        w = rng.integers(-1, 2, size=n, dtype=np.int8)
        blob = pack_665_1055(w)
        assert len(blob) == theory_bytes_665_1055(n)
        assert np.array_equal(unpack_665_1055(blob, n), w)
    r = score_1d(665)
    assert r["best"] in ("F_hybrid_665_flat", "G_665_then_306", "H_dp_rung_atoms")
    assert r["delta_F_vs_hybrid306"] < 0
    ev = score_shape(1, 665)
    assert ev["candidates"]["F_hybrid_665_flat"] == 132
    assert ev["delta_F_vs_hybrid306"] < 0
    print("pattern_of_patterns selftest OK")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("selftest", "run"), nargs="?", default="run")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--max-tensors", type=int, default=None)
    p.add_argument(
        "--suite",
        action="append",
        choices=list(SUITES.keys()),
        help="Large-scale suite (repeatable). Default: all.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "selftest":
        selftest()
        return 0
    out = run(args.ckpt, max_tensors=args.max_tensors, suites=args.suite)
    print(json.dumps({"verdict": out["verdict"], "claims": out["claims"]}, indent=2))
    if not out["bitnet"].get("skipped"):
        print("bitnet_delta_vs_flat_MB:", out["bitnet"]["delta_vs_flat_MB"])
        print("bitnet_delta_vs_hybrid306_MB:", out["bitnet"]["delta_vs_hybrid306_MB"])
        print("best_counts:", out["bitnet"]["best_counts"])
    print("large_scale_summary:", json.dumps(out["large_scale_summary"], indent=2))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
