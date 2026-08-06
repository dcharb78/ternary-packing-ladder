#!/usr/bin/env python3
"""Pad mode length to nearest tax-0 frame multiple — cost vs density.

Stock LLM widths rarely divide catalog tax-0 assemblies. This probe asks:
for each mode length L, find the smallest L' >= L that is a multiple of some
tax-0 frame_q, measure pad trits and whether framed packing on L' beats
naive packing on L (exact byte ledgers).

Hypothesis: small pads unlock zero-tax tiling grammar; pad cost may be
offset when the unlocked frame is denser than the stock remainder pattern.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from frame_formats import theory_bytes_frame
from pack_ladder import theory_bytes_41_65, theory_bytes_5_8
from scale_probe import LLM_SHAPES
from tax_graph import bits, build_catalog, split_tax


def tax0_frames(max_trits: int = 800) -> List[Tuple[int, ...]]:
    cat = build_catalog(max_trits=max_trits, max_tax=0)
    deficit = {19, 53}
    out: List[Tuple[int, ...]] = []
    seen = set()
    for a in cat["assemblies"]:
        parts = tuple(a["parts"])
        if a["tax"] != 0 or len(parts) < 2:
            continue
        if not any(p in deficit for p in parts):
            continue
        if parts in seen:
            continue
        seen.add(parts)
        out.append(parts)
    out.append((306, 306, 53))
    out.sort(key=lambda p: sum(p))
    return out


def nearest_pad(L: int, frame_q: int) -> Tuple[int, int]:
    """Return (L', pad) with L' >= L, L' % frame_q == 0, minimal pad."""
    if frame_q <= 0:
        raise ValueError("frame_q")
    if L % frame_q == 0:
        return L, 0
    L2 = ((L + frame_q - 1) // frame_q) * frame_q
    return L2, L2 - L


def evaluate_length(L: int, frames: Sequence[Tuple[int, ...]], max_pad: int = 512) -> Dict:
    """Best pad-to-tax0 option for a 1-D mode length L (as fiber length)."""
    base_5 = theory_bytes_5_8(L)
    base_41 = theory_bytes_41_65(L)
    candidates = []
    for parts in frames:
        q = sum(parts)
        if q > L + max_pad:
            continue
        Lp, pad = nearest_pad(L, q)
        if pad > max_pad:
            continue
        framed = theory_bytes_frame(parts, Lp)
        # Compare framed(Lp) vs packing original L with 41_65 / 5_8
        candidates.append(
            {
                "parts": list(parts),
                "frame_q": q,
                "L_prime": Lp,
                "pad_trits": pad,
                "bytes_framed_Lp": framed,
                "bytes_5_8_L": base_5,
                "bytes_41_65_L": base_41,
                "delta_vs_5_8": framed - base_5,
                "delta_vs_41_65": framed - base_41,
                "tax": split_tax(parts),
            }
        )
    # Prefer smallest pad among those that beat 41_65, else smallest pad overall
    wins = [c for c in candidates if c["delta_vs_41_65"] < 0]
    if wins:
        best = min(wins, key=lambda c: (c["pad_trits"], c["bytes_framed_Lp"]))
    elif candidates:
        best = min(candidates, key=lambda c: (c["pad_trits"], c["delta_vs_41_65"]))
    else:
        best = None
    return {
        "L": L,
        "bytes_5_8": base_5,
        "bytes_41_65": base_41,
        "n_candidates": len(candidates),
        "n_beat_41_65": len(wins),
        "best": best,
        "top_wins": sorted(wins, key=lambda c: c["delta_vs_41_65"])[:5],
    }


def run() -> Dict:
    frames = tax0_frames()
    # Unique mode lengths from LLM shapes + a few probes
    lengths = sorted({m for _, m, n in LLM_SHAPES} | {n for _, m, n in LLM_SHAPES})
    per_L = {str(L): evaluate_length(L, frames) for L in lengths}
    # Per full matrix: pad the longer mode only (common MLP case)
    matrices = []
    for name, m, n in LLM_SHAPES:
        # Evaluate padding each mode independently for row-fiber packing of length n
        en = evaluate_length(n, frames)
        em = evaluate_length(m, frames)
        matrices.append(
            {
                "name": name,
                "shape": [m, n],
                "pad_col_fiber_n": en["best"],
                "pad_row_fiber_m": em["best"],
                "n_beat_41_on_n": en["n_beat_41_65"],
                "n_beat_41_on_m": em["n_beat_41_65"],
            }
        )
    n_win = sum(1 for v in per_L.values() if v["n_beat_41_65"] > 0)
    return {
        "hypothesis": (
            "Padding a mode length to a tax-0 frame multiple can unlock denser "
            "tiling; measure pad cost vs fmt_41_65 on the unpadded length."
        ),
        "n_tax0_frames": len(frames),
        "n_lengths": len(lengths),
        "n_lengths_with_win_vs_41_65": n_win,
        "per_length": per_L,
        "matrices": matrices,
    }


def selftest() -> int:
    frames = tax0_frames()
    assert any(sum(p) == 665 for p in frames)
    # 60 = 41+19 tax 0
    assert nearest_pad(59, 60) == (60, 1)
    assert nearest_pad(60, 60) == (60, 0)
    r = evaluate_length(59, frames, max_pad=64)
    assert r["best"] is not None
    print(f"PAD_TO_TAX0 unit OK frames={len(frames)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path(__file__).resolve().parent / "pad_to_tax0_results.json"
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args and args[0] == "selftest":
        return selftest()
    selftest()
    report = run()
    dest = Path(args[1]) if len(args) > 1 else out
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PAD_TO_TAX0 wrote {dest}")
    print(f"lengths with any win vs fmt_41_65: {report['n_lengths_with_win_vs_41_65']}/{report['n_lengths']}")
    for L, v in sorted(report["per_length"].items(), key=lambda kv: int(kv[0])):
        b = v["best"]
        if not b:
            continue
        mark = "WIN" if b["delta_vs_41_65"] < 0 else "pad"
        print(
            f"  L={L:>6} {mark} pad={b['pad_trits']:>4} "
            f"frame_q={b['frame_q']:<5} d41={b['delta_vs_41_65']:>6} "
            f"parts={b['parts']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
