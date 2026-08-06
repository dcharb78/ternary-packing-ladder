#!/usr/bin/env python3
"""Targeted scale probe — axis choice, novel tax-0 tiles, Kronecker factors.

NOT a blind larger flat stream. Tests the levers Phase 1/2 said matter:

  1. Axis choice on realistic LLM rectangle shapes
  2. Novel tax-0 frames as tile grammar
  3. Kronecker / block-Kronecker factor packing at scale
  4. Stream-slack vs block-axis tension (unseen connection)

Exact integers for all size verdicts. Large-Q bits(Q)=(3**Q).bit_length()
is used only for mode lengths (thousands), never for area MN.

Key identity (no bits(MN) needed for axis choice):
  row_bits - col_bits = m*bits(n) - n*bits(m)
  = tax_rows(m,n) - tax_cols(m,n)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from frame_formats import theory_bytes_486, theory_bytes_665
from kronecker_tensor_pack import ledger as kron_ledger, ternary_kron
from pack_ladder import (
    theory_bytes_41_65,
    theory_bytes_5_8,
    theory_bytes_306_485,
)
from tax_graph import bits, build_catalog, split_tax
from tax_tensor import tax_cols, tax_rows

# Realistic LLM-ish weight shapes (out, in) — LLaMA-class and friends.
LLM_SHAPES: Tuple[Tuple[str, int, int], ...] = (
    ("llama7b_attn_q", 4096, 4096),
    ("llama7b_mlp_up", 11008, 4096),
    ("llama7b_mlp_down", 4096, 11008),
    ("llama13b_attn_q", 5120, 5120),
    ("llama13b_mlp_up", 13824, 5120),
    ("llama34b_attn_q", 8192, 8192),
    ("llama34b_mlp_up", 22016, 8192),
    ("bitnet2b_like_q", 2560, 2560),
    ("bitnet2b_like_mlp", 6912, 2560),
    ("head_proj_slice", 128, 4096),  # one head's worth
    ("kv_group_rect", 1024, 4096),
)


def axis_bits(m: int, n: int) -> Dict[str, int]:
    """Exact row-vs-col container bit totals (one flat container per fiber)."""
    br = m * bits(n)
    bc = n * bits(m)
    return {
        "m": m,
        "n": n,
        "row_bits": br,
        "col_bits": bc,
        "delta_bits": br - bc,  # >0 means cols win
        "best_axis": "rows" if br <= bc else "cols",
        "bits_saved_vs_worse": abs(br - bc),
    }


def layout_bytes(m: int, n: int) -> Dict[str, int]:
    """Practical block-format byte counts: flat vs row-fibers vs col-fibers."""
    mn = m * n
    return {
        "flat_5_8": theory_bytes_5_8(mn),
        "flat_41_65": theory_bytes_41_65(mn),
        "flat_306_485": theory_bytes_306_485(mn),
        "flat_486": theory_bytes_486(mn),
        "row_5_8": m * theory_bytes_5_8(n),
        "col_5_8": n * theory_bytes_5_8(m),
        "row_41_65": m * theory_bytes_41_65(n),
        "col_41_65": n * theory_bytes_41_65(m),
        "row_306": m * theory_bytes_306_485(n),
        "col_306": n * theory_bytes_306_485(m),
    }


def stream_slack_tension(m: int, n: int, slack_per_stream: int = 4) -> Dict[str, int]:
    """Unseen connection: stream coder pays ~constant slack per stream.

    BitNet measured ~4 bytes/tensor. Opening one stream per row pays M times.
    Block formats: axis choice can save bits; streams: flattening saves slack.
    """
    return {
        "slack_one_flat_stream": slack_per_stream,
        "slack_per_row_streams": m * slack_per_stream,
        "slack_per_col_streams": n * slack_per_stream,
        "slack_penalty_rows_vs_flat": (m - 1) * slack_per_stream,
        "slack_penalty_cols_vs_flat": (n - 1) * slack_per_stream,
    }


def novel_tax0_tile_fit(m: int, n: int, frames: Sequence[Tuple[int, ...]]) -> List[Dict]:
    """Which novel tax-0 assemblies tile a mode length exactly (practical grammar)."""
    hits: List[Dict] = []
    for parts in frames:
        q = sum(parts)
        if q <= 0:
            continue
        for mode_name, L in (("rows_m", m), ("cols_n", n)):
            if L % q == 0:
                hits.append(
                    {
                        "parts": list(parts),
                        "frame_q": q,
                        "mode": mode_name,
                        "length": L,
                        "n_tiles": L // q,
                        "packed_bits_per_tile": sum(bits(p) for p in parts),
                        "flat_bits_per_tile": bits(q),
                        "tax": split_tax(parts),
                    }
                )
    return hits


def gemm_alignment(frames: Sequence[Tuple[int, ...]], tiles: Sequence[int] = (16, 32, 64, 128, 256)) -> List[Dict]:
    """Catalog frames whose trit count divides or is divided by GEMM microtiles."""
    out: List[Dict] = []
    for parts in frames:
        q = sum(parts)
        for t in tiles:
            if q % t == 0 or t % q == 0:
                out.append(
                    {
                        "parts": list(parts),
                        "frame_q": q,
                        "gemm_tile": t,
                        "relation": "frame_multiple_of_tile" if q % t == 0 else "tile_multiple_of_frame",
                        "register_max_bits": max(bits(p) for p in parts),
                    }
                )
    return out


def kronecker_scale_cases() -> List[Dict]:
    """Factor pack vs expanded at increasing sizes (structure-aware upside)."""
    rng = np.random.default_rng(21)
    cases = []
    for a, b in (((8, 8), (8, 8)), ((16, 16), (16, 16)), ((32, 32), (16, 16)), ((64, 8), (8, 64))):
        A = rng.integers(-1, 2, size=a, dtype=np.int8)
        B = rng.integers(-1, 2, size=b, dtype=np.int8)
        for fmt in ("fmt_5_8", "fmt_41_65"):
            led = kron_ledger(A, B, fmt)
            cases.append(led)
    # Block-Kronecker: tiled repeats of a small core (grouped structure proxy)
    core = rng.integers(-1, 2, size=(16, 16), dtype=np.int8)
    # Simulate grouped: pack core once, store (core_bytes + index overhead)
    core_flat = theory_bytes_5_8(core.size)
    reps = 64  # 64 groups sharing structure
    expanded = reps * core.size
    cases.append(
        {
            "kind": "block_shared_core",
            "core_shape": [16, 16],
            "n_groups": reps,
            "pack_core_5_8": core_flat,
            "pack_expanded_5_8": theory_bytes_5_8(expanded),
            "delta": core_flat - theory_bytes_5_8(expanded),
            "factor_wins": True,
            "note": "shared core once vs expand; real models need valid factorisation",
        }
    )
    return cases


def probe_shape(name: str, m: int, n: int, tax0_frames: Sequence[Tuple[int, ...]]) -> Dict[str, Any]:
    t0 = time.time()
    ax = axis_bits(m, n)
    lay = layout_bytes(m, n)
    # Best practical 5_8 / 41_65 axis
    best_5 = "row_5_8" if lay["row_5_8"] <= lay["col_5_8"] else "col_5_8"
    best_41 = "row_41_65" if lay["row_41_65"] <= lay["col_41_65"] else "col_41_65"
    save_5 = abs(lay["row_5_8"] - lay["col_5_8"])
    save_41 = abs(lay["row_41_65"] - lay["col_41_65"])
    tiles = novel_tax0_tile_fit(m, n, tax0_frames)
    return {
        "name": name,
        "shape": [m, n],
        "area": m * n,
        "axis_bits": ax,
        "layout_bytes": lay,
        "best_5_8_axis": best_5,
        "best_41_65_axis": best_41,
        "bytes_saved_5_8_axis": save_5,
        "bytes_saved_41_65_axis": save_41,
        "vs_flat_5_8_row_delta": lay["row_5_8"] - lay["flat_5_8"],
        "vs_flat_41_row_delta": lay["row_41_65"] - lay["flat_41_65"],
        "stream_slack": stream_slack_tension(m, n),
        "novel_tax0_exact_tilers": tiles[:12],
        "n_novel_tilers": len(tiles),
        "secs": time.time() - t0,
    }


def run_probe() -> Dict[str, Any]:
    cat = build_catalog(max_trits=800, max_tax=0)
    deficit = {19, 53}
    tax0 = []
    for a in cat["assemblies"]:
        parts = tuple(a["parts"])
        if a["tax"] != 0 or len(parts) < 2:
            continue
        if not any(p in deficit for p in parts):
            continue
        tax0.append(parts)
    tax0.append((306, 306, 53))
    # unique
    tax0 = list({p: None for p in tax0}.keys())

    shapes = [probe_shape(name, m, n, tax0) for name, m, n in LLM_SHAPES]
    gemm = gemm_alignment(tax0)
    kron = kronecker_scale_cases()

    # Holonomy: axis delta as torus 2-form evaluation
    holonomy = [
        {
            "name": s["name"],
            "holonomy_bits": s["axis_bits"]["delta_bits"],
            "interpretation": (
                "row-then-flat vs col-then-flat path difference on mode torus; "
                "equals tax_rows - tax_cols"
            ),
        }
        for s in shapes
    ]

    total_save_41 = sum(s["bytes_saved_41_65_axis"] for s in shapes)
    return {
        "hypothesis": (
            "Larger models help only as targeted probes of axis choice and "
            "structure exploitation — not as bigger unstructured flat streams."
        ),
        "n_shapes": len(shapes),
        "shapes": shapes,
        "total_axis_bytes_saved_41_65_if_pick_best": total_save_41,
        "gemm_frame_alignments": gemm[:40],
        "n_gemm_alignments": len(gemm),
        "kronecker_scale": kron,
        "holonomy": holonomy,
        "creative_connections": [
            {
                "id": "torus_holonomy",
                "claim": (
                    "m*bits(n)-n*bits(m) is the exact holonomy of packing "
                    "around the mode torus (row path vs col path). Axis choice "
                    "is Law B chirality between orientations."
                ),
            },
            {
                "id": "stream_vs_block_tension",
                "claim": (
                    "Block formats: prefer low-tax axis. Stream formats: prefer "
                    "ONE stream (flatten) because slack ≈ constant per stream "
                    "(BitNet: ~4 B/tensor). Opposite layout advice by format family."
                ),
            },
            {
                "id": "tax0_tile_grammar",
                "claim": (
                    "Novel tax-0 assemblies are a tiling grammar for rectangular "
                    "modes: prefer mode lengths divisible by frame_q from the catalog."
                ),
            },
            {
                "id": "factor_tree_equals_decode_tree",
                "claim": (
                    "Kronecker-factor packing and hierarchical digit decode share "
                    "the same tree; a fused kernel can contract packed factors "
                    "without materialising A⊗B."
                ),
            },
            {
                "id": "gemm_microtile_resonance",
                "claim": (
                    "Search catalog for frames that resonate with GEMM tiles "
                    "(16/32/64/128/256) so packing blocks = compute blocks."
                ),
            },
            {
                "id": "bitnet_module_phase",
                "claim": (
                    "BitNet p0 varies by module type (q_proj vs v_proj). Combine "
                    "per-module adaptive coding with per-tensor axis choice — "
                    "two orthogonal levers METHOD_COMPARISON noted as unmeasured."
                ),
            },
        ],
    }


def selftest() -> int:
    # Holonomy identity without bits(MN)
    m, n = 7, 41
    d = axis_bits(m, n)
    assert d["delta_bits"] == tax_rows(m, n) - tax_cols(m, n)
    assert d["row_bits"] == m * bits(n)
    # fmt_5_8 row vs col can differ when remainders differ
    lay = layout_bytes(7, 41)
    assert lay["flat_5_8"] == theory_bytes_5_8(7 * 41)
    # Stream tension
    sl = stream_slack_tension(4096, 11008)
    assert sl["slack_per_row_streams"] == 4096 * 4
    print("SCALE_PROBE unit checks OK")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path(__file__).resolve().parent / "scale_probe_results.json"
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        print("Usage: python3 scale_probe.py [selftest|run [out.json]]")
        return 0
    if args and args[0] == "selftest":
        return selftest()
    dest = Path(args[1]) if len(args) > 1 else out
    if args and args[0] not in ("run", "selftest") and not args[0].endswith(".json"):
        # bare path?
        pass
    selftest()
    t0 = time.time()
    report = run_probe()
    report["wall_secs"] = time.time() - t0
    dest.write_text(json.dumps(report, indent=2) + "\n")

    # Human summary
    print(f"SCALE_PROBE wrote {dest} in {report['wall_secs']:.2f}s")
    print(f"hypothesis: {report['hypothesis']}")
    print(
        f"total axis bytes saved (sum over shapes, best 41_65 axis): "
        f"{report['total_axis_bytes_saved_41_65_if_pick_best']}"
    )
    print(f"{'name':<22} {'shape':>14} {'best':>5} {'save_bits':>10} {'save_41B':>9} {'tilers':>7}")
    for s in report["shapes"]:
        ax = s["axis_bits"]
        print(
            f"{s['name']:<22} {s['shape'][0]}x{s['shape'][1]:>6} "
            f"{ax['best_axis']:>5} {ax['bits_saved_vs_worse']:>10} "
            f"{s['bytes_saved_41_65_axis']:>9} {s['n_novel_tilers']:>7}"
        )
    print("\nCreative connections:")
    for c in report["creative_connections"]:
        print(f"  [{c['id']}] {c['claim'][:100]}...")
    # Kronecker highlight
    for k in report["kronecker_scale"]:
        if k.get("factor_wins") and "factor_bytes" in k:
            print(
                f"  kron {k['a_shape']}⊗{k['b_shape']} {k['fmt']}: "
                f"factor={k['factor_bytes']} flat={k['flat_bytes']} delta={k['delta']}"
            )
        elif k.get("kind") == "block_shared_core":
            print(
                f"  block-shared core: {k['pack_core_5_8']} vs expanded "
                f"{k['pack_expanded_5_8']} delta={k['delta']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
