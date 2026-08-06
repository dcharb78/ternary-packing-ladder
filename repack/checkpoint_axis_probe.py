#!/usr/bin/env python3
"""Real-checkpoint axis + pad-to-tax0 probe (BitNet b1.58-style U8 packs).

Reads safetensors header + optional trit unpack. For each packed weight:

  - Infer logical matrix shape (out, in) from U8 [out/4, in] layout
  - Exact axis-choice bit/byte ledgers (row vs col fibers)
  - Pad-to-tax0 on each mode
  - Module-type aggregates (q_proj, v_proj, …) for the BitNet phase mixture

Optional ``--rt-sample N``: unpack N tensors and round-trip fmt_5_8 + best-axis
fiber pack to keep the exact-RT discipline.

Does not claim end-to-end engine speed.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from frame_formats import pack_frame, unpack_frame, theory_bytes_frame
from mode_pack import pack_rows_framed, unpack_rows_framed, theory_bytes_rows_flat
from pack_ladder import pack_5_8, unpack_5_8, theory_bytes_41_65, theory_bytes_5_8
from pad_to_tax0 import evaluate_length, tax0_frames
from scale_probe import axis_bits, layout_bytes, stream_slack_tension
from tax_graph import bits

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "bitnet" / "model.safetensors"

MODULE_RE = re.compile(
    r"(embed_tokens|lm_head|q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|mlp|attn)"
)


def load_header(path: Path) -> Tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode())
    return hdr, 8 + n


def module_type(name: str) -> str:
    m = MODULE_RE.findall(name)
    if not m:
        return "other"
    # last match is usually the projection
    return m[-1]


def logical_shape_u8_packed(shape: Sequence[int]) -> Tuple[int, int]:
    """BitNet: U8 [out/4, in] → logical (out, in)."""
    if len(shape) != 2:
        raise ValueError(f"expected 2-D U8 pack, got {shape}")
    out4, inn = int(shape[0]), int(shape[1])
    return out4 * 4, inn


def unpack_u8_trits(raw: np.ndarray) -> np.ndarray:
    lut = np.zeros((256, 4), dtype=np.int8)
    for b in range(256):
        for k in range(4):
            lut[b, k] = (b >> (2 * k)) & 3
    fields = lut[raw.reshape(-1)]
    if fields.max() > 2:
        raise ValueError("non-ternary field in packed U8")
    return (fields.astype(np.int8) - 1).reshape(-1)


def probe_tensor_shapes(
    path: Path,
    rt_sample: int = 0,
    max_tensors: Optional[int] = None,
) -> Dict[str, Any]:
    hdr, data0 = load_header(path)
    names = sorted(
        k
        for k in hdr
        if k != "__metadata__"
        and hdr[k].get("dtype") == "U8"
        and k.endswith(".weight")
    )
    if max_tensors is not None:
        names = names[:max_tensors]

    frames = tax0_frames()
    mm = np.memmap(path, dtype=np.uint8, mode="r") if rt_sample else None

    per: List[Dict[str, Any]] = []
    by_mod: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "n_tensors": 0,
            "trits": 0,
            "axis_bits_saved": 0,
            "bytes_saved_41": 0,
            "pad_wins_n": 0,
            "pad_wins_m": 0,
        }
    )
    totals = {
        "n_tensors": 0,
        "trits": 0,
        "axis_bits_saved": 0,
        "bytes_saved_41_axis": 0,
        "flat_5_8": 0,
        "best_axis_41": 0,
        "shipped_u8": 0,
        "pad_win_fibers": 0,
    }
    t0 = time.time()
    rt_left = rt_sample

    for idx, name in enumerate(names):
        meta = hdr[name]
        shape = meta["shape"]
        m, n = logical_shape_u8_packed(shape)
        area = m * n
        ax = axis_bits(m, n)
        lay = layout_bytes(m, n)
        best_41 = min(lay["row_41_65"], lay["col_41_65"])
        save_41 = abs(lay["row_41_65"] - lay["col_41_65"])
        pad_n = evaluate_length(n, frames)
        pad_m = evaluate_length(m, frames)
        mod = module_type(name)
        beg, end = meta["data_offsets"]
        shipped = end - beg

        entry = {
            "name": name,
            "module": mod,
            "u8_shape": list(shape),
            "logical_shape": [m, n],
            "trits": area,
            "shipped_u8_bytes": shipped,
            "axis_bits": ax,
            "layout_bytes": {
                "flat_5_8": lay["flat_5_8"],
                "flat_41_65": lay["flat_41_65"],
                "row_41_65": lay["row_41_65"],
                "col_41_65": lay["col_41_65"],
                "best_41_65": best_41,
                "save_41_65": save_41,
            },
            "pad_on_n": pad_n["best"],
            "pad_on_m": pad_m["best"],
            "stream_slack": stream_slack_tension(m, n),
        }

        if rt_left > 0 and mm is not None:
            raw = np.asarray(mm[data0 + beg : data0 + end])
            trits = unpack_u8_trits(raw)
            assert trits.size == area, (name, trits.size, area)
            mat = trits.reshape(m, n)
            # Round-trip flat 5_8
            b58 = pack_5_8(trits)
            assert np.array_equal(unpack_5_8(b58, area), trits)
            # Best-axis fiber pack with (5,) frames
            if ax["best_axis"] == "rows":
                blob = pack_rows_framed(mat, (5,), [0] * m)
                rec = unpack_rows_framed(blob, (m, n), (5,), [0] * m)
            else:
                blob = pack_rows_framed(mat.T, (5,), [0] * n)
                rec = unpack_rows_framed(blob, (n, m), (5,), [0] * n).T
            assert np.array_equal(rec, mat), name
            entry["rt_ok"] = True
            entry["rt_best_axis_5_bytes"] = len(blob)
            rt_left -= 1

        per.append(entry)
        totals["n_tensors"] += 1
        totals["trits"] += area
        totals["axis_bits_saved"] += ax["bits_saved_vs_worse"]
        totals["bytes_saved_41_axis"] += save_41
        totals["flat_5_8"] += lay["flat_5_8"]
        totals["best_axis_41"] += best_41
        totals["shipped_u8"] += shipped
        if pad_n["best"] and pad_n["best"]["delta_vs_41_65"] < 0:
            totals["pad_win_fibers"] += 1
        if pad_m["best"] and pad_m["best"]["delta_vs_41_65"] < 0:
            totals["pad_win_fibers"] += 1

        bm = by_mod[mod]
        bm["n_tensors"] += 1
        bm["trits"] += area
        bm["axis_bits_saved"] += ax["bits_saved_vs_worse"]
        bm["bytes_saved_41"] += save_41
        if pad_n["best"] and pad_n["best"]["delta_vs_41_65"] < 0:
            bm["pad_wins_n"] += 1
        if pad_m["best"] and pad_m["best"]["delta_vs_41_65"] < 0:
            bm["pad_wins_m"] += 1

        if idx % 30 == 0:
            print(
                f"  [{idx+1}/{len(names)}] {mod:<12} {m}x{n} "
                f"save41={save_41} ({time.time()-t0:.1f}s)",
                flush=True,
            )

    # Structure scan: look for exact repeated row/col blocks (grouped proxy)
    structure_hits: List[Dict] = []
    if rt_sample > 0 and mm is not None:
        # already spent samples on RT; skip heavy structure unless requested
        pass

    return {
        "path": str(path),
        "n_tensors": totals["n_tensors"],
        "totals": totals,
        "by_module": dict(by_mod),
        "tensors": per,
        "wall_secs": time.time() - t0,
        "rt_sample": rt_sample,
        "notes": [
            "Axis/pad ledgers are exact from logical shapes (U8 [out/4,in] → (out,in)).",
            "Byte savings are vs the worse fiber orientation / fmt_41_65, not vs shipped 2-bit.",
            "Stream slack tension uses 4 B/stream constant from BitNet measurement.",
        ],
    }


def structure_scan_tensor(mat: np.ndarray, max_groups: int = 16) -> Dict[str, Any]:
    """Cheap grouped-structure probe: does the matrix tile into identical blocks?"""
    m, n = mat.shape
    hits = []
    for g in range(2, min(max_groups, m) + 1):
        if m % g:
            continue
        h = m // g
        blocks = mat.reshape(g, h, n)
        if all(np.array_equal(blocks[0], blocks[i]) for i in range(1, g)):
            hits.append({"axis": "rows", "n_groups": g, "block": [h, n]})
    for g in range(2, min(max_groups, n) + 1):
        if n % g:
            continue
        w = n // g
        blocks = mat.reshape(m, g, w)
        if all(np.array_equal(blocks[:, 0, :], blocks[:, i, :]) for i in range(1, g)):
            hits.append({"axis": "cols", "n_groups": g, "block": [m, w]})
    return {"shape": [m, n], "identical_tilings": hits}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="path to model.safetensors",
    )
    ap.add_argument("--rt-sample", type=int, default=3, help="round-trip sample count")
    ap.add_argument("--max-tensors", type=int, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoint_axis_results.json",
    )
    ap.add_argument(
        "--structure-sample",
        type=int,
        default=5,
        help="unpack this many tensors for identical-tile structure scan",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    if not args.path.is_file():
        print(f"MISSING checkpoint: {args.path}", file=sys.stderr)
        print(
            "Download microsoft/bitnet-b1.58-2B-4T model.safetensors into "
            "repack/data/bitnet/ then re-run.",
            file=sys.stderr,
        )
        return 2

    print(f"probing {args.path} ...")
    report = probe_tensor_shapes(args.path, rt_sample=args.rt_sample, max_tensors=args.max_tensors)

    # Structure scan on a few tensors
    if args.structure_sample > 0:
        hdr, data0 = load_header(args.path)
        names = [t["name"] for t in report["tensors"][: args.structure_sample]]
        mm = np.memmap(args.path, dtype=np.uint8, mode="r")
        struct_reports = []
        for name in names:
            meta = hdr[name]
            beg, end = meta["data_offsets"]
            m, n = logical_shape_u8_packed(meta["shape"])
            trits = unpack_u8_trits(np.asarray(mm[data0 + beg : data0 + end]))
            mat = trits.reshape(m, n)
            sr = structure_scan_tensor(mat)
            sr["name"] = name
            sr["module"] = module_type(name)
            struct_reports.append(sr)
        report["structure_scan"] = struct_reports
        report["n_structure_hits"] = sum(len(s["identical_tilings"]) for s in struct_reports)

    args.out.write_text(json.dumps(report, indent=2) + "\n")
    t = report["totals"]
    print(f"\n=== checkpoint axis probe: {t['n_tensors']} tensors, {t['trits']:,} trits ===")
    print(f"shipped U8 bytes:     {t['shipped_u8']:,}")
    print(f"flat fmt_5_8 bytes:   {t['flat_5_8']:,}")
    print(f"best-axis fmt_41_65:  {t['best_axis_41']:,}")
    print(f"axis bits saved:      {t['axis_bits_saved']:,}")
    print(f"axis bytes saved 41:  {t['bytes_saved_41_axis']:,}")
    print(f"pad-win fiber modes:  {t['pad_win_fibers']}")
    print(f"wall: {report['wall_secs']:.1f}s  wrote {args.out}")
    print("\nby module:")
    for mod, bm in sorted(report["by_module"].items(), key=lambda kv: -kv[1]["trits"]):
        print(
            f"  {mod:<12} tensors={bm['n_tensors']:<4} trits={bm['trits']:>12,} "
            f"axis_bits={bm['axis_bits_saved']:>8,} save41B={bm['bytes_saved_41']:>6,}"
        )
    if "structure_scan" in report:
        print(f"\nstructure identical-tilings hits: {report.get('n_structure_hits', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
