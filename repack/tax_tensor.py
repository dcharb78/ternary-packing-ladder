#!/usr/bin/env python3
"""Multi-linear packing tax form (Phase 2) — exact integers.

The 1-D Law B tax is the rank-1 slice of a form on mode block sizes:

  tax_rows(q0, q1) = q0 * bits(q1) - bits(q0 * q1)   # q0 rows of length q1
  tax_cols(q0, q1) = q1 * bits(q0) - bits(q0 * q1)   # q1 cols of length q0

Nested layouts (row containers inside a mode-1 assembly, etc.) are reported
explicitly. Enumerating (q0, q1) with tax_rows == 0 recovers repeated-part
assemblies from tax_graph as the rank-1 slice.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from tax_graph import (
    DEFAULT_ATOMS,
    bits,
    split_tax,
)

# Candidate mode block sizes: atoms + a few products that appear as frames.
MODE_SIZES: Tuple[int, ...] = tuple(
    sorted(
        {
            *DEFAULT_ATOMS,
            10,
            15,
            20,
            24,
            60,
            65,
            82,
            101,
            123,
            164,
            205,
            246,
            287,
            359,
            612,
            665,
        }
    )
)


def tax_rows(q0: int, q1: int) -> int:
    """Tax of packing q0 independent row-containers of length q1 vs flat q0*q1."""
    if q0 <= 0 or q1 <= 0:
        return 0
    return q0 * bits(q1) - bits(q0 * q1)


def tax_cols(q0: int, q1: int) -> int:
    """Tax of packing q1 independent col-containers of length q0 vs flat q0*q1."""
    if q0 <= 0 or q1 <= 0:
        return 0
    return q1 * bits(q0) - bits(q0 * q1)


def tax_nested_rows_in_cols(q0: int, q1: int, g1: int) -> int:
    """Nest: along mode-1, group q1 into parts of g1 (last rem allowed conceptually).

    For exact full tiles require q1 % g1 == 0. Tax vs flat q0*q1:
      packed = q0 * ( (q1/g1) * bits(g1) )
      flat   = bits(q0 * q1)
    """
    if q0 <= 0 or q1 <= 0 or g1 <= 0 or q1 % g1 != 0:
        raise ValueError("tax_nested_rows_in_cols requires q1 divisible by g1")
    n1 = q1 // g1
    packed = q0 * n1 * bits(g1)
    return packed - bits(q0 * q1)


def tax_form(q0: int, q1: int) -> Dict[str, int]:
    """Evaluate the multi-linear tax form at (q0, q1)."""
    return {
        "q0": q0,
        "q1": q1,
        "area": q0 * q1,
        "flat_bits": bits(q0 * q1),
        "tax_rows": tax_rows(q0, q1),
        "tax_cols": tax_cols(q0, q1),
        "best_axis_tax": min(tax_rows(q0, q1), tax_cols(q0, q1)),
        "bits_q0": bits(q0),
        "bits_q1": bits(q1),
    }


@dataclass(frozen=True)
class Frame2D:
    q0: int
    q1: int
    tax_rows: int
    tax_cols: int
    best: int
    area: int

    def to_dict(self) -> Dict:
        return asdict(self)


def enumerate_tax_frames(
    sizes: Sequence[int] = MODE_SIZES,
    max_tax: int = 1,
    max_area: int = 50000,
) -> List[Frame2D]:
    """Enumerate (q0,q1) with min(tax_rows,tax_cols) <= max_tax."""
    out: List[Frame2D] = []
    sizes = tuple(sorted({int(s) for s in sizes if int(s) > 0}))
    for q0 in sizes:
        for q1 in sizes:
            area = q0 * q1
            if area > max_area:
                continue
            tr = tax_rows(q0, q1)
            tc = tax_cols(q0, q1)
            best = min(tr, tc)
            if best <= max_tax:
                out.append(
                    Frame2D(
                        q0=q0,
                        q1=q1,
                        tax_rows=tr,
                        tax_cols=tc,
                        best=best,
                        area=area,
                    )
                )
    out.sort(key=lambda f: (f.best, f.area, f.q0, f.q1))
    return out


def verify_rank1_slice() -> None:
    """Rank-1: tax_rows(k, q) == split_tax((q,)*k)."""
    for q in (5, 19, 41, 53, 306):
        for k in (1, 2, 3, 7):
            assert tax_rows(k, q) == split_tax((q,) * k), (k, q)
            assert tax_cols(q, k) == split_tax((q,) * k), (q, k)


def build_catalog(max_tax: int = 1, max_area: int = 50000) -> Dict:
    verify_rank1_slice()
    frames = enumerate_tax_frames(max_tax=max_tax, max_area=max_area)
    zero = [f for f in frames if f.best == 0]
    # Novel vs pure 1-D: both axes > 1 and tax_rows != tax_cols (axis asymmetry)
    asymmetric = [f for f in zero if f.q0 > 1 and f.q1 > 1 and f.tax_rows != f.tax_cols]
    return {
        "max_tax": max_tax,
        "max_area": max_area,
        "n_frames": len(frames),
        "n_tax0": len(zero),
        "n_asymmetric_tax0": len(asymmetric),
        "rank1_verified": True,
        "zero_preview": [f.to_dict() for f in zero[:48]],
        "asymmetric_preview": [f.to_dict() for f in asymmetric[:32]],
        "examples": {
            "rows_7x41": tax_form(7, 41),
            "rows_2x306": tax_form(2, 306),
            "tile_41x5": tax_form(41, 5),
            "tile_5x41": tax_form(5, 41),
            "tile_41x41": tax_form(41, 41),
            "tile_306x5": tax_form(306, 5),
        },
    }


def write_catalog(path: Path, catalog: Dict | None = None) -> Path:
    if catalog is None:
        catalog = build_catalog()
    path.write_text(json.dumps(catalog, indent=2) + "\n")
    return path


def selftest() -> int:
    verify_rank1_slice()
    # Documented 486-frame as 7 rows of 41: tax_rows(7,41) == 1? 
    # 7*65 - bits(287). Earlier: 7x41 alone has tax 0; +19 makes 486.
    assert tax_rows(7, 41) == split_tax((41,) * 7) == 0
    assert tax_rows(2, 306) == 0  # part of 665 story before +53
    # Axis asymmetry: packing 5 rows of 41 vs 41 cols of 5
    a = tax_form(5, 41)
    b = tax_form(41, 5)
    assert a["tax_rows"] == tax_rows(5, 41)
    assert b["tax_cols"] == tax_cols(41, 5)

    cat = build_catalog(max_tax=1, max_area=20000)
    assert cat["n_tax0"] >= 1
    assert cat["rank1_verified"]

    # Nested example: 2 rows, each 2×41
    tn = tax_nested_rows_in_cols(2, 82, 41)
    assert tn == 2 * 2 * bits(41) - bits(164)

    print(
        f"TAX_TENSOR PASS frames={cat['n_frames']} tax0={cat['n_tax0']} "
        f"asymmetric_tax0={cat['n_asymmetric_tax0']} "
        f"tile_41x5_best={cat['examples']['tile_41x5']['best_axis_tax']}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path(__file__).resolve().parent / "tax_tensor_catalog.json"
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        print("Usage: python3 tax_tensor.py [selftest|catalog [path]]")
        return 0
    if not args or args[0] == "selftest":
        return selftest()
    if args[0] == "catalog":
        dest = Path(args[1]) if len(args) > 1 else out
        cat = build_catalog()
        write_catalog(dest, cat)
        print(
            f"wrote {dest} frames={cat['n_frames']} tax0={cat['n_tax0']} "
            f"asymmetric={cat['n_asymmetric_tax0']}"
        )
        return 0
    print(f"unknown command: {args[0]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
