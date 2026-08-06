#!/usr/bin/env python3
"""Per-mode Collatz / schedule phase vector (Phase 2) — exact integers.

Tracks an independent phase along each tensor mode using C_i =
(3**i).bit_length(). Descending/ascending can differ across modes; the
packer chooses row-block vs col-block sizes from the tax catalog
accordingly.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

from collatz_schedule import C_i, is_descending, tax0_frames_from_catalog
from frame_formats import pack_frame, unpack_frame, theory_bytes_frame
from mode_pack import pack_rows_framed, unpack_rows_framed, theory_bytes_rows_flat
from tax_graph import bits, split_tax
from tax_tensor import tax_rows, tax_cols


def mode_phase_vector(lengths: Sequence[int]) -> List[bool]:
    """Descending flag per mode at the end of that mode's length."""
    return [is_descending(max(int(L), 1)) for L in lengths]


def choose_axis(m: int, n: int) -> str:
    """Pick packing axis with lower multi-linear tax (rows vs cols)."""
    tr = tax_rows(m, n)
    tc = tax_cols(m, n)
    if tr < tc:
        return "rows"
    if tc < tr:
        return "cols"
    # tie: prefer descending mode
    desc = mode_phase_vector([m, n])
    if desc[0] and not desc[1]:
        return "rows"
    if desc[1] and not desc[0]:
        return "cols"
    return "rows"


def choose_row_parts(n: int, descending: bool, tax0: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
    if descending:
        for parts in tax0:
            if sum(parts) <= n and n % sum(parts) == 0:
                return parts
        for parts in tax0:
            if sum(parts) <= n:
                return parts
    for q in (306, 41, 5):
        if q <= n:
            return (q,)
    return (max(n, 1),)


def pack_mode_scheduled(mat: np.ndarray) -> Tuple[bytes, Dict]:
    """Pack matrix using per-mode phase to pick axis + frame parts."""
    mat = np.asarray(mat, dtype=np.int8)
    m, n = mat.shape
    tax0 = tax0_frames_from_catalog(max_trits=800)
    axis = choose_axis(m, n)
    desc = mode_phase_vector([m, n])
    meta: Dict = {"axis": axis, "descending": desc, "shape": [m, n]}

    if axis == "rows":
        parts = choose_row_parts(n, desc[1], tax0)
        meta["parts"] = list(parts)
        blob = pack_rows_framed(mat, parts, [0] * m)
        meta["nbytes"] = len(blob)
        meta["tax_form"] = tax_rows(m, n)
        return blob, meta

    # cols: pack transpose as rows
    parts = choose_row_parts(m, desc[0], tax0)
    meta["parts"] = list(parts)
    blob = pack_rows_framed(mat.T, parts, [0] * n)
    meta["nbytes"] = len(blob)
    meta["tax_form"] = tax_cols(m, n)
    return blob, meta


def unpack_mode_scheduled(data: bytes, meta: Dict) -> np.ndarray:
    m, n = meta["shape"]
    parts = tuple(meta["parts"])
    if meta["axis"] == "rows":
        return unpack_rows_framed(data, (m, n), parts, [0] * m)
    rec_T = unpack_rows_framed(data, (n, m), parts, [0] * n)
    return rec_T.T


def selftest() -> int:
    assert len(mode_phase_vector([41, 306])) == 2
    rng = np.random.default_rng(17)
    n_cases = 0
    for m, n in ((7, 41), (2, 306), (16, 64), (5, 53), (41, 5)):
        mat = rng.integers(-1, 2, size=(m, n), dtype=np.int8)
        blob, meta = pack_mode_scheduled(mat)
        rec = unpack_mode_scheduled(blob, meta)
        assert np.array_equal(rec, mat), (m, n, meta)
        # Axis choice is optimal for the tax form
        assert meta["axis"] == choose_axis(m, n)
        tr, tc = tax_rows(m, n), tax_cols(m, n)
        if meta["axis"] == "rows":
            assert meta["tax_form"] == tr
            assert tr <= tc
        else:
            assert meta["tax_form"] == tc
            assert tc <= tr
        n_cases += 1

    print(
        f"MODE_SCHEDULE PASS n_cases={n_cases} "
        f"example_7x41 axis={pack_mode_scheduled(np.zeros((7,41),dtype=np.int8))[1]['axis']}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
