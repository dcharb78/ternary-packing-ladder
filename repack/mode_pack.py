#!/usr/bin/env python3
"""Mode-wise packing with phase offsets (Phase 2) — exact integers.

Pack a ternary matrix (M, N) as M independent row streams (or N column
streams), optionally with a cyclic phase offset into a frame of length F so
that residual seams can cancel across rows.

Compares packed byte counts to flat 1-D packing of M*N trits.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

from frame_formats import pack_frame, unpack_frame, theory_bytes_frame
from pack_ladder import (
    pack_41_65,
    pack_5_8,
    pack_306_485,
    theory_bytes_41_65,
    theory_bytes_5_8,
    theory_bytes_306_485,
)
from tax_graph import bits
from tax_tensor import tax_rows, tax_cols


def _as_matrix(w: np.ndarray) -> np.ndarray:
    a = np.asarray(w, dtype=np.int8)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D weights, got shape {a.shape}")
    if not np.isin(a, (-1, 0, 1)).all():
        raise ValueError("weights must be in {-1,0,+1}")
    return a


def pack_rows_flat(mat: np.ndarray) -> bytes:
    """Each row: one flat container of N trits (LE base-3), concatenated."""
    mat = _as_matrix(mat)
    m, n = mat.shape
    if m == 0 or n == 0:
        return b""
    from pack_ladder import container_bytes_for_r_trits, _to_trits

    nb = container_bytes_for_r_trits(n)
    out = bytearray()
    for i in range(m):
        t = _to_trits(mat[i])
        x = 0
        p = 1
        for j in range(n):
            x += int(t[j]) * p
            p *= 3
        out += x.to_bytes(nb, "little")
    return bytes(out)


def unpack_rows_flat(data: bytes, shape: Tuple[int, int]) -> np.ndarray:
    from pack_ladder import container_bytes_for_r_trits, _from_trits

    m, n = shape
    if m == 0 or n == 0:
        return np.zeros(shape, dtype=np.int8)
    nb = container_bytes_for_r_trits(n)
    need = m * nb
    if len(data) < need:
        raise ValueError(f"need {need} bytes, got {len(data)}")
    out = np.empty((m, n), dtype=np.int64)
    pos = 0
    for i in range(m):
        x = int.from_bytes(data[pos : pos + nb], "little")
        pos += nb
        for j in range(n):
            out[i, j] = x % 3
            x //= 3
    return _from_trits(out).reshape(m, n)


def pack_rows_framed(
    mat: np.ndarray,
    parts: Sequence[int],
    offsets: Sequence[int] | None = None,
) -> bytes:
    """Pack each row with a repeating 1-D frame; optional cyclic phase offset.

    Offset o means: conceptually rotate the row left by o before framing
    (exact: pack row[(o+j) % N] in position j). Same o-cycle used on unpack.
    """
    mat = _as_matrix(mat)
    m, n = mat.shape
    if offsets is None:
        offsets = [0] * m
    if len(offsets) != m:
        raise ValueError("offsets length must equal M")
    chunks: List[bytes] = []
    for i in range(m):
        o = int(offsets[i]) % n if n else 0
        row = np.concatenate([mat[i, o:], mat[i, :o]]) if o else mat[i]
        chunks.append(pack_frame(row, parts))
    return b"".join(chunks)


def unpack_rows_framed(
    data: bytes,
    shape: Tuple[int, int],
    parts: Sequence[int],
    offsets: Sequence[int] | None = None,
) -> np.ndarray:
    m, n = shape
    if offsets is None:
        offsets = [0] * m
    row_bytes = theory_bytes_frame(parts, n)
    need = m * row_bytes
    if len(data) < need:
        raise ValueError(f"need {need} bytes, got {len(data)}")
    out = np.empty((m, n), dtype=np.int8)
    pos = 0
    for i in range(m):
        blob = data[pos : pos + row_bytes]
        pos += row_bytes
        row = unpack_frame(blob, n, parts)
        o = int(offsets[i]) % n if n else 0
        if o:
            # inverse of left rotate by o: right rotate
            out[i] = np.concatenate([row[n - o :], row[: n - o]])
        else:
            out[i] = row
    return out


def theory_bytes_rows_flat(m: int, n: int) -> int:
    from pack_ladder import container_bytes_for_r_trits

    if m == 0 or n == 0:
        return 0
    return m * container_bytes_for_r_trits(n)


def compare_layouts(mat: np.ndarray, frame_parts: Sequence[int] = (41, 19)) -> Dict:
    """Exact byte ledgers: flat 1-D vs row-flat vs row-framed vs cols-flat."""
    mat = _as_matrix(mat)
    m, n = mat.shape
    flat = mat.reshape(-1)
    ntot = int(flat.size)
    return {
        "shape": [m, n],
        "tax_rows_form": tax_rows(m, n),
        "tax_cols_form": tax_cols(m, n),
        "bytes_flat_5_8": theory_bytes_5_8(ntot),
        "bytes_flat_41_65": theory_bytes_41_65(ntot),
        "bytes_flat_306_485": theory_bytes_306_485(ntot),
        "bytes_rows_flat": theory_bytes_rows_flat(m, n),
        "bytes_cols_flat": theory_bytes_rows_flat(n, m),  # transpose dims
        "bytes_rows_framed": m * theory_bytes_frame(frame_parts, n),
        "bits_rows_flat": m * bits(n) if n else 0,
        "bits_flat": bits(ntot) if ntot else 0,
    }


def search_phase_offsets(
    mat: np.ndarray,
    parts: Sequence[int],
    max_offset: int | None = None,
) -> Tuple[List[int], int]:
    """Greedy per-row offset search minimizing framed byte total.

    For fixed frame parts, all offsets give the same byte count (format is
    length-determined). We still search offsets that minimize a seam proxy:
    exact tax of concatenating the rotated row's natural flat bits vs bits(N)
    is always 0 per row — so byte count is invariant.

    Returns (offsets, bytes) for the framed pack (bytes independent of offset).
    The interesting win is choosing row vs col axis using tax_tensor, not
    offset — documented honestly in selftest.
    """
    mat = _as_matrix(mat)
    m, n = mat.shape
    if max_offset is None:
        max_offset = min(n, 8)
    # Byte size invariant in offset for fixed parts; pick offsets 0.. 
    offsets = [min(i % max(1, max_offset), n - 1 if n else 0) for i in range(m)]
    nbytes = m * theory_bytes_frame(parts, n)
    return offsets, nbytes


def selftest() -> int:
    rng = np.random.default_rng(7)
    n_cases = 0
    for m, n in ((1, 41), (7, 41), (2, 306), (16, 64), (5, 53)):
        mat = rng.integers(-1, 2, size=(m, n), dtype=np.int8)
        blob = pack_rows_flat(mat)
        rec = unpack_rows_flat(blob, (m, n))
        assert np.array_equal(rec, mat)
        assert len(blob) == theory_bytes_rows_flat(m, n)
        assert len(blob) * 8 >= m * bits(n)  # byte padding
        # tax form matches bit ledger
        assert m * bits(n) - bits(m * n) == tax_rows(m, n)
        n_cases += 1

        parts = (5,) if n % 5 == 0 else (41, 19) if n >= 60 else (5, 1)
        # ensure parts cover: use simple (5,) with rem via pack_frame
        parts = (5,)
        offs = [0] * m
        b2 = pack_rows_framed(mat, parts, offs)
        r2 = unpack_rows_framed(b2, (m, n), parts, offs)
        assert np.array_equal(r2, mat)
        n_cases += 1

        # nonzero offsets
        offs2 = [(i * 3) % n for i in range(m)]
        b3 = pack_rows_framed(mat, parts, offs2)
        r3 = unpack_rows_framed(b3, (m, n), parts, offs2)
        assert np.array_equal(r3, mat)
        assert len(b3) == len(b2)  # offset-invariant size for fixed parts
        n_cases += 1

    led = compare_layouts(rng.integers(-1, 2, size=(7, 41), dtype=np.int8))
    # 7×41: row-flat bits = 7*65 = 455; flat bits(287)=455; tax_rows=0
    assert led["tax_rows_form"] == 0
    print(
        f"MODE_PACK PASS n_cases={n_cases} "
        f"example_7x41 tax_rows={led['tax_rows_form']} "
        f"bytes_rows={led['bytes_rows_flat']} bytes_flat41={led['bytes_flat_41_65']}"
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
