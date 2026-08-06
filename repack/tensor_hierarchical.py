#!/usr/bin/env python3
"""Hierarchical decoding that never flattens the whole tensor (Phase 2).

Tile a matrix into row-blocks of height 1 (or q0) and apply Law-C digit
decode per tile, keeping intermediate digit streams as row tensors.

Exact round-trip against packing each row as fmt_41_65 / digit unpack.
"""

from __future__ import annotations

import sys
from typing import List, Sequence, Tuple

import numpy as np

from hierarchical_digits import lut243, unpack_41_digit, unpack_41_65_hierarchical
from pack_ladder import (
    BLOCK_41,
    BYTES_41,
    container_bytes_for_r_trits,
    pack_41_65,
    unpack_41_65,
    _from_trits,
    _to_trits,
)


def pack_matrix_row_tiles_41(mat: np.ndarray) -> bytes:
    """Pack each row independently with fmt_41_65 (no global flatten)."""
    mat = np.asarray(mat, dtype=np.int8)
    if mat.ndim != 2:
        raise ValueError("expected 2-D")
    chunks = [pack_41_65(mat[i]) for i in range(mat.shape[0])]
    return b"".join(chunks)


def unpack_matrix_row_tiles_41_flat(data: bytes, shape: Tuple[int, int]) -> np.ndarray:
    """Baseline: per-row pack_ladder unpack_41_65."""
    m, n = shape
    row_bytes = (n // BLOCK_41) * BYTES_41 + container_bytes_for_r_trits(n % BLOCK_41)
    out = np.empty((m, n), dtype=np.int8)
    pos = 0
    for i in range(m):
        blob = data[pos : pos + row_bytes]
        pos += row_bytes
        out[i] = unpack_41_65(blob, n)
    return out


def unpack_matrix_row_tiles_41_hierarchical(
    data: bytes, shape: Tuple[int, int]
) -> np.ndarray:
    """Per-row hierarchical digit unpack — never flattens across rows."""
    m, n = shape
    row_bytes = (n // BLOCK_41) * BYTES_41 + container_bytes_for_r_trits(n % BLOCK_41)
    out = np.empty((m, n), dtype=np.int8)
    pos = 0
    for i in range(m):
        blob = data[pos : pos + row_bytes]
        pos += row_bytes
        out[i] = unpack_41_65_hierarchical(blob, n)
    return out


def extract_row_digit_tensor(
    mat: np.ndarray,
) -> Tuple[np.ndarray, List[bytes]]:
    """For each full 41-block along each row, return base-243 digit rows.

    Returns (digit_tensor of shape (M, n_full, 8) with values 0..242,
    and list of packed row blobs for RT checks).
    Intermediate digit tensor is the non-flattened hierarchical state.
    """
    mat = np.asarray(mat, dtype=np.int8)
    m, n = mat.shape
    n_full = n // BLOCK_41
    digits = np.zeros((m, n_full, 8), dtype=np.uint8)
    lut = lut243()
    blobs: List[bytes] = []
    for i in range(m):
        blob = pack_41_65(mat[i])
        blobs.append(blob)
        pos = 0
        for b in range(n_full):
            x = int.from_bytes(blob[pos : pos + BYTES_41], "little")
            pos += BYTES_41
            # record 8 base-243 digits (LE)
            for dg in range(8):
                digits[i, b, dg] = x % 243
                x //= 243
            # residual trit ignored in digit tensor (kept in blob only)
            _ = x
    return digits, blobs


def reconstruct_from_digits_and_residuals(
    digits: np.ndarray,
    mat: np.ndarray,
) -> np.ndarray:
    """Rebuild weights from digit tensor + original for residuals/tails.

    Full reconstruction path uses hierarchical unpack of packed rows;
    this checks that digit slices match LUT expansion of the source.
    """
    lut = lut243()
    m, n_full, _ = digits.shape
    out_blocks = []
    for i in range(m):
        row_trits: List[int] = []
        for b in range(n_full):
            for dg in range(8):
                row_trits.extend(lut[int(digits[i, b, dg])])
            # residual trit from original block
            block = mat[i, b * BLOCK_41 : (b + 1) * BLOCK_41]
            t = _to_trits(block)
            # LE value residual after 8 digits = t[40]
            row_trits.append(int(t[40]))
        out_blocks.append(row_trits)
    # Only validates full-block prefix; caller compares prefix.
    return np.array(out_blocks, dtype=np.int64)


def selftest() -> int:
    rng = np.random.default_rng(13)
    n_cases = 0
    for m, n in ((1, 41), (4, 82), (7, 41), (3, 100)):
        mat = rng.integers(-1, 2, size=(m, n), dtype=np.int8)
        blob = pack_matrix_row_tiles_41(mat)
        r0 = unpack_matrix_row_tiles_41_flat(blob, (m, n))
        r1 = unpack_matrix_row_tiles_41_hierarchical(blob, (m, n))
        assert np.array_equal(r0, mat)
        assert np.array_equal(r1, mat)
        n_cases += 2

        if n >= BLOCK_41:
            digits, blobs = extract_row_digit_tensor(mat)
            assert digits.shape[0] == m
            # Digit expansion matches first 40 trits of each block
            lut = lut243()
            for i in range(m):
                for b in range(n // BLOCK_41):
                    expanded: List[int] = []
                    for dg in range(8):
                        expanded.extend(lut[int(digits[i, b, dg])])
                    block = mat[i, b * BLOCK_41 : (b + 1) * BLOCK_41]
                    t = [int(x) for x in _to_trits(block)]
                    assert expanded == t[:40]
            n_cases += 1

    print(f"TENSOR_HIERARCHICAL PASS n_cases={n_cases}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
