#!/usr/bin/env python3
"""Kronecker-respecting tensor packing (Phase 2) — exact integers.

When a weight block is (approximately) a Kronecker product A ⊗ B, packing
the factors separately preserves the factor tree that hierarchical decode
also wants. This module:

  - builds W = kron(A, B) for ternary A, B (exact Kronecker of {-1,0,+1});
  - packs (pack(A), pack(B)) vs pack(flatten(W));
  - reports exact byte ledgers.

Note: kron of ternary matrices is ternary only if we use the algebraic
Kronecker with values in {-1,0,+1} (products stay in {-1,0,+1}).
"""

from __future__ import annotations

import sys
from typing import Dict, Sequence, Tuple

import numpy as np

from frame_formats import pack_486_frame, pack_665_frame, theory_bytes_486, theory_bytes_665
from pack_ladder import (
    pack_41_65,
    pack_5_8,
    pack_306_485,
    theory_bytes_41_65,
    theory_bytes_5_8,
    theory_bytes_306_485,
    unpack_41_65,
    unpack_5_8,
)


def ternary_kron(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Exact Kronecker product; entries stay in {-1,0,+1}."""
    A = np.asarray(A, dtype=np.int8)
    B = np.asarray(B, dtype=np.int8)
    return np.kron(A, B).astype(np.int8)


def pack_factors(
    A: np.ndarray,
    B: np.ndarray,
    fmt: str = "fmt_5_8",
) -> Tuple[bytes, bytes]:
    packers = {
        "fmt_5_8": pack_5_8,
        "fmt_41_65": pack_41_65,
        "fmt_306_485": pack_306_485,
        "fmt_486_frame": pack_486_frame,
        "fmt_665_frame": pack_665_frame,
    }
    if fmt not in packers:
        raise ValueError(fmt)
    p = packers[fmt]
    return p(A.reshape(-1)), p(B.reshape(-1))


def unpack_factors(
    a_blob: bytes,
    b_blob: bytes,
    a_shape: Tuple[int, int],
    b_shape: Tuple[int, int],
    fmt: str = "fmt_5_8",
) -> Tuple[np.ndarray, np.ndarray]:
    unpackers = {
        "fmt_5_8": unpack_5_8,
        "fmt_41_65": unpack_41_65,
    }
    if fmt not in unpackers:
        # reuse pack_ladder / frame via flatten size
        from pack_ladder import unpack_306_485
        from frame_formats import unpack_486_frame, unpack_665_frame

        unpackers = {
            "fmt_5_8": unpack_5_8,
            "fmt_41_65": unpack_41_65,
            "fmt_306_485": unpack_306_485,
            "fmt_486_frame": unpack_486_frame,
            "fmt_665_frame": unpack_665_frame,
        }
    u = unpackers[fmt]
    na = a_shape[0] * a_shape[1]
    nb = b_shape[0] * b_shape[1]
    A = u(a_blob, na).reshape(a_shape)
    B = u(b_blob, nb).reshape(b_shape)
    return A, B


def ledger(A: np.ndarray, B: np.ndarray, fmt: str = "fmt_5_8") -> Dict:
    W = ternary_kron(A, B)
    flat_n = int(W.size)
    a_blob, b_blob = pack_factors(A, B, fmt)
    theories = {
        "fmt_5_8": theory_bytes_5_8,
        "fmt_41_65": theory_bytes_41_65,
        "fmt_306_485": theory_bytes_306_485,
        "fmt_486_frame": theory_bytes_486,
        "fmt_665_frame": theory_bytes_665,
    }
    th = theories[fmt]
    factor_bytes = len(a_blob) + len(b_blob)
    flat_bytes = th(flat_n)
    return {
        "a_shape": list(A.shape),
        "b_shape": list(B.shape),
        "w_shape": list(W.shape),
        "fmt": fmt,
        "factor_bytes": factor_bytes,
        "flat_bytes": flat_bytes,
        "delta": factor_bytes - flat_bytes,
        "factor_wins": factor_bytes < flat_bytes,
        "a_nbytes": len(a_blob),
        "b_nbytes": len(b_blob),
    }


def selftest() -> int:
    rng = np.random.default_rng(11)
    n_cases = 0
    for ash, bsh in (((2, 2), (2, 2)), ((5, 5), (3, 3)), ((8, 5), (5, 8))):
        A = rng.integers(-1, 2, size=ash, dtype=np.int8)
        B = rng.integers(-1, 2, size=bsh, dtype=np.int8)
        W = ternary_kron(A, B)
        assert W.shape == (ash[0] * bsh[0], ash[1] * bsh[1])
        assert set(np.unique(W).tolist()).issubset({-1, 0, 1})

        for fmt in ("fmt_5_8", "fmt_41_65"):
            ab, bb = pack_factors(A, B, fmt)
            Ar, Br = unpack_factors(ab, bb, ash, bsh, fmt)
            assert np.array_equal(Ar, A) and np.array_equal(Br, B)
            Wr = ternary_kron(Ar, Br)
            assert np.array_equal(Wr, W)
            led = ledger(A, B, fmt)
            assert led["factor_bytes"] == len(ab) + len(bb)
            n_cases += 1

    # Structure-aware win is common: |A|+|B| << |A⊗B|
    A = rng.integers(-1, 2, size=(16, 16), dtype=np.int8)
    B = rng.integers(-1, 2, size=(16, 16), dtype=np.int8)
    led = ledger(A, B, "fmt_5_8")
    assert led["factor_wins"], led  # 2*256 vs 65536 trits

    print(
        f"KRONECKER_TENSOR_PACK PASS n_cases={n_cases} "
        f"example_16x16⊗16x16 factor_bytes={led['factor_bytes']} "
        f"flat_bytes={led['flat_bytes']} delta={led['delta']}"
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
