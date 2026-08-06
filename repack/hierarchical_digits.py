#!/usr/bin/env python3
"""Hierarchical / Kronecker digit decode for Law C (exact integers).

Law C: decode rung k in the alphabet of rung k-1.
  - Rung 1 alphabet size 243 = 3^5  → 5 trits via LUT
  - Rung 2 (41 trits): 8 base-243 digits + 1 residual trit
  - Rung 3 (306 trits): nested via base-3^41 digits, then rung-2 decode;
    equivalently 61 base-243 digits + 1 residual (flat Law C).

Materialising a full 3^41 LUT is impossible. This module implements:
  1. Nested multi-index unpack (structural Kronecker / digit tower)
  2. Optional 243×243 Kronecker micro-table (10 trits per entry) for C2 accel

Endian: little-endian base-3, matching pack_ladder._pack_bigint_blocks
(x = sum t_i * 3^i). Round-trip against pack_ladder unpackers.
"""

from __future__ import annotations

import sys
from typing import List, Sequence, Tuple

import numpy as np

from pack_ladder import (
    BLOCK_41,
    BLOCK_306,
    BYTES_41,
    BYTES_306,
    _from_trits,
    _to_trits,
    container_bytes_for_r_trits,
    pack_41_65,
    pack_306_485,
    unpack_41_65,
    unpack_306_485,
)

POW3_41 = 3 ** 41
POW3_5 = 3 ** 5  # 243


def lut243() -> List[Tuple[int, ...]]:
    """243-entry table: value v in 0..242 → 5 little-endian trits in {0,1,2}."""
    table: List[Tuple[int, ...]] = []
    for v in range(243):
        x = v
        digits = []
        for _ in range(5):
            digits.append(x % 3)
            x //= 3
        table.append(tuple(digits))
    # pad 243..255 unused
    while len(table) < 256:
        table.append((0, 0, 0, 0, 0))
    return table


def build_kronecker_243x243(lut: List[Tuple[int, ...]] | None = None) -> List[Tuple[int, ...]]:
    """Kronecker-style 243×243 table: entry i = low243 + 243*high243 → 10 trits.

    Index = d0 + 243*d1 where d0,d1 are successive base-243 digits (LE).
    """
    if lut is None:
        lut = lut243()
    out: List[Tuple[int, ...]] = [()] * (243 * 243)
    for hi in range(243):
        thi = lut[hi]
        for lo in range(243):
            out[lo + 243 * hi] = lut[lo] + thi
    return out


def unpack_41_digit(x: int, lut: List[Tuple[int, ...]] | None = None) -> List[int]:
    """Unpack one 41-trit LE integer via 8×÷243 + 1 residual trit."""
    if lut is None:
        lut = lut243()
    out: List[int] = []
    for _ in range(8):
        d = x % 243
        x //= 243
        out.extend(lut[d])
    out.append(x % 3)  # residual trit; x should be in {0,1,2}
    return out


def unpack_41_kronecker(
    x: int,
    kron: List[Tuple[int, ...]] | None = None,
    lut: List[Tuple[int, ...]] | None = None,
) -> List[int]:
    """Unpack 41 trits using 4× (243⊗243) lookups + residual trit."""
    if kron is None:
        kron = build_kronecker_243x243(lut)
    out: List[int] = []
    for _ in range(4):
        # two base-243 digits as one index
        d0 = x % 243
        x //= 243
        d1 = x % 243
        x //= 243
        out.extend(kron[d0 + 243 * d1])
    out.append(x % 3)
    return out


def unpack_306_flat_digit(x: int, lut: List[Tuple[int, ...]] | None = None) -> List[int]:
    """Flat Law C: 61 base-243 digits + 1 residual trit (306 = 61*5+1)."""
    if lut is None:
        lut = lut243()
    out: List[int] = []
    for _ in range(61):
        d = x % 243
        x //= 243
        out.extend(lut[d])
    out.append(x % 3)
    return out


def unpack_306_hierarchical(x: int, lut: List[Tuple[int, ...]] | None = None) -> List[int]:
    """Hierarchical Law C: extract base-3^41 digits then decode each with C2.

    306 = 7*41 + 19, so seven full rung-2 digits plus a 19-trit remainder,
    matching the 486-frame split but keeping the flat 485-bit integer.
    """
    if lut is None:
        lut = lut243()
    out: List[int] = []
    for _ in range(7):
        digit = x % POW3_41
        x //= POW3_41
        out.extend(unpack_41_digit(digit, lut))
    # remainder: 19 trits
    rem = x
    for _ in range(19):
        out.append(rem % 3)
        rem //= 3
    return out


def unpack_41_65_hierarchical(data: bytes, n: int) -> np.ndarray:
    """Digit-nested unpack for fmt_41_65 storage (same bytes as pack_41_65)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    lut = lut243()
    full, rem = divmod(n, BLOCK_41)
    need = full * BYTES_41 + container_bytes_for_r_trits(rem)
    if len(data) < need:
        raise ValueError(f"need {need} bytes for n={n}, got {len(data)}")
    out = np.empty(n, dtype=np.int64)
    pos = 0
    base = 0
    for _ in range(full):
        x = int.from_bytes(data[pos : pos + BYTES_41], "little")
        pos += BYTES_41
        digits = unpack_41_digit(x, lut)
        out[base : base + BLOCK_41] = digits
        base += BLOCK_41
    if rem:
        nb = container_bytes_for_r_trits(rem)
        x = int.from_bytes(data[pos : pos + nb], "little")
        for i in range(rem):
            out[base + i] = x % 3
            x //= 3
    return _from_trits(out)


def unpack_306_485_hierarchical(data: bytes, n: int, mode: str = "nested") -> np.ndarray:
    """Digit-nested unpack for fmt_306_485. mode: 'nested' | 'flat'."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    lut = lut243()
    full, rem = divmod(n, BLOCK_306)
    need = full * BYTES_306 + container_bytes_for_r_trits(rem)
    if len(data) < need:
        raise ValueError(f"need {need} bytes for n={n}, got {len(data)}")
    decode = unpack_306_hierarchical if mode == "nested" else unpack_306_flat_digit
    out = np.empty(n, dtype=np.int64)
    pos = 0
    base = 0
    for _ in range(full):
        x = int.from_bytes(data[pos : pos + BYTES_306], "little")
        pos += BYTES_306
        digits = decode(x, lut)
        if len(digits) != BLOCK_306:
            raise AssertionError(f"expected {BLOCK_306} trits, got {len(digits)}")
        out[base : base + BLOCK_306] = digits
        base += BLOCK_306
    if rem:
        nb = container_bytes_for_r_trits(rem)
        x = int.from_bytes(data[pos : pos + nb], "little")
        for i in range(rem):
            out[base + i] = x % 3
            x //= 3
    return _from_trits(out)


def selftest() -> int:
    rng = np.random.default_rng(1)
    lut = lut243()
    kron = build_kronecker_243x243(lut)
    n_cases = 0

    # Unit: single 41-block digit vs serial
    for _ in range(32):
        w = rng.integers(-1, 2, size=41, dtype=np.int8)
        packed = pack_41_65(w)
        x = int.from_bytes(packed, "little")
        serial = [int(t) for t in _to_trits(w)]
        d = unpack_41_digit(x, lut)
        k = unpack_41_kronecker(x, kron, lut)
        assert d == serial, (d[:5], serial[:5])
        assert k == serial
        n_cases += 2

    # Unit: 306 nested and flat vs serial
    for _ in range(8):
        w = rng.integers(-1, 2, size=306, dtype=np.int8)
        packed = pack_306_485(w)
        x = int.from_bytes(packed, "little")
        serial = [int(t) for t in _to_trits(w)]
        assert unpack_306_flat_digit(x, lut) == serial
        assert unpack_306_hierarchical(x, lut) == serial
        n_cases += 2

    # Full unpackers vs pack_ladder
    for n in (0, 1, 40, 41, 42, 82, 305, 306, 307, 612, 1000):
        w = rng.integers(-1, 2, size=n, dtype=np.int8)
        p41 = pack_41_65(w)
        assert np.array_equal(unpack_41_65_hierarchical(p41, n), unpack_41_65(p41, n))
        p306 = pack_306_485(w)
        assert np.array_equal(
            unpack_306_485_hierarchical(p306, n, "nested"), unpack_306_485(p306, n)
        )
        assert np.array_equal(
            unpack_306_485_hierarchical(p306, n, "flat"), unpack_306_485(p306, n)
        )
        n_cases += 3

    print(f"HIERARCHICAL_DIGITS PASS n_cases={n_cases} kronecker_entries={len(kron)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    return selftest()


if __name__ == "__main__":
    # Allow `python3 hierarchical_digits.py` from repack/ or repo root.
    raise SystemExit(main())
