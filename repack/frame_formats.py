#!/usr/bin/env python3
"""Catalog-driven Law B frame pack/unpack (exact integers).

Frames store a concatenation of surplus/deficit blocks rather than one flat
bigint. Density = sum(bits(part)) which may tax 0 or 1 over the flat container.

Documented:
  486-frame: 7×(41,65) + (19,31)  → 306 trits in 486 bits (tax 1)
  665-frame: 2×(306,485) + (53,85) → 665 trits in 1055 bits (tax 0)

Endian per block matches pack_ladder LE base-3.
"""

from __future__ import annotations

import sys
from typing import List, Sequence, Tuple

import numpy as np

from pack_ladder import (
    container_bytes_for_r_trits,
    _as_flat_weights,
    _from_trits,
    _to_trits,
)
from tax_graph import (
    LAW_B_486,
    LAW_B_665,
    Assembly,
    bits,
    split_tax,
)

# Frame ids
FMT_486_FRAME = "fmt_486_frame"
FMT_665_FRAME = "fmt_665_frame"

# Documented layouts (trit order = concatenation order of parts as written
# in the note: seven 41-blocks then 19; two 306-blocks then 53).
PARTS_486: Tuple[int, ...] = LAW_B_486  # 7×41 + 19
PARTS_665: Tuple[int, ...] = LAW_B_665  # 2×306 + 53


def theory_bits_frame(parts: Sequence[int], n: int) -> int:
    """Exact packed bit count for n trits under a repeating frame of `parts`."""
    frame_q = sum(parts)
    if frame_q <= 0:
        raise ValueError("empty parts")
    full, rem = divmod(n, frame_q)
    frame_bits = sum(bits(q) for q in parts)
    total = full * frame_bits
    if rem == 0:
        return total
    # Tail: greedily emit whole parts that fit, then a flat tail container.
    left = rem
    for q in parts:
        if left >= q:
            total += bits(q)
            left -= q
        else:
            break
    if left:
        total += bits(left)
    return total


def theory_bytes_frame(parts: Sequence[int], n: int) -> int:
    """Byte length: each full part uses ceil(bits/8); same for tail.

    We pack each part as an independent little-endian byte blob of
    container_bytes_for_r_trits(q) — matching how 486-frame stores 7×u128 + u32
    padded to whole bytes in practice.
    """
    frame_q = sum(parts)
    full, rem = divmod(n, frame_q)
    frame_bytes = sum(container_bytes_for_r_trits(q) for q in parts)
    total = full * frame_bytes
    if rem == 0:
        return total
    left = rem
    for q in parts:
        if left >= q:
            total += container_bytes_for_r_trits(q)
            left -= q
        else:
            break
    if left:
        total += container_bytes_for_r_trits(left)
    return total


def pack_frame(trits: np.ndarray, parts: Sequence[int]) -> bytes:
    """Pack weights into a repeating multi-block frame."""
    w = _as_flat_weights(trits)
    n = int(w.size)
    if n == 0:
        return b""
    t = _to_trits(w)
    frame_q = sum(parts)
    out = bytearray()
    base = 0
    while base < n:
        left = n - base
        for q in parts:
            if left <= 0:
                break
            take = q if left >= q else left
            x = 0
            p = 1
            for i in range(take):
                x += int(t[base + i]) * p
                p *= 3
            nb = container_bytes_for_r_trits(take)
            out += x.to_bytes(nb, "little")
            base += take
            left -= take
            if take < q:
                # partial last part — done
                left = 0
                break
    return bytes(out)


def unpack_frame(data: bytes, n: int, parts: Sequence[int]) -> np.ndarray:
    """Unpack n weights from a repeating multi-block frame."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    need = theory_bytes_frame(parts, n)
    if len(data) < need:
        raise ValueError(f"frame: need {need} bytes for n={n}, got {len(data)}")
    out = np.empty(n, dtype=np.int64)
    pos = 0
    base = 0
    while base < n:
        left = n - base
        for q in parts:
            if left <= 0:
                break
            take = q if left >= q else left
            nb = container_bytes_for_r_trits(take)
            x = int.from_bytes(data[pos : pos + nb], "little")
            pos += nb
            for i in range(take):
                out[base + i] = x % 3
                x //= 3
            base += take
            left -= take
            if take < q:
                left = 0
                break
    return _from_trits(out)


def pack_486_frame(trits: np.ndarray) -> bytes:
    return pack_frame(trits, PARTS_486)


def unpack_486_frame(data: bytes, n: int) -> np.ndarray:
    return unpack_frame(data, n, PARTS_486)


def pack_665_frame(trits: np.ndarray) -> bytes:
    return pack_frame(trits, PARTS_665)


def unpack_665_frame(data: bytes, n: int) -> np.ndarray:
    return unpack_frame(data, n, PARTS_665)


def theory_bytes_486(n: int) -> int:
    return theory_bytes_frame(PARTS_486, n)


def theory_bytes_665(n: int) -> int:
    return theory_bytes_frame(PARTS_665, n)


def pack_from_assembly(trits: np.ndarray, asm: Assembly) -> bytes:
    """Pack using an Assembly's parts in documented / catalog order.

    Catalog stores canonical (sorted) parts; for decode-parallel frames the
    caller should pass an ordered parts tuple. Here we use asm.parts as-is.
    """
    return pack_frame(trits, asm.parts)


def selftest() -> int:
    # Tax identities
    assert split_tax(PARTS_486) == 1
    assert sum(PARTS_486) == 306
    assert sum(bits(q) for q in PARTS_486) == 486
    assert split_tax(PARTS_665) == 0
    assert sum(PARTS_665) == 665
    assert sum(bits(q) for q in PARTS_665) == 1055

    rng = np.random.default_rng(2)
    n_cases = 0
    sizes = [0, 1, 19, 41, 53, 305, 306, 307, 664, 665, 666, 1330, 2000]
    for n in sizes:
        w = rng.integers(-1, 2, size=n, dtype=np.int8)
        for pack, unpack, theory, name in (
            (pack_486_frame, unpack_486_frame, theory_bytes_486, FMT_486_FRAME),
            (pack_665_frame, unpack_665_frame, theory_bytes_665, FMT_665_FRAME),
        ):
            blob = pack(w)
            rec = unpack(blob, n)
            if not np.array_equal(rec, w):
                bad = np.flatnonzero(rec != w)
                i = int(bad[0]) if bad.size else -1
                raise AssertionError(f"RT fail {name} n={n} i={i}")
            if len(blob) != theory(n):
                raise AssertionError(
                    f"size fail {name} n={n}: {len(blob)} != {theory(n)}"
                )
            n_cases += 1

    # Full-block density display (float for print only)
    b486 = theory_bits_frame(PARTS_486, 306)
    b665 = theory_bits_frame(PARTS_665, 665)
    print(
        f"FRAME_FORMATS PASS n_cases={n_cases} "
        f"486_bpw={b486/306:.6f} 665_bpw={b665/665:.6f}"
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
