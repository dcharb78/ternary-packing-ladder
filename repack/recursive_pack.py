#!/usr/bin/env python3
"""Recursive / self-similar packing operators via convergent recurrence.

Continued-fraction denominators of log2(3) obey a matrix recurrence; we mirror
that as a blocked composition:

  P_{n+1}  ≈  blocked concatenation of copies of P_n and P_{n-1}
              with an exact Law-B tax correction in the size ledger.

Rung sequence used here (surplus family): Q = (1, 5, 41, 306, …)
(1 is the residual trit / trivial alphabet; 5, 41, 306 are certified rungs).

This module does NOT claim density better than the flat rung or the stream
coder — it records exact byte/bit ledgers and round-trips.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from frame_formats import pack_frame, unpack_frame, theory_bytes_frame
from tax_graph import bits

# Surplus rung trit counts (extendable).
RUNGS: Tuple[int, ...] = (1, 5, 41, 306)


@dataclass(frozen=True)
class RecursiveSpec:
    """How to build P_n from earlier operators."""

    level: int
    parts: Tuple[int, ...]  # leaf block sizes in decode order
    total_trits: int
    packed_bits: int
    flat_bits: int
    tax: int
    note: str


def rung_q(level: int) -> int:
    if level < 0 or level >= len(RUNGS):
        raise ValueError(f"level {level} out of range")
    return RUNGS[level]


def compose_level(level: int) -> RecursiveSpec:
    """Build P_level as a Kronecker-like blocked composition.

    Level 0: single trit.
    Level 1: rung (5,8).
    Level 2: rung (41,65) = 8×5 + 1  (digit nesting identity).
    Level 3: 7×41 + 19 — Law B frame composition of P_2 with a remainder
             (19 is not a prior P_n; tax correction recorded exactly).

    The 7 and 19 come from 306 = 7*41 + 19 (same split as the 486-frame).
    """
    if level == 0:
        parts = (1,)
    elif level == 1:
        parts = (5,)
    elif level == 2:
        # 41 = 8*5 + 1 — structural nesting of P_1 in P_2
        parts = (5,) * 8 + (1,)
    elif level == 3:
        # 306 = 7*41 + 19 — composition of P_2 blocks + remainder
        parts = (41,) * 7 + (19,)
    else:
        raise ValueError(f"compose_level only defined for 0..3, got {level}")

    total = sum(parts)
    packed = sum(bits(q) for q in parts)
    flat = bits(total)
    tax = packed - flat
    return RecursiveSpec(
        level=level,
        parts=parts,
        total_trits=total,
        packed_bits=packed,
        flat_bits=flat,
        tax=tax,
        note=f"P_{level} parts={parts}",
    )


def pack_recursive(trits: np.ndarray, level: int) -> bytes:
    spec = compose_level(level)
    return pack_frame(trits, spec.parts)


def unpack_recursive(data: bytes, n: int, level: int) -> np.ndarray:
    spec = compose_level(level)
    return unpack_frame(data, n, spec.parts)


def ledger(level: int, n: int) -> Dict:
    """Exact size ledger vs flat rung container for n trits."""
    spec = compose_level(level)
    frame_bytes = theory_bytes_frame(spec.parts, n)
    # Flat container at the same total block size if n is multiple of rung.
    q = spec.total_trits
    full, rem = divmod(n, q)
    flat_bits = full * bits(q) + (bits(rem) if rem else 0)
    flat_bytes = (flat_bits + 7) // 8 if n else 0
    # Prefer exact byte theory from container_bytes when single flat block:
    from pack_ladder import container_bytes_for_r_trits

    flat_bytes_exact = full * container_bytes_for_r_trits(q) + (
        container_bytes_for_r_trits(rem) if rem else 0
    )
    return {
        "level": level,
        "n": n,
        "parts": list(spec.parts),
        "tax_per_frame": spec.tax,
        "frame_bytes": frame_bytes,
        "flat_bytes": flat_bytes_exact,
        "byte_delta": frame_bytes - flat_bytes_exact,
        "packed_bits_per_frame": spec.packed_bits,
        "flat_bits_per_frame": spec.flat_bits,
    }


def selftest() -> int:
    s2 = compose_level(2)
    assert s2.total_trits == 41
    assert s2.tax == sum(bits(q) for q in s2.parts) - bits(41)
    # 8×8 + 2 - 65 = 64+2-65 = 1? bits(5)=8, bits(1)=2, bits(41)=65
    assert s2.packed_bits == 8 * 8 + 2
    assert s2.flat_bits == 65
    assert s2.tax == 1  # digit-nesting assembly pays 1 vs flat 41

    s3 = compose_level(3)
    assert s3.total_trits == 306
    assert s3.packed_bits == 486
    assert s3.tax == 1

    rng = np.random.default_rng(3)
    n_cases = 0
    for level in (1, 2, 3):
        spec = compose_level(level)
        for n in (0, 1, spec.total_trits, spec.total_trits + 3, 2 * spec.total_trits):
            w = rng.integers(-1, 2, size=n, dtype=np.int8)
            blob = pack_recursive(w, level)
            rec = unpack_recursive(blob, n, level)
            assert np.array_equal(rec, w), (level, n)
            assert len(blob) == theory_bytes_frame(spec.parts, n)
            n_cases += 1

    # Level-2 recursive vs flat 41_65 density comparison (display)
    led = ledger(3, 306)
    print(
        f"RECURSIVE_PACK PASS n_cases={n_cases} "
        f"P3_tax={s3.tax} P3_bytes={led['frame_bytes']} "
        f"flat_bytes={led['flat_bytes']} delta={led['byte_delta']}"
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
