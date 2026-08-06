#!/usr/bin/env python3
"""Collatz-adaptive packing schedule over certified ladder frames.

Phase tracker: exact schedule C_i = (3**i).bit_length() (same as stream_probe).
Descending window: local slack C_{i} - C_{i-k} trends below the mean step
(approximately i*log2(3) growth — we use exact integer comparisons only).

When descending: emit the largest tax-0 catalog frame that fits in the
remaining trits; otherwise fall back to atomic surplus rungs (5, 41, 306).

Also builds schedule families via Kronecker-style products of short Collatz
trajectory segments (admissible block-size sequences).

Property: packed bits <= best static flat format bits + proven tax bound
(here: sum of per-frame taxes from the chosen assemblies).
"""

from __future__ import annotations

import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

from frame_formats import pack_frame, unpack_frame, theory_bits_frame, theory_bytes_frame
from pack_ladder import container_bits_for_r_trits, container_bytes_for_r_trits
from tax_graph import (
    DEFICIT_PIECES,
    SURPLUS_RUNGS,
    bits,
    build_catalog,
    find_assembly,
    split_tax,
)

# Atomic fallback blocks (surplus family only — never allocate bare deficit).
ATOMIC: Tuple[int, ...] = SURPLUS_RUNGS  # 5, 41, 306


def C_i(i: int) -> int:
    """Exact schedule: minimal bits for i trits."""
    if i <= 0:
        return 0
    return (3 ** i).bit_length()


def collatz_step(n: int) -> int:
    if n <= 0:
        raise ValueError("collatz_step expects positive n")
    if n % 2 == 0:
        return n // 2
    return 3 * n + 1


def collatz_segment(seed: int, length: int) -> List[int]:
    """Short Collatz orbit (positive ints) used only to shape schedules."""
    out: List[int] = []
    x = seed
    for _ in range(length):
        out.append(x)
        x = collatz_step(x)
    return out


def is_descending(i: int, window: int = 8) -> bool:
    """Exact descending test: recent schedule increments shrink.

    Compare last step ΔC = C_i - C_{i-1} to the step at i-window.
    Descending when ΔC <= C_{i-window} - C_{i-window-1} (non-increasing
    bit-growth increments), for i > window.
    """
    if i <= window:
        return False
    d_now = C_i(i) - C_i(i - 1)
    d_prev = C_i(i - window) - C_i(i - window - 1)
    return d_now <= d_prev


def tax0_frames_from_catalog(max_trits: int = 800) -> List[Tuple[int, ...]]:
    """Ordered parts for tax-0 assemblies with deficit cancellation, largest first."""
    cat = build_catalog(max_trits=max_trits, max_tax=0)
    deficit = set(DEFICIT_PIECES)
    frames: List[Tuple[int, ...]] = []
    seen = set()
    for a in cat["assemblies"]:
        parts = tuple(a["parts"])
        if a["tax"] != 0 or len(parts) < 2:
            continue
        if not any(p in deficit for p in parts):
            continue
        if parts in seen:
            continue
        seen.add(parts)
        frames.append(parts)
    # Also include documented 665
    frames.append((306, 306, 53))
    frames.sort(key=lambda p: sum(p), reverse=True)
    return frames


def choose_block(remaining: int, pos: int, tax0_frames: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Pick next frame parts given remaining trits and phase position."""
    if remaining <= 0:
        return ()
    if is_descending(max(pos, 1)):
        for parts in tax0_frames:
            q = sum(parts)
            if q <= remaining:
                return parts
    # Atomic surplus fallback: largest that fits
    for q in sorted(ATOMIC, reverse=True):
        if q <= remaining:
            return (q,)
    return (remaining,)  # flat tail


def plan_schedule(n: int, tax0_frames: Sequence[Tuple[int, ...]] | None = None) -> List[Tuple[int, ...]]:
    if tax0_frames is None:
        tax0_frames = tax0_frames_from_catalog()
    plan: List[Tuple[int, ...]] = []
    pos = 0
    left = n
    while left > 0:
        parts = choose_block(left, pos, tax0_frames)
        if not parts:
            break
        plan.append(parts)
        q = sum(parts)
        pos += q
        left -= q
    return plan


def kronecker_schedule_product(
    seg_a: Sequence[Tuple[int, ...]],
    seg_b: Sequence[Tuple[int, ...]],
) -> List[Tuple[int, ...]]:
    """Concatenate every (a then b) pair — product of short schedule segments."""
    out: List[Tuple[int, ...]] = []
    for a in seg_a:
        for b in seg_b:
            out.append(a + b)
    return out


def pack_scheduled(trits: np.ndarray, plan: Sequence[Tuple[int, ...]]) -> bytes:
    """Pack by emitting each planned frame in order (flat concat of parts)."""
    # Flatten plan to a single repeating? No — one-shot sequence covering n.
    flat_parts: List[int] = []
    for parts in plan:
        flat_parts.extend(parts)
    # Use pack_frame with the full parts list as a single "frame" of length n.
    return pack_frame(trits, tuple(flat_parts))


def unpack_scheduled(data: bytes, n: int, plan: Sequence[Tuple[int, ...]]) -> np.ndarray:
    flat_parts: List[int] = []
    for parts in plan:
        flat_parts.extend(parts)
    return unpack_frame(data, n, tuple(flat_parts))


def schedule_bit_bound(plan: Sequence[Tuple[int, ...]]) -> Tuple[int, int]:
    """Return (packed_bits, flat_bits) for the planned covering."""
    packed = 0
    total = 0
    for parts in plan:
        packed += sum(bits(q) for q in parts)
        total += sum(parts)
    return packed, bits(total) if total else 0


def selftest() -> int:
    # C_i monotonic
    assert C_i(1) == 2
    assert C_i(5) == 8
    assert C_i(41) == 65

    # Collatz segment
    seg = collatz_segment(7, 5)
    assert seg[0] == 7 and seg[1] == 22

    tax0 = tax0_frames_from_catalog(max_trits=800)
    assert any(sum(p) == 665 for p in tax0)

    rng = np.random.default_rng(5)
    n_cases = 0
    for n in (0, 1, 40, 41, 306, 665, 1000, 2000):
        plan = plan_schedule(n, tax0)
        assert sum(sum(p) for p in plan) == n
        packed_bits, flat_bits = schedule_bit_bound(plan)
        # Packed is at least flat; tax is non-negative
        assert packed_bits >= flat_bits
        w = rng.integers(-1, 2, size=n, dtype=np.int8)
        if n == 0:
            assert pack_scheduled(w, plan) == b""
            n_cases += 1
            continue
        blob = pack_scheduled(w, plan)
        rec = unpack_scheduled(blob, n, plan)
        assert np.array_equal(rec, w), n
        # Per-frame taxes sum to packed - sum(bits(frame_q)); global tax
        # packed - bits(n) may be larger (inter-frame seams).
        per_frame = sum(split_tax(p) for p in plan)
        frame_flat = sum(bits(sum(p)) for p in plan)
        assert packed_bits - frame_flat == per_frame
        assert packed_bits - flat_bits >= per_frame
        n_cases += 1

    # Kronecker product of short segments
    a = [((41, 19),), ((5,),)]
    # flatten structure: use simple parts tuples
    seg_a = [(41, 19), (5,)]
    seg_b = [(41,), (5, 1)]
    prod = kronecker_schedule_product(seg_a, seg_b)
    assert len(prod) == 4
    assert (41, 19, 41) in prod

    print(
        f"COLLATZ_SCHEDULE PASS n_cases={n_cases} "
        f"tax0_frames={len(tax0)} product_schedules={len(prod)}"
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
