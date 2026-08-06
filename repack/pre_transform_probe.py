#!/usr/bin/env python3
"""Structured pre-transform probe (idea #4) — working experiment.

Applies a Kronecker product of small Walsh–Hadamard (±1) blocks to a float
matrix, ternarizes, then measures alphabet entropy, pack sizes, and a
per-block phase-slack proxy — versus ternarize-then-pack on the same floats.

HONEST LIMIT: any density win changes the ternary *word* (the weights).
It does not cancel {Q log2 3} for a fixed trit stream. Floats appear only in
the transform front-end and display ratios; pack size verdicts stay exact.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

from pack_ladder import (
    pack_41_65,
    pack_5_8,
    stream_pack,
    theory_bytes_41_65,
    theory_bytes_5_8,
)
from tax_graph import bits


def hadamard(n: int) -> np.ndarray:
    """Sylvester Hadamard matrix of order n (n power of 2), entries ±1."""
    if n < 1 or n & (n - 1):
        raise ValueError("n must be a power of 2")
    h = np.array([[1]], dtype=np.int8)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def kronecker_hadamard(factors: Sequence[int]) -> np.ndarray:
    """Kronecker product of Hadamard blocks with the given orders."""
    mats = [hadamard(f) for f in factors]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out.astype(np.int8)


def ternarize_absmean(x: np.ndarray) -> np.ndarray:
    """Absmean threshold → {-1,0,+1} (BitNet-style). Exact after threshold."""
    flat = x.astype(np.float64).ravel()
    scale = np.mean(np.abs(flat))
    if scale == 0.0:
        return np.zeros_like(flat, dtype=np.int8)
    y = flat / scale
    out = np.zeros_like(y, dtype=np.int8)
    out[y > 0.5] = 1
    out[y < -0.5] = -1
    return out


def alphabet_entropy(w: np.ndarray) -> Tuple[float, Dict[int, float]]:
    """Shannon entropy in bits (display float) and empirical probs."""
    flat = w.astype(np.int64).ravel()
    n = flat.size
    if n == 0:
        return 0.0, {-1: 0.0, 0: 0.0, 1: 0.0}
    probs = {}
    h = 0.0
    for v in (-1, 0, 1):
        p = float(np.count_nonzero(flat == v)) / n
        probs[v] = p
        if p > 0.0:
            h -= p * math.log2(p)
    return h, probs


def phase_slack_proxy(n_blocks: int, block_q: int) -> int:
    """Exact total slack of n_blocks independent flat containers vs one flat.

    slack = n*bits(Q) - bits(n*Q)  (Law B tax of n equal parts).
    """
    if n_blocks <= 0 or block_q <= 0:
        return 0
    return n_blocks * bits(block_q) - bits(n_blocks * block_q)


def pack_sizes(w: np.ndarray) -> Dict[str, int]:
    flat = w.ravel()
    n = int(flat.size)
    return {
        "n": n,
        "fmt_5_8": len(pack_5_8(flat)),
        "fmt_41_65": len(pack_41_65(flat)),
        "fmt_stream": len(stream_pack(flat)),
        "th_5_8": theory_bytes_5_8(n),
        "th_41_65": theory_bytes_41_65(n),
    }


def run_probe(
    rows: int = 64,
    cols: int = 64,
    seed: int = 0,
    factors: Sequence[int] = (4, 4, 4),
) -> Dict:
    """Compare baseline ternarize vs Hadamard-Kronecker → ternarize."""
    rng = np.random.default_rng(seed)
    dim = 1
    for f in factors:
        dim *= f
    if rows * cols < dim:
        raise ValueError("matrix too small for transform block")

    # Use a dim×dim block for the structured transform.
    W = rng.normal(size=(dim, dim))
    H = kronecker_hadamard(factors)
    # Orthogonal up to scale: (1/sqrt(n)) H; we apply unscaled H then ternarize
    # (scale absorbed by absmean). Exact integer H; float only in W and multiply.
    Wt = (H.astype(np.float64) @ W @ H.astype(np.float64).T) / dim

    w0 = ternarize_absmean(W)
    w1 = ternarize_absmean(Wt)

    h0, p0 = alphabet_entropy(w0)
    h1, p1 = alphabet_entropy(w1)
    s0 = pack_sizes(w0)
    s1 = pack_sizes(w1)

    # Phase proxy on 41-trit blocks
    n41_0 = w0.size // 41
    n41_1 = w1.size // 41
    slack0 = phase_slack_proxy(n41_0, 41)
    slack1 = phase_slack_proxy(n41_1, 41)

    return {
        "dim": dim,
        "factors": list(factors),
        "baseline": {
            "H": h0,
            "probs": p0,
            "sizes": s0,
            "phase_slack_41": slack0,
        },
        "transformed": {
            "H": h1,
            "probs": p1,
            "sizes": s1,
            "phase_slack_41": slack1,
        },
        "delta_stream_bytes": s1["fmt_stream"] - s0["fmt_stream"],
        "delta_5_8_bytes": s1["fmt_5_8"] - s0["fmt_5_8"],
        "note": (
            "Wins/losses reflect a different ternary word after the transform, "
            "not a cancellation of log2(3) irrationality inside a fixed stream."
        ),
    }


def selftest() -> int:
    H2 = hadamard(2)
    assert H2.shape == (2, 2)
    assert int(np.linalg.det(H2.astype(float))) != 0

    K = kronecker_hadamard((2, 2))
    assert K.shape == (4, 4)
    # Kronecker of Hadamards is Hadamard (up to order)
    assert set(np.unique(K).tolist()) == {-1, 1}

    w = ternarize_absmean(np.array([0.0, 0.1, -0.9, 2.0]))
    assert set(w.tolist()).issubset({-1, 0, 1})

    # Exact phase slack identity for 2×41
    assert phase_slack_proxy(2, 41) == 2 * bits(41) - bits(82)

    probe = run_probe(seed=0, factors=(4, 4, 4))
    assert probe["baseline"]["sizes"]["fmt_5_8"] == probe["baseline"]["sizes"]["th_5_8"]
    assert probe["transformed"]["sizes"]["fmt_5_8"] == probe["transformed"]["sizes"]["th_5_8"]

    print(
        f"PRE_TRANSFORM PASS dim={probe['dim']} "
        f"H0={probe['baseline']['H']:.4f} H1={probe['transformed']['H']:.4f} "
        f"d_stream={probe['delta_stream_bytes']} "
        f"d_5_8={probe['delta_5_8_bytes']}"
    )
    print(f"  note: {probe['note']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
