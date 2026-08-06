#!/usr/bin/env python3
"""Farther probes — underexplored angles beyond the 665-flat frontier.

Grounded in what worked (flat Law-B containers, α-circle language, base-p
same machine, three loci) and what failed (fiber-41 density, chiral frames,
digit nesting, pad-to-align, seam/twin predictors, dynamical nulls).

Does NOT reopen fiber-41 / characters / Stokes as density levers.
Does NOT claim a win unless measured bytes beat hybrid665 on ternary weights.

Probe-only: theory_bytes_flat_Q / flat-1277 BitNet absolute are NOT wired into
pack_ladder or ledger_packer (no --hybrid1277). Promote only with a clean
optional flag + RT selftests; until then treat as FARTHER.md measurement.

Probes implemented (cheap):
  1. Mixed-ledger portfolio (global budget + decode complexity score)
  2. Activation / KV / runtime shapes (synthetic) vs 665
  3. Non-uniform trit entropy (BitNet samples → adaptive vs geometric 665)
  4. Tile ∩ 665/306 (GEMM-friendly joint byte+alignment score)
  5. Shared remainder dictionary (cross-tensor rem<665 codebook)
  6. Streaming chunk = rung (I/O @ 665 fragmentation)
  7. Next flat after 665 (theory scan for mid-size surplus flats)
  8. Base-5 mini codec stub (round-trip small convergent + theory @ 643)
  9. Inverse/dual (bit→trit packing for metadata-sized streams)
 10. Blind: rem-phase correlation; scale-byte dual-radix sketch

Docs: FARTHER.md · DIMENSIONS.md
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from collections import Counter, defaultdict
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from better_density import best_layout
from checkpoint_axis_probe import load_header, unpack_u8_trits
from harmonic_tax import bits_from_alpha, frac_Q_alpha
from pack_ladder import (
    BLOCK_306,
    BLOCK_665,
    BYTES_306,
    BYTES_665,
    container_bytes_for_r_trits,
    theory_bytes_5_8,
    theory_bytes_306_485,
    theory_bytes_665_1055,
)
from packing_stack import load_bitnet_shapes
from scale_probe import LLM_SHAPES

getcontext().prec = 120

DEFAULT_CKPT = Path(__file__).resolve().parent / "data" / "bitnet" / "model.safetensors"
OUT = Path(__file__).resolve().parent / "farther_probes_results.json"

# Decode complexity score (arbitrary but fixed): higher = harder decode.
# Flat 5_8 is cheapest; bigint blocks cost more per block family.
COMPLEXITY: Dict[str, float] = {
    "flat58": 1.0,
    "hybrid306": 2.5,
    "hybrid665": 4.0,
}

# Synthetic activation / KV shapes (not weights) — typical inference tensors.
ACTIVATION_SHAPES: Tuple[Tuple[str, int, int], ...] = (
    # (name, m, n) — flattened as m*n ternary-like elements for size theory
    ("act_b8_s2k_h4k", 8 * 2048, 4096),       # hidden act [B*S, H]
    ("act_b1_s4k_h2k", 1 * 4096, 2560),       # BitNet-ish act
    ("kv_b8_s2k_h1k", 8 * 2048, 1024),        # KV cache slice (GQA-ish)
    ("kv_b1_s8k_h512", 1 * 8192, 512),
    ("mlp_act_b4_s1k", 4 * 1024, 11008),      # intermediate act
    ("attn_scores_b4_h32_s1k", 4 * 32, 1024 * 1024),  # scores (large)
    ("layernorm_stats", 32, 4096),              # tiny runtime
    ("rope_cache", 8192, 128),
)

# Common GEMM tile candidates (hardware-friendly) + packing factors
TILE_CANDIDATES: Tuple[int, ...] = (
    16, 32, 64, 128, 256, 512, 640, 665, 1024, 1330, 1536, 1995, 2048, 2560, 3060, 4096,
)


def ledger_bytes(m: int, n: int) -> Dict[str, int]:
    return {
        "flat58": theory_bytes_5_8(m * n),
        "hybrid306": best_layout(m, n, theory_bytes_306_485)["bytes"],
        "hybrid665": best_layout(m, n, theory_bytes_665_1055)["bytes"],
    }


def winning_ledger(bmap: Dict[str, int]) -> str:
    return min(bmap, key=lambda k: bmap[k])


# ---------------------------------------------------------------------------
# 1. Mixed-ledger portfolio
# ---------------------------------------------------------------------------

def probe_mixed_portfolio(shapes: Sequence[Tuple[str, int, int]]) -> Dict[str, Any]:
    """Per-tensor best vs all-665 vs weighted portfolio (bytes + λ·complexity)."""
    uniq: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for name, m, n in shapes:
        uniq[(m, n)].append(name)

    rows: List[Dict[str, Any]] = []
    tot_flat = tot_306 = tot_665 = tot_best = tot_port = 0
    n_tensors = 0
    best_counts: Counter = Counter()
    port_counts: Counter = Counter()

    # λ chosen so complexity only flips near-ties (~few bytes)
    lambdas = (0.0, 0.5, 2.0, 10.0)

    for (m, n), names in sorted(uniq.items()):
        cnt = len(names)
        n_tensors += cnt
        b = ledger_bytes(m, n)
        flat, h306, h665 = b["flat58"], b["hybrid306"], b["hybrid665"]
        best_k = winning_ledger(b)
        best_b = b[best_k]
        tot_flat += flat * cnt
        tot_306 += h306 * cnt
        tot_665 += h665 * cnt
        tot_best += best_b * cnt
        best_counts[best_k] += cnt

        # portfolio pick under λ=2 default for row table
        score = {
            k: b[k] + 2.0 * COMPLEXITY[k] for k in ("flat58", "hybrid306", "hybrid665")
        }
        port_k = min(score, key=lambda k: score[k])
        port_counts[port_k] += cnt
        tot_port += b[port_k] * cnt

        rows.append(
            {
                "m": m,
                "n": n,
                "count": cnt,
                "bytes": b,
                "best": best_k,
                "portfolio_lambda2": port_k,
                "delta_best_vs_665": best_b - h665,
            }
        )

    lambda_sweep: List[Dict[str, Any]] = []
    for lam in lambdas:
        t_bytes = 0
        counts: Counter = Counter()
        for (m, n), names in uniq.items():
            cnt = len(names)
            b = ledger_bytes(m, n)
            score = {k: b[k] + lam * COMPLEXITY[k] for k in b}
            k = min(score, key=lambda x: score[x])
            counts[k] += cnt
            t_bytes += b[k] * cnt
        lambda_sweep.append(
            {
                "lambda": lam,
                "total_bytes": t_bytes,
                "delta_vs_all665": t_bytes - tot_665,
                "delta_vs_best_bytes": t_bytes - tot_best,
                "counts": dict(counts),
            }
        )

    beats_665 = tot_best < tot_665
    return {
        "section": "mixed_ledger_portfolio",
        "tag": "measured",
        "n_tensors": n_tensors,
        "n_unique_shapes": len(uniq),
        "totals": {
            "flat58": tot_flat,
            "all_hybrid306": tot_306,
            "all_hybrid665": tot_665,
            "per_tensor_best_bytes": tot_best,
            "portfolio_lambda2_bytes": tot_port,
        },
        "delta_MB": {
            "best_vs_665": round((tot_best - tot_665) / 1e6, 6),
            "best_vs_flat": round((tot_best - tot_flat) / 1e6, 6),
            "port_l2_vs_665": round((tot_port - tot_665) / 1e6, 6),
        },
        "best_counts": dict(best_counts),
        "portfolio_lambda2_counts": dict(port_counts),
        "lambda_sweep": lambda_sweep,
        "beats_665_flat_on_ternary_weights": beats_665,
        "note": (
            "On BitNet, hybrid665 already wins every tensor on pure bytes, so "
            "per-tensor best == all-665. Complexity λ only flips to cheaper "
            "ledgers when λ is large enough to buy decode simplicity."
        ),
        "verdict": (
            "null as density win past 665 — portfolio cannot beat all-665 on "
            "bytes when 665 wins every shape; λ>0 only trades bytes for simpler decode."
            if not beats_665
            else "measured: per-tensor best beats forced-665"
        ),
    }


# ---------------------------------------------------------------------------
# 2. Activation / KV / runtime shapes
# ---------------------------------------------------------------------------

def probe_activation_shapes(
    shapes: Sequence[Tuple[str, int, int]] = ACTIVATION_SHAPES,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    wins = Counter()
    tot = defaultdict(int)
    for name, m, n in shapes:
        b = ledger_bytes(m, n)
        w = winning_ledger(b)
        wins[w] += 1
        for k, v in b.items():
            tot[k] += v
        rows.append(
            {
                "name": name,
                "m": m,
                "n": n,
                "trits": m * n,
                "bytes": b,
                "winner": w,
                "delta_665_vs_flat": b["hybrid665"] - b["flat58"],
                "delta_665_vs_306": b["hybrid665"] - b["hybrid306"],
            }
        )
    return {
        "section": "activation_kv_runtime",
        "tag": "measured",
        "note": (
            "Synthetic activation/KV shapes — theory bytes as if ternary. "
            "Real activations are usually float/int8; this asks whether 665 "
            "geometry still wins *if* those buffers were ternary."
        ),
        "n_shapes": len(rows),
        "win_counts": dict(wins),
        "totals": dict(tot),
        "delta_665_vs_flat_MB": round((tot["hybrid665"] - tot["flat58"]) / 1e6, 4),
        "delta_665_vs_306_MB": round((tot["hybrid665"] - tot["hybrid306"]) / 1e6, 4),
        "beats_665_flat": False,  # comparing other ledgers TO 665
        "665_still_wins": wins.most_common(1)[0][0] == "hybrid665" if wins else None,
        "rows": rows,
        "verdict": (
            "measured: 665-flat still wins synthetic act/KV shapes on theory bytes"
            if wins and wins.most_common(1)[0][0] == "hybrid665"
            else "measured: some act/KV shapes prefer another ledger"
        ),
    }


# ---------------------------------------------------------------------------
# 3. Non-uniform trit entropy (BitNet samples)
# ---------------------------------------------------------------------------

def _sample_bitnet_histograms(
    ckpt: Path,
    max_tensors: int = 12,
    max_trits_per: int = 2_000_000,
) -> List[Dict[str, Any]]:
    """Unpack up to max_tensors; optionally subsample for adaptive round-trip."""
    if not ckpt.is_file():
        return []
    hdr, data0 = load_header(ckpt)
    names = sorted(
        k
        for k in hdr
        if k != "__metadata__"
        and hdr[k].get("dtype") == "U8"
        and k.endswith(".weight")
    )
    # Spread across layers: take every N-th
    step = max(1, len(names) // max_tensors)
    pick = names[::step][:max_tensors]
    mm = np.memmap(ckpt, dtype=np.uint8, mode="r")

    from adaptive_entropy import (
        adaptive_bound_bytes,
        empirical_H_bits,
        static_roundtrip,
        weights_to_trits,
    )

    rows: List[Dict[str, Any]] = []
    for name in pick:
        beg, end = hdr[name]["data_offsets"]
        raw = np.asarray(mm[data0 + beg : data0 + end])
        w = unpack_u8_trits(raw)  # {-1,0,1}
        n = int(w.size)
        # Full histogram always (cheap)
        n_m1 = int(np.sum(w == -1))
        n_0 = int(np.sum(w == 0))
        n_p1 = int(np.sum(w == 1))
        assert n_m1 + n_0 + n_p1 == n
        counts = (n_m1, n_0, n_p1)  # as trit symbols after +1: same order
        H = empirical_H_bits(counts, n)
        b58 = theory_bytes_5_8(n)
        b665 = theory_bytes_665_1055(n)
        b_bound = adaptive_bound_bytes(n, H)

        # Adaptive real on subsample for speed (header overhead honest +12)
        sample_n = min(n, max_trits_per)
        if sample_n < n:
            rng = np.random.default_rng(abs(hash(name)) % (2**31))
            idx = rng.choice(n, size=sample_n, replace=False)
            ws = w[idx]
        else:
            ws = w
        trits = weights_to_trits(ws)
        payload, b_real_sample = static_roundtrip(trits)
        # Scale sample adaptive rate to full n + 12 B header
        rate = b_real_sample / sample_n
        b_real_est = int(math.ceil(rate * n)) + 12

        shape = hdr[name]["shape"]
        m, inn = int(shape[0]) * 4, int(shape[1])
        rows.append(
            {
                "name": name.split("model.layers.")[-1] if "model.layers." in name else name,
                "m": m,
                "n_cols": inn,
                "n_trits": n,
                "p_m1": n_m1 / n,
                "p_0": n_0 / n,
                "p_p1": n_p1 / n,
                "H_bits": round(H, 6),
                "log2_3": round(math.log2(3), 6),
                "b_flat58": b58,
                "b_hybrid665": b665,
                "b_adaptive_bound": b_bound,
                "b_adaptive_est": b_real_est,
                "sample_n": sample_n,
                "adaptive_beats_665": b_real_est < b665,
                "bound_beats_665": b_bound < b665,
                "delta_adapt_est_vs_665": b_real_est - b665,
            }
        )
    return rows


def probe_nonuniform_entropy(ckpt: Path) -> Dict[str, Any]:
    rows = _sample_bitnet_histograms(ckpt)
    if not rows:
        return {
            "section": "nonuniform_trit_entropy",
            "tag": "speculative",
            "verdict": "no ckpt — skipped",
            "beats_665_flat_on_ternary_weights": False,
        }
    n_adapt_win = sum(1 for r in rows if r["adaptive_beats_665"])
    n_bound_win = sum(1 for r in rows if r["bound_beats_665"])
    tot_665 = sum(r["b_hybrid665"] for r in rows)
    tot_adapt = sum(r["b_adaptive_est"] for r in rows)
    tot_bound = sum(r["b_adaptive_bound"] for r in rows)
    mean_H = sum(r["H_bits"] for r in rows) / len(rows)
    mean_p0 = sum(r["p_0"] for r in rows) / len(rows)
    beats = tot_adapt < tot_665
    return {
        "section": "nonuniform_trit_entropy",
        "tag": "measured",
        "n_sampled_tensors": len(rows),
        "mean_H_bits": round(mean_H, 6),
        "mean_p0": round(mean_p0, 6),
        "uniform_H": round(math.log2(3), 6),
        "n_adaptive_est_beats_665": n_adapt_win,
        "n_bound_beats_665": n_bound_win,
        "totals_sampled": {
            "hybrid665": tot_665,
            "adaptive_est": tot_adapt,
            "adaptive_bound": tot_bound,
        },
        "delta_adapt_vs_665_MB": round((tot_adapt - tot_665) / 1e6, 4),
        "delta_bound_vs_665_MB": round((tot_bound - tot_665) / 1e6, 4),
        "beats_665_flat_on_ternary_weights": beats,
        "rows": rows,
        "note": (
            "Adaptive uses per-tensor empirical static range coder (+12 B header). "
            "Sampled tensors scaled to full length when subsampled. "
            "BitNet p(0)≈0.43 > 1/3 ⇒ H < log₂3 ⇒ entropy coding beats any "
            "uniform-ternary geometric ledger (including 665). Different family: "
            "variable-rate + counts header; not a fixed-width Law-B flat."
        ),
        "verdict": (
            "measured: adaptive entropy beats 665 on sampled BitNet "
            f"(Δ≈{(tot_adapt - tot_665)/1e6:.3f} MB on sample) — non-uniform p(0); "
            "decode/complexity Pareto, not a new geometric container"
            if beats
            else "null: BitNet histograms near-uniform — adaptive does not beat 665-flat"
        ),
    }


# ---------------------------------------------------------------------------
# 4. Tile ∩ 665/306
# ---------------------------------------------------------------------------

def _alignment_score(tile: int) -> Dict[str, Any]:
    """Joint score: packing density + hardware alignment friendliness."""
    b665 = theory_bytes_665_1055(tile)
    b306 = theory_bytes_306_485(tile)
    b58 = theory_bytes_5_8(tile)
    # Alignment: prefer power-of-2 and multiples of 16/32/64
    pow2 = tile > 0 and (tile & (tile - 1)) == 0
    mult16 = tile % 16 == 0
    mult32 = tile % 32 == 0
    mult64 = tile % 64 == 0
    mult5 = tile % 5 == 0
    rem665 = tile % BLOCK_665
    rem306 = tile % BLOCK_306
    # packing density (lower B/trit better)
    dens = b665 / tile
    # alignment bonus 0..1
    align = (
        (0.35 if pow2 else 0.0)
        + (0.20 if mult64 else 0.10 if mult32 else 0.05 if mult16 else 0.0)
        + (0.15 if mult5 else 0.0)
        + (0.20 if rem665 == 0 else 0.05 if rem665 < 50 else 0.0)
        + (0.10 if rem306 == 0 else 0.0)
    )
    # joint: density rank + (1-align) penalty — lower better
    joint = dens * 1000.0 + (1.0 - min(align, 1.0)) * 0.5
    return {
        "tile": tile,
        "b_flat58": b58,
        "b_hybrid306": b306,
        "b_hybrid665": b665,
        "B_per_trit_665": round(dens, 8),
        "rem_665": rem665,
        "rem_306": rem306,
        "pow2": pow2,
        "mult16": mult16,
        "mult5": mult5,
        "align_score": round(align, 4),
        "joint_score": round(joint, 6),
        "exact_665": rem665 == 0,
        "exact_306": rem306 == 0,
    }


def probe_tile_intersection() -> Dict[str, Any]:
    # Also generate tiles that are LCM-ish: multiples of 5 near 665 factors
    extra: List[int] = []
    for k in range(1, 12):
        extra.append(5 * k * 64)  # 5×64 multiples
        extra.append(BLOCK_665 * k)
        extra.append(BLOCK_306 * k)
    # Near-665 with pow2-ish: 640, 672, 768, 1280, 1330
    extra.extend([640, 672, 768, 1024, 1280, 1330, 1536, 1920, 1995, 2048])
    tiles = sorted(set(TILE_CANDIDATES) | set(extra))
    rows = [_alignment_score(t) for t in tiles if t > 0]
    rows.sort(key=lambda r: r["joint_score"])
    top = rows[:12]
    exact_665 = [r for r in rows if r["exact_665"]]
    # Best joint that is not pure 665-multiple vs pure 665
    best_joint = rows[0]
    pure_665 = next(r for r in rows if r["tile"] == BLOCK_665)
    return {
        "section": "tile_intersect_665_306",
        "tag": "measured",
        "n_tiles": len(rows),
        "top_joint": top,
        "exact_665_tiles": exact_665[:8],
        "pure_665": pure_665,
        "best_joint_vs_pure665": {
            "best_tile": best_joint["tile"],
            "best_joint_score": best_joint["joint_score"],
            "pure665_joint_score": pure_665["joint_score"],
            "best_denser_than_665": best_joint["B_per_trit_665"] < pure_665["B_per_trit_665"],
        },
        "beats_665_flat_on_ternary_weights": False,
        "note": (
            "Joint score mixes B/trit with hardware alignment. Exact ×665 "
            "always densest packing; pow2 tiles win alignment but lose density. "
            "No tile beats 665 density — only softens the design-time tradeoff."
        ),
        "verdict": (
            "null as density past 665; measured design-time prior: prefer "
            "multiples of 665 that are also mult of 16/64 when redesigning widths"
        ),
    }


# ---------------------------------------------------------------------------
# 5. Shared remainder dictionary
# ---------------------------------------------------------------------------

def probe_shared_remainder(shapes: Sequence[Tuple[str, int, int]]) -> Dict[str, Any]:
    """Many rem = n % 665 tails — shared codebook of rem→bytes vs per-tensor rem."""
    rem_counts: Counter = Counter()
    rem_bytes_naive = 0  # each rem packed independently as bigint
    n_tensors = 0
    for name, m, n in shapes:
        # best layout length for 665
        layouts = [
            ("flat", m * n),
            ("row", n),  # rem per row fiber — but we score flat stream rem
            ("col", m),
        ]
        # Use flat stream rem (same as hybrid665 path on flat)
        L = m * n
        full, rem = divmod(L, BLOCK_665)
        rem_counts[rem] += 1
        n_tensors += 1
        if rem:
            rem_bytes_naive += container_bytes_for_r_trits(rem)

    # Shared dictionary: one copy of each unique rem payload template
    # Savings ≈ sum over rem>0 of (count-1) * container_bytes(rem)
    # (first occurrence pays the codebook entry; rest pay an index)
    unique_rems = {r: container_bytes_for_r_trits(r) for r in rem_counts if r > 0}
    codebook_bytes = sum(unique_rems.values())
    # Index: ceil(log2(|codebook|))/8 per tensor with rem>0 — use 2 B index
    n_with_rem = sum(c for r, c in rem_counts.items() if r > 0)
    index_bytes = 2 * n_with_rem
    shared_total = codebook_bytes + index_bytes
    # Naive: every rem packed inline
    savings = rem_bytes_naive - shared_total

    # Also compare to full hybrid665 total
    tot_665 = sum(
        theory_bytes_665_1055(m * n) for _, m, n in shapes
    )
    # Shared only helps rem tails; full blocks unchanged
    tot_shared_scheme = tot_665 - rem_bytes_naive + shared_total

    return {
        "section": "shared_remainder_dictionary",
        "tag": "measured",
        "n_tensors": n_tensors,
        "n_unique_rems": len(rem_counts),
        "n_nonzero_unique_rems": len(unique_rems),
        "rem_histogram_top": rem_counts.most_common(15),
        "rem_bytes_naive_inline": rem_bytes_naive,
        "shared_codebook_bytes": codebook_bytes,
        "shared_index_bytes": index_bytes,
        "shared_total_rem_cost": shared_total,
        "savings_vs_inline_rem_B": savings,
        "tot_hybrid665": tot_665,
        "tot_with_shared_rem": tot_shared_scheme,
        "delta_vs_665_B": tot_shared_scheme - tot_665,
        "beats_665_flat_on_ternary_weights": tot_shared_scheme < tot_665,
        "note": (
            "Shared rem codebook only wins when many tensors share the same "
            "nonzero rem. On BitNet only 3 rem classes appear — tiny absolute "
            "save (~KB), bookkeeping not a new geometry. Tag as micro-win."
        ),
        "verdict": (
            f"measured micro-win: shared rem saves {savings} B vs inline rem "
            f"({(tot_shared_scheme - tot_665)/1e6:.4f} MB vs hybrid665 total) "
            "— not a frontier shift"
            if savings > 0 and tot_shared_scheme < tot_665
            else "null: shared rem dictionary does not beat 665-flat (+index overhead)"
        ),
    }


# ---------------------------------------------------------------------------
# 6. Streaming chunk = rung
# ---------------------------------------------------------------------------

def probe_streaming_chunks() -> Dict[str, Any]:
    """Fragmentation when I/O chunks are forced to size C vs stream length L."""
    lengths = [2560 * 2560, 6912 * 2560, 4096 * 4096, 665 * 100, 306 * 200, 10**7]
    chunk_sizes = [4096, 65536, 1024 * 1024, BLOCK_665, BLOCK_306, BYTES_665 * 1024]
    # Interpret chunk as *trit* chunk for packing, or byte chunk for I/O waste
    rows: List[Dict[str, Any]] = []
    for L in lengths:
        packed = theory_bytes_665_1055(L)
        for C in chunk_sizes:
            # Pack each trit-chunk of size C independently, then rem
            full, rem = divmod(L, C)
            if C in (BLOCK_665, BLOCK_306):
                # aligned packing quantum
                chunk_pack = (
                    theory_bytes_665_1055(C)
                    if C == BLOCK_665
                    else theory_bytes_306_485(C)
                )
            else:
                chunk_pack = theory_bytes_665_1055(C)
            frag_bytes = full * chunk_pack + (
                theory_bytes_665_1055(rem) if rem else 0
            )
            waste = frag_bytes - packed
            rows.append(
                {
                    "L": L,
                    "chunk_trits": C,
                    "n_full_chunks": full,
                    "rem": rem,
                    "packed_flat_665_stream": packed,
                    "packed_chunked": frag_bytes,
                    "fragmentation_waste_B": waste,
                    "waste_pct": round(100.0 * waste / packed, 4) if packed else 0.0,
                }
            )
    # Summarize: does chunk=665 eliminate waste vs arbitrary chunks?
    by_chunk: Dict[int, List[int]] = defaultdict(list)
    for r in rows:
        by_chunk[r["chunk_trits"]].append(r["fragmentation_waste_B"])
    summary = {
        str(C): {
            "mean_waste": round(sum(ws) / len(ws), 2),
            "max_waste": max(ws),
            "zero_waste_count": sum(1 for w in ws if w == 0),
        }
        for C, ws in by_chunk.items()
    }
    return {
        "section": "streaming_chunk_rung",
        "tag": "measured",
        "summary_by_chunk": summary,
        "sample_rows": [r for r in rows if r["L"] == lengths[0]][:8],
        "beats_665_flat_on_ternary_weights": False,
        "note": (
            "Chunking a stream into independent packs always ≥ flat 665 stream. "
            "Chunk=665 gives zero *extra* waste vs one flat 665 stream when L "
            "is arbitrary (same as theory_bytes_665_1055). Arbitrary byte I/O "
            "chunks can force rem-heavy fragmentation if misaligned."
        ),
        "verdict": (
            "measured: I/O trit-chunks at 665 match flat 665 (no new density); "
            "misaligned chunks only add waste — null as a beat-665 lever"
        ),
    }


# ---------------------------------------------------------------------------
# 7. Next flat after 665
# ---------------------------------------------------------------------------

def theory_bytes_flat_Q(n: int, Q: int, block_bytes: int) -> int:
    full, rem = divmod(n, Q)
    return full * block_bytes + container_bytes_for_r_trits(rem)


def probe_next_flat(
    shapes: Optional[Sequence[Tuple[str, int, int]]] = None,
) -> Dict[str, Any]:
    """Theory scan: surplus CF / Law-B flats denser than 665 and still 'practical'."""
    from tax_graph import SURPLUS_RUNGS, DEFICIT_PIECES

    candidates: List[int] = []
    # Known CF dens
    cf_like = [5, 12, 41, 53, 306, 665, 15601]
    candidates.extend(cf_like)
    # Law B sums: pairs / triples of known parts (e.g. 665+2×306 = 1277)
    parts = sorted(set(list(SURPLUS_RUNGS) + list(DEFICIT_PIECES) + [5, 19, 41, 53, 306, 665]))
    for a in parts:
        for b in parts:
            candidates.append(a + b)
            candidates.append(2 * a + b)
            candidates.append(a + 2 * b)
            candidates.append(3 * a + b)
    # Scan Q in 666..4000 for strong surplus (φ>0.95) — cheap Decimal
    for Q in range(666, 4001):
        phi = frac_Q_alpha(Q)
        if phi >= Decimal("0.98"):
            candidates.append(Q)

    uniq = sorted(set(Q for Q in candidates if Q >= 5))
    rows: List[Dict[str, Any]] = []
    bps_665 = BYTES_665 / BLOCK_665
    for Q in uniq:
        if Q == 15601 or Q > 5000:
            bits = bits_from_alpha(Q)
            b = (bits + 7) // 8
        else:
            b = container_bytes_for_r_trits(Q)
        bps = b / Q
        rows.append(
            {
                "Q": Q,
                "bytes": b,
                "B_per_trit": round(bps, 8),
                "beats_665_density": bps < bps_665,
                "delta_bps": round(bps - bps_665, 8),
                "practical": Q <= 2000 and b <= 400,  # heuristic
                "phi": float(frac_Q_alpha(Q)),
            }
        )
    denser = [r for r in rows if r["beats_665_density"]]
    denser.sort(key=lambda r: r["B_per_trit"])
    practical_denser = [r for r in denser if r["practical"]]

    # BitNet absolute measure for best practical denser (expect 1277 = 665+2×306)
    bitnet: Optional[Dict[str, Any]] = None
    beats_665 = False
    if shapes and practical_denser:
        best_Q = practical_denser[0]["Q"]
        best_B = practical_denser[0]["bytes"]
        tot_665 = tot_new = tot_flat = 0
        win_new = 0
        for _, m, n in shapes:
            b665 = best_layout(m, n, theory_bytes_665_1055)["bytes"]
            bnew = best_layout(
                m, n, lambda L, Q=best_Q, B=best_B: theory_bytes_flat_Q(L, Q, B)
            )["bytes"]
            bflat = theory_bytes_5_8(m * n)
            tot_665 += b665
            tot_new += bnew
            tot_flat += bflat
            if bnew < b665:
                win_new += 1
        beats_665 = tot_new < tot_665
        bitnet = {
            "Q": best_Q,
            "block_bytes": best_B,
            "assembly_hint": "665+2×306" if best_Q == 1277 else None,
            "n_tensors": len(shapes),
            "tot_flat58": tot_flat,
            "tot_hybrid665": tot_665,
            f"tot_hybrid{best_Q}": tot_new,
            "delta_vs_665_MB": round((tot_new - tot_665) / 1e6, 4),
            "delta_vs_flat_MB": round((tot_new - tot_flat) / 1e6, 4),
            "n_tensors_new_beats_665": win_new,
            "tag": "measured",
        }

    return {
        "section": "next_flat_after_665",
        "tag": "measured",
        "bps_665": bps_665,
        "n_candidates": len(rows),
        "n_denser_than_665": len(denser),
        "n_practical_denser": len(practical_denser),
        "top_denser": denser[:10],
        "practical_denser": practical_denser[:10],
        "bitnet_best_practical": bitnet,
        "beats_665_operational": beats_665,
        "beats_665_flat_on_ternary_weights": beats_665,
        "note": (
            "Law-B-sum flat 1277 (=665+2×306) at 253 B/block is denser than "
            "132/665 and still practical. 15601 remains denser but huge. "
            "Same pattern as 665: flat container of a sum, not chiral parts."
        ),
        "verdict": (
            f"measured: flat Q={bitnet['Q']} ({bitnet.get('assembly_hint')}) "
            f"beats 665 by {bitnet['delta_vs_665_MB']} MB on BitNet"
            if bitnet and beats_665
            else (
                "measured: practical denser flats exist in theory but BitNet "
                "absolute did not beat 665 (inspect)"
                if practical_denser
                else "measured: no practical flat denser than 665 in scan"
            )
        ),
    }


# ---------------------------------------------------------------------------
# 8. Base-5 mini codec stub
# ---------------------------------------------------------------------------

def _pack_base_p_block(symbols: np.ndarray, p: int, block_Q: int) -> bytes:
    """Bigint block pack for alphabet {0..p-1}; mirrors pack_ladder bigint style."""
    s = np.asarray(symbols, dtype=np.int64).reshape(-1)
    assert np.all((s >= 0) & (s < p))
    n = int(s.size)
    full, rem = divmod(n, block_Q)
    from base_p_ladder import container_bytes

    b_block = container_bytes(p, block_Q)
    out = bytearray()
    for i in range(full):
        chunk = s[i * block_Q : (i + 1) * block_Q]
        v = 0
        for sym in chunk.tolist():
            v = v * p + int(sym)
        out.extend(int(v).to_bytes(b_block, "little"))
    if rem:
        chunk = s[full * block_Q :]
        v = 0
        for sym in chunk.tolist():
            v = v * p + int(sym)
        b_rem = container_bytes(p, rem)
        out.extend(int(v).to_bytes(b_rem, "little"))
    return bytes(out)


def _unpack_base_p_block(data: bytes, n: int, p: int, block_Q: int) -> np.ndarray:
    from base_p_ladder import container_bytes

    full, rem = divmod(n, block_Q)
    b_block = container_bytes(p, block_Q)
    out = np.empty(n, dtype=np.int64)
    pos = 0
    for i in range(full):
        raw = data[pos : pos + b_block]
        pos += b_block
        v = int.from_bytes(raw, "little")
        for j in range(block_Q - 1, -1, -1):
            v, out[i * block_Q + j] = divmod(v, p)
    if rem:
        b_rem = container_bytes(p, rem)
        raw = data[pos : pos + b_rem]
        v = int.from_bytes(raw, "little")
        base = full * block_Q
        for j in range(rem - 1, -1, -1):
            v, out[base + j] = divmod(v, p)
    return out


def probe_base5_codec() -> Dict[str, Any]:
    """Round-trip small p=5 convergent blocks; theory at Q=643."""
    from base_p_ladder import container_bytes, naive_symbols_per_byte, theory_bytes_block, theory_bytes_naive

    p = 5
    # Small first rungs that are feasible for bigint RT
    small_Qs = (3, 28, 59)  # CF dens; 3 is tiny surplus-ish
    rt_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(42)
    for Q in small_Qs:
        syms = rng.integers(0, p, size=Q * 3 + 7, dtype=np.int64)
        data = _pack_base_p_block(syms, p, Q)
        back = _unpack_base_p_block(data, int(syms.size), p, Q)
        ok = bool(np.array_equal(back, syms))
        naive = theory_bytes_naive(p, int(syms.size))
        block = theory_bytes_block(p, int(syms.size), Q)
        rt_rows.append(
            {
                "block_Q": Q,
                "n": int(syms.size),
                "rt_ok": ok,
                "packed_B": len(data),
                "theory_block_B": block,
                "theory_naive_B": naive,
                "beats_naive": block < naive,
            }
        )
    Q643 = 643
    b643 = container_bytes(p, Q643)
    naive_k = naive_symbols_per_byte(p)
    return {
        "section": "base5_mini_codec",
        "tag": "measured",
        "rt_small_blocks": rt_rows,
        "all_rt_ok": all(r["rt_ok"] for r in rt_rows),
        "theory_Q643": {
            "Q": Q643,
            "bytes": b643,
            "B_per_sym": round(b643 / Q643, 6),
            "naive_k": naive_k,
            "naive_Bps": round(1.0 / naive_k, 6),
            "beats_naive": (b643 / Q643) < (1.0 / naive_k),
        },
        "beats_665_flat_on_ternary_weights": False,
        "note": (
            "Pattern transfer works: p=5 flat blocks round-trip. Q=643 is the "
            "natural large surplus flat (theory). Not a ternary BitNet lever."
        ),
        "verdict": (
            "measured: base-5 mini codec RT ok on small CF blocks; theory 643 "
            "beats naïve — pattern transfer confirmed; does not beat ternary 665"
        ),
    }


# ---------------------------------------------------------------------------
# 9. Inverse / dual — bits into p-ary
# ---------------------------------------------------------------------------

def probe_inverse_dual() -> Dict[str, Any]:
    """Pack bitstreams as base-3 digits; compare to raw bytes for metadata sizes."""
    bit_lengths = [64, 128, 256, 512, 1024, 4096, 65536]
    rows: List[Dict[str, Any]] = []
    for nb in bit_lengths:
        raw_B = (nb + 7) // 8
        # Interpret bits as little-endian integer → base-3 digits
        # Number of trits needed: ceil(nb * log(2)/log(3))
        n_trits = int(math.ceil(nb * math.log(2) / math.log(3)))
        # Pack those trits with 665 / 5_8
        b58 = theory_bytes_5_8(n_trits)
        b665 = theory_bytes_665_1055(n_trits)
        # Also: group bits into base-8 digits already = bytes — control
        rows.append(
            {
                "n_bits": nb,
                "raw_bytes": raw_B,
                "n_trits_needed": n_trits,
                "trit_pack_5_8": b58,
                "trit_pack_665": b665,
                "665_beats_raw": b665 < raw_B,
                "expansion_vs_raw": round(b665 / raw_B, 4) if raw_B else None,
            }
        )
    any_win = any(r["665_beats_raw"] for r in rows)
    return {
        "section": "inverse_dual_bit_to_trit",
        "tag": "measured",
        "rows": rows,
        "any_665_beats_raw_bits": any_win,
        "beats_665_flat_on_ternary_weights": False,
        "note": (
            "Information theory: log2(3)>1 so trit-coding a bitstring expands. "
            "No byte win for metadata/scales via 2→3 packing."
        ),
        "verdict": "null / false_id: packing bits into trits expands — never beats raw bytes",
    }


# ---------------------------------------------------------------------------
# 10. Blind spots
# ---------------------------------------------------------------------------

def probe_blind_spots(shapes: Sequence[Tuple[str, int, int]]) -> Dict[str, Any]:
    """(a) rem vs phase correlation (b) dual-radix scale byte sketch."""
    rem_phase: List[Dict[str, Any]] = []
    for name, m, n in shapes[:50]:  # cap
        L = m * n
        rem = L % BLOCK_665
        phi = float(frac_Q_alpha(rem)) if rem else 0.0
        rem_phase.append({"rem": rem, "phi_rem": phi, "L": L})

    # Correlation: does rem near surplus predict anything about bytes?
    # Bytes for rem is determined by rem alone — phase is just a rename.
    # Tag as applies_as_language only.
    rem_set = {r["rem"] for r in rem_phase}
    # Dual-radix: store a float16 scale as (mantissa bits via base-3? ) — sketch
    # float16 = 16 bits = 2 B. Trit pack of 16 bits needs ~11 trits → ≥3 B.
    dual = {
        "float16_raw_B": 2,
        "float16_as_trits_665": theory_bytes_665_1055(
            int(math.ceil(16 * math.log(2) / math.log(3)))
        ),
        "float32_raw_B": 4,
        "uint8_scale_raw_B": 1,
        "tag": "null",
        "note": "Dual-radix for scales expands; keep scales in binary.",
    }
    return {
        "section": "blind_spots",
        "tag": "measured",
        "rem_phase": {
            "n": len(rem_phase),
            "n_unique_rem": len(rem_set),
            "tag": "applies_as_language",
            "note": (
                "rem phase is a coordinate rename of rem size; does not predict "
                "bytes beyond container_bytes(rem)."
            ),
            "sample": rem_phase[:8],
        },
        "dual_radix_scales": dual,
        "beats_665_flat_on_ternary_weights": False,
        "verdict": (
            "null: rem-phase and dual-radix scales are not density levers past 665"
        ),
    }


# ---------------------------------------------------------------------------
# Also score LLM weight shapes for activation comparison context
# ---------------------------------------------------------------------------

def probe_llm_weight_control() -> Dict[str, Any]:
    rows = []
    wins = Counter()
    for name, m, n in LLM_SHAPES:
        b = ledger_bytes(m, n)
        w = winning_ledger(b)
        wins[w] += 1
        rows.append({"name": name, "m": m, "n": n, "winner": w, "bytes": b})
    return {
        "section": "llm_weight_shape_control",
        "tag": "measured",
        "win_counts": dict(wins),
        "rows": rows,
        "665_dominates": wins.get("hybrid665", 0) == len(rows),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(ckpt: Path) -> Dict[str, Any]:
    t0 = time.time()
    shapes = load_bitnet_shapes(ckpt) if ckpt.is_file() else []
    if not shapes:
        # fallback synthetic
        shapes = list(LLM_SHAPES)

    portfolio = probe_mixed_portfolio(shapes)
    activations = probe_activation_shapes()
    entropy = probe_nonuniform_entropy(ckpt)
    tiles = probe_tile_intersection()
    shared = probe_shared_remainder(shapes)
    streaming = probe_streaming_chunks()
    next_flat = probe_next_flat(shapes)
    base5 = probe_base5_codec()
    inverse = probe_inverse_dual()
    blinds = probe_blind_spots(shapes)
    llm_ctrl = probe_llm_weight_control()

    probes = [
        portfolio,
        activations,
        entropy,
        tiles,
        shared,
        streaming,
        next_flat,
        base5,
        inverse,
        blinds,
        llm_ctrl,
    ]

    any_beat_665 = any(p.get("beats_665_flat_on_ternary_weights") for p in probes)
    geometric_beat = bool(
        next_flat.get("beats_665_flat_on_ternary_weights")
    )
    board_summary = []
    for p in probes:
        board_summary.append(
            {
                "section": p.get("section"),
                "tag": p.get("tag"),
                "verdict": p.get("verdict"),
                "beats_665": p.get("beats_665_flat_on_ternary_weights", False),
            }
        )

    return {
        "stance": (
            "Explore farther from 665-flat frontier without collapsing dimensions. "
            "Tag honestly. Nothing here reopens fiber-41/chiral/nesting as density."
        ),
        "elapsed_s": round(time.time() - t0, 3),
        "ckpt": str(ckpt),
        "ckpt_exists": ckpt.is_file(),
        "n_bitnet_shapes": len(shapes) if ckpt.is_file() else 0,
        "any_probe_beats_665_on_ternary_weights": any_beat_665,
        "geometric_flat_beats_665": geometric_beat,
        "board_summary": board_summary,
        "probes": {p["section"]: p for p in probes if "section" in p},
        "global_verdict": (
            "Geometric: flat 1277 (665+2×306, 253 B/block) beats 665-flat on "
            "BitNet (~−0.66 MB) — same Law-B-sum-as-flat pattern. "
            "Entropy: adaptive beats 665 when p(0) elevates (different family). "
            "Shared-rem: micro-KB win only. Portfolio/tiles/streaming/inverse/"
            "base-5: null or non-ternary transfer."
            if geometric_beat
            else (
                "No geometric flat beat 665; inspect entropy/shared-rem micro-wins."
                if any_beat_665
                else "No probe beats 665-flat on ternary weight bytes."
            )
        ),
    }


def selftest() -> None:
    # Portfolio identity
    r = probe_mixed_portfolio([("a", 1, 665), ("b", 1, 670)])
    assert r["totals"]["all_hybrid665"] == theory_bytes_665_1055(665) + theory_bytes_665_1055(670)
    # Base-5 RT
    b5 = probe_base5_codec()
    assert b5["all_rt_ok"]
    # Inverse expands
    inv = probe_inverse_dual()
    assert not inv["any_665_beats_raw_bits"]
    # Tile scores
    t = probe_tile_intersection()
    assert t["pure_665"]["exact_665"]
    # Streaming: chunk=665 waste vs flat should be 0 for any L
    st = probe_streaming_chunks()
    for row in st["sample_rows"]:
        if row["chunk_trits"] == BLOCK_665:
            assert row["fragmentation_waste_B"] == 0
    print("farther_probes selftest OK")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("selftest", "run"), nargs="?", default="run")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "selftest":
        selftest()
        return 0

    out = run_all(args.ckpt)
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps({"wrote": str(OUT), "global_verdict": out["global_verdict"],
                      "any_beat_665": out["any_probe_beats_665_on_ternary_weights"],
                      "board": out["board_summary"], "elapsed_s": out["elapsed_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
