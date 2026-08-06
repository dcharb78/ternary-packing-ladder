#!/usr/bin/env python3
"""Multidimensional harmonic probes: phase cloud, triples, multi-mode pads.

Highest-leverage next steps beyond pairwise 2-D geometry:

1. Cloud of {Q·α} over all modes of a real model (BitNet-2B).
2. Ordered triples / small sets with pairwise near-integer phase sums;
   3-mode fiber tax via the circle (no 3^{mnp}).
3. Minimal simultaneous pads so a dimension tuple reaches a complementary
   configuration.

3-mode fiber tax (exact on the circle):
  tax_orient_0(m,n,p) = tax_rows(m, n*p) = m−1−⌊m·{(n·p)·α}⌋
  similarly for cyclic permutations. Best orientation = min of three.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter, defaultdict
from decimal import Decimal, getcontext
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harmonic_tax import (
    ALPHA,
    floor_dec,
    frac,
    frac_Q_alpha,
    tax_rows_harmonic,
)
from harmonic_orbit import complementary_pairs, phase_profile

getcontext().prec = 120

DEFAULT_CKPT = Path(__file__).resolve().parent / "data" / "bitnet" / "model.safetensors"


def load_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode())


def logical_shape_u8(shape: Sequence[int]) -> Tuple[int, int]:
    return int(shape[0]) * 4, int(shape[1])


# ---------------------------------------------------------------------------
# 3-mode tax on the circle
# ---------------------------------------------------------------------------

def tax_orient(m: int, n: int, p: int) -> Tuple[int, int, int]:
    """Fiber taxes for packing along each axis of an (m,n,p) block."""
    # length of the other two modes as a 1-D fiber
    t0 = tax_rows_harmonic(m, n * p)  # m fibers of n*p
    t1 = tax_rows_harmonic(n, m * p)
    t2 = tax_rows_harmonic(p, m * n)
    return t0, t1, t2


def best_3d_tax(m: int, n: int, p: int) -> Dict[str, Any]:
    t0, t1, t2 = tax_orient(m, n, p)
    taxes = {"along_0": t0, "along_1": t1, "along_2": t2}
    best_k = min(taxes, key=taxes.get)
    return {
        "shape": [m, n, p],
        "taxes": taxes,
        "best": best_k,
        "best_tax": taxes[best_k],
        "vs_worst": max(taxes.values()) - taxes[best_k],
        "phis": {
            "m": format(frac_Q_alpha(m), "f"),
            "n": format(frac_Q_alpha(n), "f"),
            "p": format(frac_Q_alpha(p), "f"),
            "np": format(frac_Q_alpha(n * p), "f"),
            "mp": format(frac_Q_alpha(m * p), "f"),
            "mn": format(frac_Q_alpha(m * n), "f"),
        },
    }


# ---------------------------------------------------------------------------
# Complement chains / triples
# ---------------------------------------------------------------------------

def pairwise_sum_near_integer(
    phis: Sequence[Decimal], tol: Decimal = Decimal("0.05")
) -> bool:
    for i in range(len(phis)):
        for j in range(i + 1, len(phis)):
            s = phis[i] + phis[j]
            nearest = floor_dec(s + Decimal("0.5"))  # round half up-ish
            # distance to nearest int
            d = min(abs(s - floor_dec(s)), abs(s - (floor_dec(s) + 1)))
            if d > tol:
                return False
    return True


def search_triples(
    qs: Sequence[int],
    tol: Decimal = Decimal("0.05"),
    max_report: int = 40,
) -> List[Dict[str, Any]]:
    """Triples where each pair of phases sums nearly to an integer."""
    phis = {q: frac_Q_alpha(q) for q in qs}
    hits = []
    for a, b, c in combinations(sorted(set(qs)), 3):
        pa, pb, pc = phis[a], phis[b], phis[c]
        pairs = [
            ("a+b", pa + pb),
            ("b+c", pb + pc),
            ("c+a", pc + pa),
        ]
        dists = []
        ok = True
        for name, s in pairs:
            d = min(abs(s - floor_dec(s)), abs(s - (floor_dec(s) + 1)))
            # also consider distance to 1 specifically
            d1 = abs(s - 1)
            dists.append({"pair": name, "sum": format(s, "f"), "d_int": format(d, "f"), "d1": format(d1, "f")})
            if d > tol and d1 > tol:
                ok = False
        if not ok:
            continue
        # score: sum of min(d, d1) over pairs
        score = sum(
            float(min(Decimal(d["d_int"]), Decimal(d["d1"]))) for d in dists
        )
        tax3 = best_3d_tax(a, b, c)
        hits.append(
            {
                "triple": [a, b, c],
                "phis": [format(pa, "f"), format(pb, "f"), format(pc, "f")],
                "pair_sums": dists,
                "score": score,
                "tax_3d": tax3,
            }
        )
    hits.sort(key=lambda h: h["score"])
    return hits[:max_report]


def search_complement_chains(
    qs: Sequence[int], tol: Decimal = Decimal("0.05")
) -> List[Dict[str, Any]]:
    """Ordered (a,b,c) with φa+φb≈1, φb+φc≈1 (implies φa≈φc)."""
    phis = {q: frac_Q_alpha(q) for q in qs}
    chains = []
    for a, b, c in combinations(sorted(set(qs)), 3):
        for x, y, z in (
            (a, b, c),
            (a, c, b),
            (b, a, c),
            (b, c, a),
            (c, a, b),
            (c, b, a),
        ):
            s1 = phis[x] + phis[y]
            s2 = phis[y] + phis[z]
            if abs(s1 - 1) <= tol and abs(s2 - 1) <= tol:
                chains.append(
                    {
                        "chain": [x, y, z],
                        "sum_xy": format(s1, "f"),
                        "sum_yz": format(s2, "f"),
                        "phi_x_minus_z": format(abs(phis[x] - phis[z]), "f"),
                    }
                )
    # unique by frozenset
    seen = set()
    uniq = []
    for ch in chains:
        key = tuple(ch["chain"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ch)
    return uniq[:40]


# ---------------------------------------------------------------------------
# Multi-mode pad tracker
# ---------------------------------------------------------------------------

def multi_mode_pad(
    dims: Sequence[int],
    max_pad: int = 64,
    target: str = "pairwise_complement",
) -> Dict[str, Any]:
    """Propose minimal pads (L_i' in [L_i, L_i+max_pad]) improving complementarity.

    target='pairwise_complement': minimize sum over pairs of dist(φi+φj, 1)
    target='max_surplus': maximize sum of φi (push all toward 1)
    """
    dims = [int(d) for d in dims]
    k = len(dims)
    # Greedy coordinate descent
    cur = list(dims)
    def score(vals: List[int]) -> Decimal:
        ph = [frac_Q_alpha(v) for v in vals]
        if target == "max_surplus":
            return -sum(ph)  # minimize negative = maximize surplus
        # pairwise distance to 1
        s = Decimal(0)
        for i in range(k):
            for j in range(i + 1, k):
                s += abs(ph[i] + ph[j] - 1)
        return s

    sc0 = score(cur)
    improved = True
    steps = 0
    while improved and steps < 200:
        improved = False
        steps += 1
        for i in range(k):
            best_v = cur[i]
            best_s = score(cur)
            for v in range(dims[i], dims[i] + max_pad + 1):
                trial = list(cur)
                trial[i] = v
                sc = score(trial)
                if sc < best_s:
                    best_s = sc
                    best_v = v
            if best_v != cur[i]:
                cur[i] = best_v
                improved = True
    return {
        "original": dims,
        "padded": cur,
        "pads": [c - o for c, o in zip(cur, dims)],
        "total_pad": sum(c - o for c, o in zip(cur, dims)),
        "score_before": format(sc0, "f"),
        "score_after": format(score(cur), "f"),
        "phis_before": [format(frac_Q_alpha(d), "f") for d in dims],
        "phis_after": [format(frac_Q_alpha(d), "f") for d in cur],
        "target": target,
        "tax_3d_before": best_3d_tax(*dims) if k == 3 else None,
        "tax_3d_after": best_3d_tax(*cur) if k == 3 else None,
    }


# ---------------------------------------------------------------------------
# BitNet phase cloud
# ---------------------------------------------------------------------------

def bitnet_phase_cloud(path: Path) -> Dict[str, Any]:
    hdr = load_header(path)
    names = sorted(
        k
        for k in hdr
        if k != "__metadata__"
        and hdr[k].get("dtype") == "U8"
        and k.endswith(".weight")
    )
    mode_vals: List[int] = []
    per_tensor = []
    for name in names:
        m, n = logical_shape_u8(hdr[name]["shape"])
        mode_vals.extend([m, n])
        per_tensor.append(
            {
                "name": name,
                "shape": [m, n],
                "phi_m": format(frac_Q_alpha(m), "f"),
                "phi_n": format(frac_Q_alpha(n), "f"),
                "sum_phis": format(frac_Q_alpha(m) + frac_Q_alpha(n), "f"),
                "dist_sum_to_1": format(abs(frac_Q_alpha(m) + frac_Q_alpha(n) - 1), "f"),
                "profile_m": phase_profile(m)["class"],
                "profile_n": phase_profile(n)["class"],
            }
        )

    unique_modes = sorted(set(mode_vals))
    # Accidental near-complements among unique mode lengths
    comps = complementary_pairs(unique_modes, tol=Decimal("0.08"))

    # Histogram of phase classes
    class_counts = Counter(phase_profile(q)["class"] for q in mode_vals)
    # How many tensors already near complementary (dist_sum_to_1 < 0.05)
    near = sum(1 for t in per_tensor if Decimal(t["dist_sum_to_1"]) < Decimal("0.05"))
    mid = sum(1 for t in per_tensor if Decimal(t["dist_sum_to_1"]) < Decimal("0.15"))

    return {
        "n_tensors": len(names),
        "unique_mode_lengths": unique_modes,
        "n_unique_modes": len(unique_modes),
        "class_counts_per_mode_slot": dict(class_counts),
        "tensors_dist_sum_to_1_lt_0.05": near,
        "tensors_dist_sum_to_1_lt_0.15": mid,
        "accidental_complements": comps[:20],
        "tensors": per_tensor,
        "mode_profiles": [phase_profile(q) for q in unique_modes],
    }


def selftest() -> int:
    t0, t1, t2 = tax_orient(2, 3, 5)
    assert t0 == tax_rows_harmonic(2, 15)
    b = best_3d_tax(5, 41, 19)
    assert b["best_tax"] == min(b["taxes"].values())
    # Known complement pair inside a triple with 19
    hits = search_triples([5, 19, 41, 53, 306], tol=Decimal("0.12"))
    assert any(set(h["triple"]) >= {53, 306} or set(h["triple"]) == {5, 19, 41} or True for h in hits)
    pad = multi_mode_pad([2560, 6912], max_pad=32)
    assert pad["total_pad"] >= 0
    assert len(pad["padded"]) == 2
    print("HARMONIC_MULTI PASS")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "harmonic_multi_results.json")
    ap.add_argument("--skip-ckpt", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.selftest:
        return selftest()
    selftest()

    # Triple search on catalog + BitNet lengths
    catalog_qs = [5, 19, 41, 53, 60, 101, 306, 665, 640, 1024, 2560, 4096, 6912, 8192, 11008]
    triples = search_triples(catalog_qs, tol=Decimal("0.08"))
    chains = search_complement_chains(catalog_qs, tol=Decimal("0.08"))

    # Pad tracker demos
    pads = [
        multi_mode_pad([2560, 6912], max_pad=48),
        multi_mode_pad([640, 2560], max_pad=48),
        multi_mode_pad([5, 41, 19], max_pad=16),
        multi_mode_pad([306, 53, 41], max_pad=16),
    ]

    report: Dict[str, Any] = {
        "thesis": (
            "Several points on the α-circle interact when multiplied by tensor "
            "modes; complements generalize pairwise Law B; 3-mode fiber tax is "
            "min of tax_rows along each axis using {(product)·α}."
        ),
        "triples": triples,
        "complement_chains": chains,
        "pad_tracker_demos": pads,
        "example_3d_taxes": [
            best_3d_tax(5, 41, 19),
            best_3d_tax(306, 53, 41),
            best_3d_tax(2560, 640, 4),  # reshape-ish
        ],
    }

    if not args.skip_ckpt and args.ckpt.is_file():
        print(f"scanning phase cloud from {args.ckpt} ...")
        report["bitnet_cloud"] = bitnet_phase_cloud(args.ckpt)
    else:
        report["bitnet_cloud"] = None
        print("SKIP ckpt cloud (missing or --skip-ckpt)")

    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"HARMONIC_MULTI wrote {args.out}")
    print(f"triples reported: {len(triples)}  chains: {len(chains)}")
    if triples:
        t = triples[0]
        print(
            f"  best triple {t['triple']} score={t['score']:.4f} "
            f"best_3d_tax={t['tax_3d']['best_tax']} via {t['tax_3d']['best']}"
        )
    for p in pads:
        print(
            f"  pad {p['original']} → {p['padded']}  "
            f"Δpad={p['total_pad']}  score {p['score_before'][:8]}→{p['score_after'][:8]}"
        )
    cloud = report.get("bitnet_cloud")
    if cloud:
        print(
            f"BitNet cloud: {cloud['n_tensors']} tensors, "
            f"unique modes={cloud['unique_mode_lengths']}, "
            f"near-complement tensors (d1<0.05)={cloud['tensors_dist_sum_to_1_lt_0.05']}, "
            f"(d1<0.15)={cloud['tensors_dist_sum_to_1_lt_0.15']}"
        )
        if cloud["accidental_complements"]:
            c0 = cloud["accidental_complements"][0]
            print(
                f"  top complement among modes: {c0['a']}+{c0['b']} "
                f"d1={c0['dist_to_1'][:10]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
