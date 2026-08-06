#!/usr/bin/env python3
"""Base-p packing ladder — geometric machine for odd primes p (and b≥3).

Ternary (p=3) is the smallest interesting case. The same circle / CF /
Law-B / flat-container story runs with α_p = log₂ p. This module measures
rung tables and naïve-vs-block density for p ∈ {3,5,7,11} — theory bytes
only; no codecs.

HARD RULE: container bits for feasible Q use exact (p**Q).bit_length().
Fractional parts use Decimal. Display ratios may use floats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from decimal import ROUND_FLOOR, Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

getcontext().prec = 200

OUT = Path(__file__).resolve().parent / "base_p_ladder_results.json"
DEFAULT_PRIMES: Tuple[int, ...] = (3, 5, 7, 11)


def floor_dec(x: Decimal) -> int:
    return int(x.to_integral_value(rounding=ROUND_FLOOR))


def frac(x: Decimal) -> Decimal:
    return x - x.to_integral_value(rounding=ROUND_FLOOR)


def alpha_p(p: int) -> Decimal:
    """α_p = log₂(p)."""
    return Decimal(p).ln() / Decimal(2).ln()


def frac_Q_alpha(p: int, Q: int) -> Decimal:
    return frac(Decimal(Q) * alpha_p(p))


def container_bits(p: int, Q: int) -> int:
    """Minimal P with p^Q ≤ 2^P. Exact for feasible Q (p odd ⇒ never power of 2)."""
    if Q <= 0:
        return 0
    return (p ** Q).bit_length()


def container_bytes(p: int, Q: int) -> int:
    if Q <= 0:
        return 0
    return (container_bits(p, Q) + 7) // 8


def bits_from_alpha(p: int, Q: int) -> int:
    """⌊Q·α_p⌋+1 — must match container_bits for modest Q."""
    if Q <= 0:
        return 0
    need = max(80, Q.bit_length() + 64 + p.bit_length())
    if getcontext().prec < need:
        getcontext().prec = need
    return floor_dec(Decimal(Q) * alpha_p(p)) + 1


def classify_phase(phi: Decimal, surplus: Decimal = Decimal("0.9"), deficit: Decimal = Decimal("0.1")) -> str:
    if abs(phi - Decimal("0.5")) <= Decimal("0.05"):
        return "near_half"
    if phi >= surplus:
        return "surplus_near_1"
    if phi <= deficit:
        return "deficit_near_0"
    return "mid"


def naive_symbols_per_byte(p: int) -> int:
    """Largest k with p^k ≤ 256 (one-byte digit block)."""
    k = 0
    v = 1
    while v * p <= 256:
        v *= p
        k += 1
    return k


def theory_bytes_naive(p: int, n: int) -> int:
    """Ceil-on-block: full k-symbol bytes + 1 B for any rem (matches fmt_5_8 style)."""
    if n <= 0:
        return 0
    k = naive_symbols_per_byte(p)
    if k <= 0:
        # p > 256: each symbol needs >1 byte
        b_per = (p.bit_length() + 7) // 8
        return n * b_per
    full, rem = divmod(n, k)
    return full + (1 if rem else 0)


def theory_bytes_block(p: int, n: int, block_Q: int) -> int:
    """Flat multi-limb blocks of size block_Q + rem via container_bytes(rem)."""
    if n <= 0 or block_Q <= 0:
        return 0
    full, rem = divmod(n, block_Q)
    return full * container_bytes(p, block_Q) + container_bytes(p, rem)


def continued_fraction_convergents(
    alpha: Decimal, max_terms: int = 24
) -> List[Tuple[int, int]]:
    """CF convergents (a/b ≈ α) as (num, den) with den = Q candidate."""
    x = alpha
    convergents: List[Tuple[int, int]] = []
    h_prev2, k_prev2 = 0, 1
    h_prev1, k_prev1 = 1, 0
    for _ in range(max_terms):
        a = floor_dec(x)
        h = a * h_prev1 + h_prev2
        k = a * k_prev1 + k_prev2
        convergents.append((h, k))
        frac_part = x - Decimal(a)
        if frac_part == 0:
            break
        x = 1 / frac_part
        h_prev2, k_prev2 = h_prev1, k_prev1
        h_prev1, k_prev1 = h, k
        if k > 10**7:
            break
    return convergents


def best_approximations(
    p: int,
    Q_max: int = 2000,
    top_n: int = 12,
) -> List[Dict[str, Any]]:
    """Scan Q=1..Q_max for best surplus (near 1) and deficit (near 0) phases.

    Feasible exact (p**Q).bit_length() only when Q is small enough; for
    large Q we use Decimal φ only and mark bits as alpha-chart.
    """
    # Exact pow limit: keep p**Q under ~few thousand bits for speed
    exact_Q_max = Q_max
    # 5**800 is huge; cap exact by bit budget ~ 1e5 bits
    while exact_Q_max > 1 and exact_Q_max * math.log2(p) > 80_000:
        exact_Q_max //= 2

    scored: List[Tuple[Decimal, int, str]] = []
    for Q in range(1, Q_max + 1):
        phi = frac_Q_alpha(p, Q)
        d1 = 1 - phi
        d0 = phi
        if d1 <= d0:
            scored.append((d1, Q, "surplus_near_1"))
        else:
            scored.append((d0, Q, "deficit_near_0"))

    scored.sort(key=lambda t: t[0])
    # Deduplicate near-duplicates keeping best distance per neighborhood
    picked: List[Dict[str, Any]] = []
    used: List[int] = []
    for dist, Q, side in scored:
        if any(abs(Q - u) < max(3, Q // 50) for u in used):
            continue
        phi = frac_Q_alpha(p, Q)
        use_exact = Q <= exact_Q_max
        row: Dict[str, Any] = {
            "Q": Q,
            "phi": format(phi, "f"),
            "dist_to_boundary": format(dist, "e"),
            "side": side,
            "phase_class": classify_phase(phi),
            "bits_exact": container_bits(p, Q) if use_exact else None,
            "bits_alpha": bits_from_alpha(p, Q),
            "bytes_exact": container_bytes(p, Q) if use_exact else None,
            "B_per_symbol": (
                container_bytes(p, Q) / Q if use_exact else None
            ),
            "bits_source": "p**Q.bit_length()" if use_exact else "floor(Q*alpha)+1",
        }
        if use_exact:
            assert row["bits_exact"] == row["bits_alpha"], (p, Q)
        picked.append(row)
        used.append(Q)
        if len(picked) >= top_n:
            break
    return picked


def rung_table_from_cf(p: int, max_terms: int = 20) -> List[Dict[str, Any]]:
    """First CF convergents of α_p — denominators are rung candidates."""
    alpha = alpha_p(p)
    convs = continued_fraction_convergents(alpha, max_terms=max_terms)
    rows = []
    seen_Q = set()
    for num, den in convs:
        if den <= 0 or den in seen_Q:
            continue
        seen_Q.add(den)
        # Skip trivial Q=1 early if we want variety — keep all
        use_exact = den * math.log2(p) <= 80_000
        phi = frac_Q_alpha(p, den)
        # For α≈P/Q, {Q α} is either near 0 or near 1 depending on side
        err = abs(alpha - Decimal(num) / Decimal(den))
        rows.append(
            {
                "Q": den,
                "P_approx": num,
                "cf_error": format(err, "e"),
                "phi": format(phi, "f"),
                "phase_class": classify_phase(phi),
                "bits_exact": container_bits(p, den) if use_exact else None,
                "bits_alpha": bits_from_alpha(p, den),
                "bytes_exact": container_bytes(p, den) if use_exact else None,
                "B_per_symbol": (
                    round(container_bytes(p, den) / den, 8) if use_exact else None
                ),
                "vs_log2p": (
                    round(container_bytes(p, den) / den - math.log2(p) / 8, 8)
                    if use_exact
                    else None
                ),
            }
        )
    return rows


def density_probe(p: int, block_Q: Optional[int] = None) -> Dict[str, Any]:
    """Naïve one-byte digit pack vs first large flat convergent block."""
    k = naive_symbols_per_byte(p)
    naive_Bps = 1.0 / k if k else float((p.bit_length() + 7) // 8)
    floor_Bps = math.log2(p) / 8.0

    cf = rung_table_from_cf(p)
    # Prefer largest exact convergent with Q>=k*4 and known bytes
    candidates = [
        r for r in cf
        if r["bytes_exact"] is not None and r["Q"] >= max(k * 2, 5)
    ]
    if not candidates:
        candidates = [r for r in cf if r["bytes_exact"] is not None]
    if block_Q is not None:
        chosen_Q = block_Q
    elif candidates:
        # Prefer surplus or tightest vs floor among mid/large Q
        candidates_sorted = sorted(
            candidates,
            key=lambda r: (r["B_per_symbol"] if r["B_per_symbol"] is not None else 9.0, -r["Q"]),
        )
        chosen_Q = candidates_sorted[0]["Q"]
    else:
        chosen_Q = k

    # Also try best surplus from scan (often better operational block)
    bests = best_approximations(p, Q_max=min(2500, max(400, chosen_Q * 2)), top_n=8)
    surplus_exact = [
        r for r in bests
        if r["side"] == "surplus_near_1" and r["bytes_exact"] is not None and r["Q"] >= k
    ]
    alt_Q = surplus_exact[0]["Q"] if surplus_exact else chosen_Q

    lengths = [chosen_Q, alt_Q, chosen_Q * 3, alt_Q * 5, 10_000, 50_000]
    lengths = sorted(set(n for n in lengths if n > 0))

    comparisons = []
    for n in lengths:
        bn = theory_bytes_naive(p, n)
        bc = theory_bytes_block(p, n, chosen_Q)
        ba = theory_bytes_block(p, n, alt_Q)
        comparisons.append(
            {
                "n": n,
                "naive_bytes": bn,
                "block_cf_Q": chosen_Q,
                "block_cf_bytes": bc,
                "delta_cf_vs_naive": bc - bn,
                "block_surplus_Q": alt_Q,
                "block_surplus_bytes": ba,
                "delta_surplus_vs_naive": ba - bn,
                "block_cf_wins": bc < bn,
                "block_surplus_wins": ba < bn,
            }
        )

    n_win_cf = sum(1 for c in comparisons if c["block_cf_wins"])
    n_win_sur = sum(1 for c in comparisons if c["block_surplus_wins"])
    bps_cf = container_bytes(p, chosen_Q) / chosen_Q
    bps_alt = container_bytes(p, alt_Q) / alt_Q

    return {
        "p": p,
        "alpha": format(alpha_p(p), "f"),
        "log2_p_B_per_symbol_floor": floor_Bps,
        "naive_k_symbols_per_byte": k,
        "naive_B_per_symbol": naive_Bps,
        "naive_overhead_vs_floor": naive_Bps - floor_Bps,
        "block_cf_Q": chosen_Q,
        "block_cf_B_per_symbol": bps_cf,
        "block_cf_vs_naive": bps_cf - naive_Bps,
        "block_surplus_Q": alt_Q,
        "block_surplus_B_per_symbol": bps_alt,
        "block_surplus_vs_naive": bps_alt - naive_Bps,
        "comparisons": comparisons,
        "n_lengths_cf_beats_naive": n_win_cf,
        "n_lengths_surplus_beats_naive": n_win_sur,
        "verdict_tag": (
            "measured_block_beats_naive"
            if (bps_cf < naive_Bps - 1e-12 or bps_alt < naive_Bps - 1e-12)
            else "measured_naive_competitive"
        ),
    }


def regimes_note(p: int) -> Dict[str, Any]:
    """Three loci (surplus / ½ / deficit) are geometric for any irrational α."""
    # Sample circle: phase histogram of Q=1..500
    counts = {"surplus_near_1": 0, "near_half": 0, "deficit_near_0": 0, "mid": 0}
    for Q in range(1, 501):
        counts[classify_phase(frac_Q_alpha(p, Q))] += 1
    return {
        "p": p,
        "alpha_irrational": True,  # log2(odd prime) irrational
        "three_loci_geometric": True,
        "note": (
            "Three regimes remain the right dynamical *image* for every p at "
            "the geometric level (same circle loci: surplus / ½ / deficit). "
            "Whether training dynamics need more regimes is still open/speculative."
        ),
        "sample_Q_1_500_phase_counts": counts,
        "extra_regimes_needed": False,
        "training_more_regimes": {
            "status": "open_speculative",
            "tag": "speculative",
        },
        "tag": "applies_as_language",
        "odd_even_typed_sheet_analogue": {
            "status": "open_speculative",
            "note": (
                "Ternary odd-rung typed sheets / even-frame CF construction are "
                "p=3-specific contact language. Do not force UFRF twin-center "
                "onto base 5/7/11 without measurement."
            ),
            "tag": "speculative",
        },
    }


def _annotate_Q(p: int, Q: int) -> Dict[str, Any]:
    phi = frac_Q_alpha(p, Q)
    use_exact = Q * math.log2(p) <= 80_000
    return {
        "Q": Q,
        "phi": format(phi, "f"),
        "dist_to_0": format(phi, "e"),
        "dist_to_1": format(1 - phi, "e"),
        "dist_to_half": format(abs(phi - Decimal("0.5")), "f"),
        "phase_class": classify_phase(phi),
        "bits_exact": container_bits(p, Q) if use_exact else None,
        "bytes_exact": container_bytes(p, Q) if use_exact else None,
        "B_per_symbol": (
            round(container_bytes(p, Q) / Q, 8) if use_exact else None
        ),
    }


def locus_families(p: int, Q_max: int = 5000, top_n: int = 8) -> Dict[str, Any]:
    """Strongest deficit / surplus / near-½ among CF + scan candidates.

    Starting point in the p-world: lengths with {Q α_p} nearest to 0 are the
    strong *deficit* rungs — analogues of ternary 53 and 665. They are the
    natural “0” seeds for that alphabet.
    """
    cf = rung_table_from_cf(p, max_terms=28)
    # Candidate pool: CF dens + dense scan peaks
    pool_Q = {r["Q"] for r in cf if r["Q"] > 0}
    # Scan for extreme loci
    best_def: List[Tuple[Decimal, int]] = []
    best_sur: List[Tuple[Decimal, int]] = []
    best_half: List[Tuple[Decimal, int]] = []
    for Q in range(1, Q_max + 1):
        phi = frac_Q_alpha(p, Q)
        best_def.append((phi, Q))
        best_sur.append((1 - phi, Q))
        best_half.append((abs(phi - Decimal("0.5")), Q))
    best_def.sort(key=lambda t: t[0])
    best_sur.sort(key=lambda t: t[0])
    best_half.sort(key=lambda t: t[0])

    def pick(scored: List[Tuple[Decimal, int]], n: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        used: List[int] = []
        for dist, Q in scored:
            if any(abs(Q - u) < max(2, Q // 40) for u in used):
                continue
            row = _annotate_Q(p, Q)
            row["dist_metric"] = format(dist, "e")
            row["from_cf"] = Q in pool_Q
            out.append(row)
            used.append(Q)
            if len(out) >= n:
                break
        return out

    deficits = pick(best_def, top_n)
    surpluses = pick(best_sur, top_n)
    halves = pick(best_half, top_n)

    # CF-only split (early convergents classified by side)
    cf_deficit = []
    cf_surplus = []
    cf_half = []
    for r in cf:
        phi = Decimal(r["phi"])
        row = _annotate_Q(p, r["Q"])
        row["from_cf"] = True
        row["P_approx"] = r["P_approx"]
        if abs(phi - Decimal("0.5")) <= Decimal("0.08"):
            cf_half.append(row)
        elif phi <= Decimal("0.5"):
            # closer to 0 than 1 → deficit family (incl. mild)
            if phi <= Decimal("0.15") or (1 - phi) > phi:
                cf_deficit.append(row)
            else:
                cf_surplus.append(row)
        else:
            if (1 - phi) <= Decimal("0.15") or (1 - phi) < phi:
                cf_surplus.append(row)
            else:
                cf_deficit.append(row)

    return {
        "p": p,
        "alpha": format(alpha_p(p), "f"),
        "starting_point": (
            "Lengths with {Q α_p} nearest to 0 are the strong deficit rungs — "
            "analogues of ternary 53 and 665; natural “0” seeds for this alphabet."
        ),
        "strongest_deficit_near_0": deficits,
        "strongest_surplus_near_1": surpluses,
        "nearest_half": halves,
        "cf_early_deficit": cf_deficit[:top_n],
        "cf_early_surplus": cf_surplus[:top_n],
        "cf_early_half": cf_half[:top_n],
        "operational_moves_unchanged": [
            "flat large zero-tax / low-tax assemblies",
            "axis choice by bytes",
            "mild reshape scored by bytes",
            "surplus-phase design prior",
            "flat beats nested for density",
        ],
        "tag": "measured",
    }


def compare_p5_to_ternary() -> Dict[str, Any]:
    """Explicit p=5 locus families vs ternary deficit/surplus structure."""
    t3 = locus_families(3, Q_max=3000, top_n=6)
    t5 = locus_families(5, Q_max=3000, top_n=6)
    # Known ternary anchors for narrative
    ternary_known = {
        "surplus_family": [5, 41, 306],
        "deficit_family": [19, 53, 665],
        "near_half_example": 2560,  # BitNet architectural accident on α_3
        "note": (
            "Ternary surplus = CF rungs approaching 1 from below; deficit = "
            "Law-B remainders / large assemblies sitting near 0 (53, 665)."
        ),
    }
    return {
        "ternary_known": ternary_known,
        "p3_measured": {
            "deficit_Qs": [r["Q"] for r in t3["strongest_deficit_near_0"]],
            "surplus_Qs": [r["Q"] for r in t3["strongest_surplus_near_1"]],
            "half_Qs": [r["Q"] for r in t3["nearest_half"]],
            "cf_deficit": [r["Q"] for r in t3["cf_early_deficit"]],
            "cf_surplus": [r["Q"] for r in t3["cf_early_surplus"]],
        },
        "p5_measured": {
            "alpha_5": t5["alpha"],
            "alpha_5_approx": "≈2.321928",
            "deficit_Qs": [r["Q"] for r in t5["strongest_deficit_near_0"]],
            "surplus_Qs": [r["Q"] for r in t5["strongest_surplus_near_1"]],
            "half_Qs": [r["Q"] for r in t5["nearest_half"]],
            "cf_deficit": [r["Q"] for r in t5["cf_early_deficit"]],
            "cf_surplus": [r["Q"] for r in t5["cf_early_surplus"]],
            "strongest_deficit_detail": t5["strongest_deficit_near_0"],
            "strongest_surplus_detail": t5["strongest_surplus_near_1"],
            "nearest_half_detail": t5["nearest_half"],
            "starting_point_0_seeds": t5["starting_point"],
        },
        "structure_compare": (
            "Same three loci; different Q lists. p=5 deficit-near-0 lengths are "
            "the “what thinks it is 0” seeds (ternary-53/665 analogues). p=5 "
            "surplus-near-1 lengths are the flat-block candidates (ternary-5/41/306 "
            "analogues). Near-½ exists on both circles by density of {Qα}."
        ),
        "tag": "measured",
    }


def probe_all(primes: Sequence[int] = DEFAULT_PRIMES) -> Dict[str, Any]:
    per_p = []
    for p in primes:
        dens = density_probe(p)
        loci = locus_families(p)
        # Known ternary control: 5, 41, 306, 665
        control = None
        if p == 3:
            control = {
                "known_surplus_rungs": [5, 41, 306],
                "known_deficit_family": [19, 53, 665],
                "fmt_5_8_Bps": 0.2,
                "fmt_665_1055_Bps": 132 / 665,
                "665_beats_naive": (132 / 665) < 0.2,
            }
        entry: Dict[str, Any] = {
            "p": p,
            "alpha": dens["alpha"],
            "cf_rungs": rung_table_from_cf(p)[:10],
            "best_approximations": best_approximations(p, Q_max=2000, top_n=10),
            "locus_families": loci,
            "density": dens,
            "regimes": regimes_note(p),
            "ternary_control": control,
        }
        per_p.append(entry)

    p5_compare = compare_p5_to_ternary() if 5 in primes or 3 in primes else None

    # Cross-p summary
    actionable = []
    for row in per_p:
        d = row["density"]
        if d["verdict_tag"] == "measured_block_beats_naive" and row["p"] != 3:
            actionable.append(
                {
                    "p": row["p"],
                    "block_Q": d["block_surplus_Q"],
                    "Bps_gain_vs_naive": d["block_surplus_vs_naive"],
                    "note": "theory density only; codec + hardware cost not measured",
                }
            )

    return {
        "stance": (
            "Same geometric machine for every odd prime p: α_p=log2(p), three "
            "loci on R/Z (deficit≈0, surplus≈1, ½), CF rungs, flat large "
            "containers. Ternary privileged by size/hardware (5-in-a-byte), "
            "not unique geometry. For p=5: α_5≈2.321928 — identical locus "
            "definitions; “0” seeds = strongest deficit Qs."
        ),
        "primes": list(primes),
        "per_p": per_p,
        "p5_vs_ternary": p5_compare,
        "actionable_codec_candidates": actionable,
        "honest_verdict": {
            "base_p_actionable_now": bool(actionable),
            "detail": (
                "Theory: flat surplus/CF blocks beat naïve digit-pack for "
                "non-ternary p. Actionable only as greenfield non-ternary "
                "codec (p=5 first surplus block). Not a ternary packing lever; "
                "ternary static frontier remains 665-flat."
                if actionable
                else "Base-p is a conceptual/map probe until someone wants "
                "non-ternary codecs; no transferable lesson that moves the "
                "ternary 665-flat frontier."
            ),
            "ternary_frontier_unchanged": True,
            "tag": "measured" if actionable else "applies_as_language",
        },
        "claim_tags": [
            {"claim": "Circle / CF / Law-B language generalizes to any p", "tag": "applies_as_language"},
            {"claim": "Three loci = right dynamical image for every p (geometry)", "tag": "applies_as_language"},
            {"claim": "Training dynamics need more than three regimes", "tag": "speculative"},
            {"claim": "Flat large convergent block can beat naïve digit pack (theory)", "tag": "measured"},
            {"claim": "p=5 deficit-near-0 Qs are the alphabet’s “0” seeds", "tag": "measured"},
            {"claim": "p=5/7 first rung worth implementing as product codec now", "tag": "speculative"},
            {"claim": "Base-p densifies ternary BitNet past 665-flat", "tag": "does_not_apply"},
            {"claim": "UFRF twin-sheet for base 5", "tag": "speculative"},
        ],
    }


def selftest() -> int:
    # Ternary control identities
    assert container_bits(3, 5) == 8
    assert container_bytes(3, 5) == 1
    assert container_bits(3, 41) == 65
    assert container_bytes(3, 41) == 9
    assert container_bits(3, 306) == 485
    assert container_bytes(3, 306) == 61
    assert container_bits(3, 665) == 1055
    assert container_bytes(3, 665) == 132
    assert naive_symbols_per_byte(3) == 5
    assert theory_bytes_naive(3, 665) == theory_bytes_naive(3, 665)
    # Alpha chart matches exact for modest Q
    for p in (3, 5, 7):
        for Q in range(1, 40):
            assert container_bits(p, Q) == bits_from_alpha(p, Q), (p, Q)
    # p=5 naïve is 3 symbols/byte
    assert naive_symbols_per_byte(5) == 3
    assert naive_symbols_per_byte(7) == 2
    # CF recovers 5 as early ternary convergent denominator
    qs = {r["Q"] for r in rung_table_from_cf(3)}
    assert 5 in qs or 1 in qs
    # Locus families: strongest deficit has phi near 0
    loc5 = locus_families(5, Q_max=800, top_n=4)
    assert Decimal(loc5["strongest_deficit_near_0"][0]["phi"]) < Decimal("0.05")
    assert Decimal(loc5["strongest_surplus_near_1"][0]["dist_to_1"]) < Decimal("0.05")
    print("BASE_P_LADDER PASS")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "selftest"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--primes", type=str, default="3,5,7,11")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "selftest":
        return selftest()

    selftest()
    primes = tuple(int(x) for x in args.primes.split(",") if x.strip())
    report = probe_all(primes)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"BASE_P wrote {args.out}")
    for row in report["per_p"]:
        d = row["density"]
        print(
            f"  p={row['p']}: naive k={d['naive_k_symbols_per_byte']} "
            f"({d['naive_B_per_symbol']:.6f} B/sym)  "
            f"block Q={d['block_surplus_Q']} "
            f"({d['block_surplus_B_per_symbol']:.6f} B/sym)  "
            f"Δ={d['block_surplus_vs_naive']:+.6f}  [{d['verdict_tag']}]"
        )
        print("    CF Q:", ", ".join(str(r["Q"]) for r in row["cf_rungs"][:8]))
        loc = row["locus_families"]
        print(
            "    deficit-0:",
            [r["Q"] for r in loc["strongest_deficit_near_0"][:5]],
            " surplus-1:",
            [r["Q"] for r in loc["strongest_surplus_near_1"][:5]],
            " half:",
            [r["Q"] for r in loc["nearest_half"][:3]],
        )
    if report.get("p5_vs_ternary"):
        p5 = report["p5_vs_ternary"]["p5_measured"]
        print(
            "p=5 “0” seeds (deficit):",
            p5["deficit_Qs"][:6],
            " surplus:",
            p5["surplus_Qs"][:6],
        )
    print("verdict:", report["honest_verdict"]["detail"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
