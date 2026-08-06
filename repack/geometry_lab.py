#!/usr/bin/env python3
"""Geometry lab — only keep moves that cheaply improve tax.

Out-of-box probes (no Collatz predictors, no density claims vs (5,8)):

  A. Coboundary / associator of tax_rows  (symbolic 2→3 lift)
  B. Holonomy Stokes on size triangles   (closed 2-form?)
  C. Reshape lattice: m×n → (m/d)×(n·d)  (volume-preserving, free)
  D. Fiber-only surplus pad vs pad-both / pad-multiplicity
  E. Break-a-square: Q×Q → (Q/d)×(Q·d) when d|Q
  F. Character sketch: Re(e^{2πi {Qα}}) vs tax  (Fourier of phase)

Kill any probe that does not produce a low-cost improvement or a clean
negative (useful null). Write surviving moves into the summary.
"""

from __future__ import annotations

import json
import math
import sys
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harmonic_tax import (
    ALPHA,
    frac_Q_alpha,
    tax_cols_harmonic,
    tax_rows_harmonic,
)
from pad_to_tax0 import pad_toward_surplus_phase
from scale_probe import LLM_SHAPES, axis_bits

getcontext().prec = 120


def best_tax(m: int, n: int) -> int:
    return min(tax_rows_harmonic(m, n), tax_cols_harmonic(m, n))


def holonomy(m: int, n: int) -> int:
    """tax_rows − tax_cols = m·bits(n) − n·bits(m)."""
    return tax_rows_harmonic(m, n) - tax_cols_harmonic(m, n)


# ---------------------------------------------------------------------------
# A. Coboundary / associator
# ---------------------------------------------------------------------------

def associator(m: int, n: int, p: int) -> Dict[str, int]:
    """Does tax behave like a derivation / cocycle under (m,n)·p vs m·(n,p)?

    Exact bits identity:
      bits(mnp) is unique, so
      m·bits(np) − bits(mnp) = tax_rows(m, np)
      (mn)·bits(p) − bits(mnp) = tax_rows(mn, p)
      ...
    Associator residual for row-tax nesting:
      A = tax_rows(m,n)*p? no — compare nested fiber taxes:
      tax_rows(m, n*p)  vs  tax_rows(m*n, p)  (different multiplicities)
    We measure the *associator of container bits*:
      left  = m * bits(n) + (m*n) wait — use circle only:

      nested_rows = tax_rows(m, n) + tax_rows(m*n, p)   # pack n-fibers, then p
      flat_rows   = tax_rows(m, n*p)                    # pack (n*p)-fibers
    These are NOT equal in general — residual = nested − flat.
    """
    # Avoid huge products: only when m*n*p is "bits-safe" via circle (always ok)
    # but m*n and n*p should stay reasonable for interpretation
    mn, np_ = m * n, n * p
    nested = tax_rows_harmonic(m, n) + tax_rows_harmonic(mn, p)
    flat = tax_rows_harmonic(m, np_)
    # alternate nesting
    nested2 = tax_rows_harmonic(n, p) + tax_rows_harmonic(m, np_)  # wrong dims
    # Better alternate: pack p-fibers of size n first then m of (n*p)... use cols path
    # Standard: (m)×(n)×(p) row-nesting vs flat
    return {
        "m": m,
        "n": n,
        "p": p,
        "nested_tax": nested,
        "flat_tax": flat,
        "associator": nested - flat,  # >0 means nesting costs more than flat
        "hol_mn": holonomy(m, n),
        "hol_np": holonomy(n, p),
        "hol_mp": holonomy(m, p),
    }


def run_A(samples: Sequence[Tuple[int, int, int]]) -> Dict[str, Any]:
    rows = [associator(m, n, p) for m, n, p in samples]
    assoc = [r["associator"] for r in rows]
    return {
        "n": len(rows),
        "mean_associator": sum(assoc) / len(assoc),
        "min_associator": min(assoc),
        "max_associator": max(assoc),
        "fraction_nested_worse": sum(1 for a in assoc if a > 0) / len(assoc),
        "fraction_flat_worse": sum(1 for a in assoc if a < 0) / len(assoc),
        "samples": rows[:12],
        # Improvement rule: if associator>0, prefer flat (m)×(np) over nested
        "actionable": "prefer_flat_over_nested_when_associator_positive",
    }


# ---------------------------------------------------------------------------
# B. Holonomy Stokes on triangles
# ---------------------------------------------------------------------------

def stokes(m: int, n: int, p: int) -> int:
    """hol(m,n)+hol(n,p)+hol(p,m). Zero ⟺ closed 2-form on size triangle."""
    return holonomy(m, n) + holonomy(n, p) + holonomy(p, m)


def run_B(triples: Sequence[Tuple[int, int, int]]) -> Dict[str, Any]:
    vals = [stokes(m, n, p) for m, n, p in triples]
    zeros = sum(1 for v in vals if v == 0)
    return {
        "n": len(vals),
        "n_exact_zero": zeros,
        "mean_abs": sum(abs(v) for v in vals) / len(vals),
        "max_abs": max(abs(v) for v in vals),
        "closed": zeros == len(vals),
        "samples": [
            {"m": m, "n": n, "p": p, "stokes": stokes(m, n, p)}
            for m, n, p in triples[:10]
        ],
        "actionable": (
            "holonomy_is_closed_exact"
            if zeros == len(vals)
            else "holonomy_has_flux_use_as_design_signal"
        ),
    }


# ---------------------------------------------------------------------------
# C. Reshape lattice (volume-preserving)
# ---------------------------------------------------------------------------

def divisors(x: int, cap: int = 256) -> List[int]:
    out = []
    for d in range(1, min(int(math.isqrt(x)) + 1, cap + 1)):
        if x % d == 0:
            out.append(d)
            q = x // d
            if q != d and q <= cap:
                out.append(q)
    return sorted(set(out))


def _aspect(a: int, b: int) -> float:
    return max(a, b) / min(a, b)


def reshape_search(
    m: int,
    n: int,
    d_cap: int = 128,
    max_aspect: Optional[float] = None,
) -> Dict[str, Any]:
    """Search (m/d)×(n·d) and (m·d)×(n/d) preserving MN when division exact.

    max_aspect: if set, reject reshapes with max/min > max_aspect (deployable).
    """
    base = best_tax(m, n)
    best = {
        "m": m,
        "n": n,
        "best_tax": base,
        "delta": 0,
        "d": 1,
        "kind": "identity",
        "aspect": _aspect(m, n),
    }
    candidates = [best]

    def consider(m2: int, n2: int, d: int, kind: str) -> None:
        nonlocal best
        if m2 <= 0 or n2 <= 0:
            return
        if max_aspect is not None and _aspect(m2, n2) > max_aspect:
            return
        t = best_tax(m2, n2)
        cand = {
            "m": m2,
            "n": n2,
            "best_tax": t,
            "delta": t - base,
            "d": d,
            "kind": kind,
            "aspect": _aspect(m2, n2),
        }
        candidates.append(cand)
        if t < best["best_tax"]:
            best = cand

    for d in divisors(m, cap=d_cap):
        if d == 1:
            continue
        consider(m // d, n * d, d, "div_m")
    for d in divisors(n, cap=d_cap):
        if d == 1:
            continue
        consider(m * d, n // d, d, "div_n")
    improved = [c for c in candidates if c["delta"] < 0]
    improved.sort(key=lambda c: c["delta"])
    return {
        "baseline": {"m": m, "n": n, "best_tax": base, "aspect": _aspect(m, n)},
        "best": best,
        "n_improved": len(improved),
        "top": improved[:5],
        "max_aspect": max_aspect,
    }


def run_C(shapes: Sequence[Tuple[str, int, int]]) -> Dict[str, Any]:
    """Unconstrained (math) + aspect-bounded (deployable, max_aspect=16)."""
    rows_free = []
    rows_bound = []
    for name, m, n in shapes:
        free = reshape_search(m, n, max_aspect=None)
        free["name"] = name
        rows_free.append(free)
        bound = reshape_search(m, n, max_aspect=16.0)
        bound["name"] = name
        rows_bound.append(bound)

    def pack(rows: List[Dict[str, Any]], tag: str) -> Dict[str, Any]:
        wins = sum(1 for r in rows if r["best"]["delta"] < 0)
        return {
            "tag": tag,
            "n": len(rows),
            "n_improved": wins,
            "mean_best_delta": sum(r["best"]["delta"] for r in rows) / len(rows),
            "shapes": rows,
            "actionable": (
                "reshape_before_pad" if wins else "reshape_null_under_constraint"
            ),
        }

    free = pack(rows_free, "unconstrained")
    bound = pack(rows_bound, "max_aspect_16")
    return {
        "unconstrained": free,
        "bounded": bound,
        # Prefer deployable signal for survivor gate
        "n_improved": bound["n_improved"],
        "mean_best_delta": bound["mean_best_delta"],
        "n_improved_unconstrained": free["n_improved"],
        "actionable": bound["actionable"],
        "shapes": rows_bound,
    }


# ---------------------------------------------------------------------------
# D. Fiber-only surplus pad
# ---------------------------------------------------------------------------

def fiber_pad_strategies(m: int, n: int, max_pad: int = 64) -> Dict[str, Any]:
    """Compare: pad fiber only / multiplicity only / both / none.

    Fiber = the mode being amplified in the winning orientation.
    If tax_rows <= tax_cols, fiber is n (rows amplify φ_n); else fiber is m.
    """
    tr, tc = tax_rows_harmonic(m, n), tax_cols_harmonic(m, n)
    if tr <= tc:
        fiber, mult, orient = n, m, "rows"
    else:
        fiber, mult, orient = m, n, "cols"

    sp_f = pad_toward_surplus_phase(fiber, max_pad=max_pad)
    sp_u = pad_toward_surplus_phase(mult, max_pad=max_pad)
    f2, u2 = sp_f["L_prime"], sp_u["L_prime"]

    def tax_pair(a: int, b: int) -> int:
        # restore orientation: if rows, shape is (mult, fiber)=(m,n)
        if orient == "rows":
            return best_tax(a, b)  # a=mult, b=fiber
        return best_tax(b, a)

    # carefully:
    if orient == "rows":
        none_t = best_tax(m, n)
        fiber_t = best_tax(m, f2)
        mult_t = best_tax(u2, n)
        both_t = best_tax(u2, f2)
        shapes = {
            "none": (m, n),
            "fiber": (m, f2),
            "mult": (u2, n),
            "both": (u2, f2),
        }
        pads = {
            "none": 0,
            "fiber": sp_f["pad_trits"],
            "mult": sp_u["pad_trits"],
            "both": sp_f["pad_trits"] + sp_u["pad_trits"],
        }
        taxes = {
            "none": none_t,
            "fiber": fiber_t,
            "mult": mult_t,
            "both": both_t,
        }
    else:
        none_t = best_tax(m, n)
        fiber_t = best_tax(f2, n)
        mult_t = best_tax(m, u2)
        both_t = best_tax(f2, u2)
        shapes = {
            "none": (m, n),
            "fiber": (f2, n),
            "mult": (m, u2),
            "both": (f2, u2),
        }
        pads = {
            "none": 0,
            "fiber": sp_f["pad_trits"],
            "mult": sp_u["pad_trits"],
            "both": sp_f["pad_trits"] + sp_u["pad_trits"],
        }
        taxes = {
            "none": none_t,
            "fiber": fiber_t,
            "mult": mult_t,
            "both": both_t,
        }

    # efficiency: tax drop per pad trit
    eff = {}
    for k in ("fiber", "mult", "both"):
        pad = pads[k]
        drop = taxes["none"] - taxes[k]
        eff[k] = (drop / pad) if pad > 0 else (drop if drop > 0 else 0.0)

    best_eff = max(eff, key=lambda k: eff[k])
    return {
        "m": m,
        "n": n,
        "orient": orient,
        "fiber": fiber,
        "mult": mult,
        "taxes": taxes,
        "pads": pads,
        "shapes": {k: list(v) for k, v in shapes.items()},
        "efficiency": eff,
        "best_efficiency": best_eff,
        "fiber_beats_both_tax": taxes["fiber"] <= taxes["both"],
        "fiber_beats_mult_tax": taxes["fiber"] < taxes["mult"],
    }


def run_D(shapes: Sequence[Tuple[str, int, int]], max_pad: int = 64) -> Dict[str, Any]:
    rows = []
    for name, m, n in shapes:
        r = fiber_pad_strategies(m, n, max_pad=max_pad)
        r["name"] = name
        rows.append(r)
    fiber_best = sum(1 for r in rows if r["best_efficiency"] == "fiber")
    both_best = sum(1 for r in rows if r["best_efficiency"] == "both")
    mult_best = sum(1 for r in rows if r["best_efficiency"] == "mult")
    fiber_ge_both = sum(1 for r in rows if r["fiber_beats_both_tax"])
    return {
        "n": len(rows),
        "fiber_best_eff": fiber_best,
        "both_best_eff": both_best,
        "mult_best_eff": mult_best,
        "fiber_tax_le_both": fiber_ge_both,
        "shapes": rows,
        "actionable": (
            "pad_fiber_only_default"
            if fiber_best >= both_best and fiber_best >= mult_best
            else "pad_both_sometimes_needed"
        ),
    }


# ---------------------------------------------------------------------------
# E. Break-a-square
# ---------------------------------------------------------------------------

def break_square(
    Q: int, d_cap: int = 64, max_aspect: Optional[float] = 16.0
) -> Dict[str, Any]:
    base = best_tax(Q, Q)
    best = {"d": 1, "shape": [Q, Q], "best_tax": base, "delta": 0, "aspect": 1.0}
    improved = []
    for d in divisors(Q, cap=d_cap):
        if d == 1:
            continue
        m2, n2 = Q // d, Q * d
        if max_aspect is not None and _aspect(m2, n2) > max_aspect:
            continue
        t = best_tax(m2, n2)
        cand = {
            "d": d,
            "shape": [m2, n2],
            "best_tax": t,
            "delta": t - base,
            "aspect": _aspect(m2, n2),
        }
        if t < base:
            improved.append(cand)
        if t < best["best_tax"]:
            best = cand
    improved.sort(key=lambda c: c["delta"])
    return {
        "Q": Q,
        "phi": format(frac_Q_alpha(Q), "f"),
        "baseline_tax": base,
        "best": best,
        "n_improved": len(improved),
        "top": improved[:5],
        "max_aspect": max_aspect,
    }


def run_E(Qs: Sequence[int] = (640, 2560, 4096, 5120, 8192)) -> Dict[str, Any]:
    rows = [break_square(Q, max_aspect=16.0) for Q in Qs]
    rows_free = [break_square(Q, max_aspect=None) for Q in Qs]
    wins = sum(1 for r in rows if r["best"]["delta"] < 0)
    wins_free = sum(1 for r in rows_free if r["best"]["delta"] < 0)
    return {
        "n": len(rows),
        "n_improved": wins,
        "n_improved_unconstrained": wins_free,
        "squares": rows,
        "squares_unconstrained_best": [
            {"Q": r["Q"], "delta": r["best"]["delta"], "shape": r["best"]["shape"]}
            for r in rows_free
        ],
        "actionable": (
            "break_square_before_pack" if wins else "break_square_null_under_aspect_16"
        ),
    }


# ---------------------------------------------------------------------------
# F. Character sketch
# ---------------------------------------------------------------------------

def run_F(Qs: Sequence[int], m_fixed: int = 41) -> Dict[str, Any]:
    """Correlate tax_rows(m_fixed, Q) with cos(2π φ) and sin(2π φ)."""
    import statistics

    taxes = []
    cos_p = []
    sin_p = []
    surplus = []
    for Q in Qs:
        phi = float(frac_Q_alpha(Q))
        taxes.append(tax_rows_harmonic(m_fixed, Q))
        cos_p.append(math.cos(2 * math.pi * phi))
        sin_p.append(math.sin(2 * math.pi * phi))
        surplus.append(phi)  # raw phase

    def pearson(a: List[float], b: List[float]) -> float:
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        da = math.sqrt(sum((x - ma) ** 2 for x in a))
        db = math.sqrt(sum((y - mb) ** 2 for y in b))
        return num / (da * db) if da and db else 0.0

    # Also distance to 1 (known strong)
    dist1 = [1.0 - float(frac_Q_alpha(Q)) for Q in Qs]
    return {
        "m_fixed": m_fixed,
        "n_Q": len(Qs),
        "r_tax_cos": pearson([float(t) for t in taxes], cos_p),
        "r_tax_sin": pearson([float(t) for t in taxes], sin_p),
        "r_tax_phi": pearson([float(t) for t in taxes], surplus),
        "r_tax_dist_to_1": pearson([float(t) for t in taxes], dist1),
        "actionable": (
            "character_adds_signal"
            if abs(pearson([float(t) for t in taxes], cos_p))
            > abs(pearson([float(t) for t in taxes], dist1))
            else "surplus_dist_dominates_character"
        ),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _triples_from_shapes() -> List[Tuple[int, int, int]]:
    modes = sorted({m for _, m, n in LLM_SHAPES} | {n for _, m, n in LLM_SHAPES})
    out = []
    for i, m in enumerate(modes):
        for j, n in enumerate(modes):
            if j == i:
                continue
            for k, p in enumerate(modes):
                if k == i or k == j:
                    continue
                if m * n > 500_000 or n * p > 500_000:
                    continue
                out.append((m, n, p))
                if len(out) >= 40:
                    return out
    return out


def run_all(max_pad: int = 64) -> Dict[str, Any]:
    triples = _triples_from_shapes()
    # denser small triples for associator
    small = [
        (5, 8, 7),
        (5, 41, 3),
        (8, 5, 41),
        (41, 19, 5),
        (53, 306, 2),
        (306, 53, 5),
        (128, 64, 32),
        (256, 16, 16),
        (64, 64, 16),
        (100, 10, 10),
    ] + triples[:20]

    A = run_A(small)
    B = run_B(small)
    C = run_C(LLM_SHAPES)
    D = run_D(LLM_SHAPES, max_pad=max_pad)
    E = run_E()
    Qs = list(range(1, 400)) + [640, 2560, 4096, 5120, 6912, 8192]
    F = run_F(Qs, m_fixed=41)

    survivors = []
    kills = []
    if A["fraction_nested_worse"] > 0.55:
        survivors.append(
            {
                "id": "A",
                "move": "prefer flat (m)×(n·p) over nested tax when associator>0",
                "evidence": A,
            }
        )
    else:
        kills.append({"id": "A", "reason": "associator not systematically positive"})

    if B["closed"]:
        survivors.append(
            {
                "id": "B",
                "move": "holonomy is an exact coboundary (Stokes=0) — axis math path-independent",
                "evidence": {"n_exact_zero": B["n_exact_zero"], "n": B["n"]},
            }
        )
    else:
        # Non-closed is interesting symbolically but no cheap move yet — do not promote.
        kills.append(
            {
                "id": "B",
                "reason": (
                    "Stokes holonomy flux nonzero (mean_abs="
                    f"{B['mean_abs']:.1f}) but no low-cost design rule yet"
                ),
            }
        )

    if C["n_improved"] > 0:
        survivors.append(
            {
                "id": "C",
                "move": "aspect-bounded reshape (≤16) before pad",
                "evidence": {
                    "n_improved": C["n_improved"],
                    "mean_best_delta": C["mean_best_delta"],
                    "n_improved_unconstrained": C["n_improved_unconstrained"],
                },
            }
        )
    else:
        kills.append(
            {
                "id": "C",
                "reason": (
                    "no reshape improved suite under aspect≤16 "
                    f"(unconstrained wins={C['n_improved_unconstrained']} — extreme aspect only)"
                ),
            }
        )

    if D["actionable"] == "pad_fiber_only_default":
        survivors.append(
            {
                "id": "D",
                "move": "pad fiber-only toward surplus (best tax/pad efficiency)",
                "evidence": {
                    "fiber_best_eff": D["fiber_best_eff"],
                    "both_best_eff": D["both_best_eff"],
                    "mult_best_eff": D["mult_best_eff"],
                },
            }
        )
    else:
        survivors.append(
            {
                "id": "D",
                "move": D["actionable"],
                "evidence": {
                    "fiber_best_eff": D["fiber_best_eff"],
                    "both_best_eff": D["both_best_eff"],
                },
            }
        )

    if E["n_improved"] > 0:
        survivors.append(
            {
                "id": "E",
                "move": "break squares into rectangles (aspect≤16)",
                "evidence": {
                    "n_improved": E["n_improved"],
                    "n_improved_unconstrained": E["n_improved_unconstrained"],
                },
            }
        )
    else:
        kills.append(
            {
                "id": "E",
                "reason": (
                    "break-square null under aspect≤16 "
                    f"(unconstrained wins={E['n_improved_unconstrained']} — extreme only)"
                ),
            }
        )

    if F["actionable"] == "character_adds_signal":
        survivors.append({"id": "F", "move": "use cos(2πφ) character", "evidence": F})
    else:
        kills.append(
            {
                "id": "F",
                "reason": "dist-to-1 dominates Fourier character",
                "r_dist": F["r_tax_dist_to_1"],
                "r_cos": F["r_tax_cos"],
            }
        )

    return {
        "stance": (
            "Geometry lab: keep only testable low-cost improvements. "
            "Nulls are progress."
        ),
        "A_associator": {k: A[k] for k in A if k != "samples"} | {"samples": A["samples"]},
        "B_stokes": B,
        "C_reshape": {k: C[k] for k in C if k != "shapes"}
        | {"shapes": C["shapes"]},
        "D_fiber_pad": {k: D[k] for k in D if k != "shapes"}
        | {"shapes": D["shapes"]},
        "E_break_square": E,
        "F_character": F,
        "survivors": survivors,
        "kills": kills,
    }


def selftest() -> None:
    a = associator(5, 8, 7)
    assert "associator" in a
    assert stokes(5, 8, 7) == holonomy(5, 8) + holonomy(8, 7) + holonomy(7, 5)
    r = reshape_search(2560, 2560, d_cap=32)
    assert r["baseline"]["best_tax"] >= 0
    f = fiber_pad_strategies(2560, 6912, max_pad=16)
    assert f["taxes"]["none"] >= 0
    print("geometry_lab selftest OK")


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: geometry_lab.py {selftest|run} [--max-pad N]")
        return 0
    cmd = argv[0]
    max_pad = 64
    if "--max-pad" in argv:
        max_pad = int(argv[argv.index("--max-pad") + 1])
    if cmd == "selftest":
        selftest()
        return 0
    if cmd == "run":
        out = run_all(max_pad=max_pad)
        path = Path(__file__).resolve().parent / "geometry_lab_results.json"
        summary = {
            "survivors": [
                {"id": s["id"], "move": s["move"], "evidence": s.get("evidence")}
                for s in out["survivors"]
            ],
            "kills": out["kills"],
            "A": {
                "mean_associator": out["A_associator"]["mean_associator"],
                "fraction_nested_worse": out["A_associator"]["fraction_nested_worse"],
            },
            "B": {
                "closed": out["B_stokes"]["closed"],
                "n_exact_zero": out["B_stokes"]["n_exact_zero"],
                "n": out["B_stokes"]["n"],
                "mean_abs": out["B_stokes"]["mean_abs"],
            },
            "C": {
                "n_improved_aspect16": out["C_reshape"]["n_improved"],
                "mean_best_delta_aspect16": out["C_reshape"]["mean_best_delta"],
                "n_improved_unconstrained": out["C_reshape"]["n_improved_unconstrained"],
                "actionable": out["C_reshape"]["actionable"],
            },
            "D": {
                "actionable": out["D_fiber_pad"]["actionable"],
                "fiber_best_eff": out["D_fiber_pad"]["fiber_best_eff"],
                "both_best_eff": out["D_fiber_pad"]["both_best_eff"],
                "mult_best_eff": out["D_fiber_pad"]["mult_best_eff"],
            },
            "E": {
                "n_improved_aspect16": out["E_break_square"]["n_improved"],
                "n_improved_unconstrained": out["E_break_square"][
                    "n_improved_unconstrained"
                ],
                "actionable": out["E_break_square"]["actionable"],
                "squares": [
                    {
                        "Q": s["Q"],
                        "delta": s["best"]["delta"],
                        "shape": s["best"]["shape"],
                    }
                    for s in out["E_break_square"]["squares"]
                ],
            },
            "F": out["F_character"],
        }
        print(json.dumps(summary, indent=2))
        path.write_text(json.dumps(out, indent=2))
        print(f"wrote {path}")
        return 0
    print(f"unknown: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
