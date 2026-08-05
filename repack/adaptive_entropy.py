#!/usr/bin/env python3
"""Per-tensor adaptive (static empirical) range coding vs fixed ternary formats.

HARD RULE: verdict seams use exact integers (counts, byte sizes, conservation
asserts). Floats only for display ratios (H bits/weight, savings %, bpw).

QUESTION: how many bytes does per-tensor ADAPTIVE range coding save on a real
trained ternary checkpoint, vs the fixed uniform-ternary formats?

INPUT: ckpt_ternary_384.npz — int8 arrays q__* with values in {-1,0,1};
gamma__* scalars skipped.

Formats compared (bytes per tensor / totals):
  (a) 2-bit:           ceil(n/4)
  (b) (5,8) 5-per-byte: ceil(n/5)
  (c) uniform-ternary ideal: ceil(C_n / 8), C_n = (3**n).bit_length()
  (d) adaptive bound:  ceil(n*H/8) + 2
  + REAL adaptive static range-coder byte count (round-trip asserted)

House gates: nulls printed beside claimed rates; count conservation asserts.
"""

from __future__ import annotations

import math
import sys
from decimal import ROUND_FLOOR, Decimal, getcontext
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Exact integer constants (mirror pack_ladder.py fmt_stream)
# ---------------------------------------------------------------------------

STREAM_RENORM_BITS = 32
STREAM_FULL = 1 << STREAM_RENORM_BITS  # 2**32
STREAM_MASK = STREAM_FULL - 1
STREAM_TOP = 1 << 24  # renorm while range < 2**24
STREAM_FLUSH_SHIFTS = 5
STREAM_INIT_BYTES = 5

# Decimal for exact C_n = floor(n*log2(3))+1 = (3**n).bit_length()
# (3**n never a power of two). prec=80 covers n up to > 10**7.
getcontext().prec = 80
_LOG2_3 = Decimal(3).ln() / Decimal(2).ln()
LOG2_3_F = float(_LOG2_3)  # display only
NULL_P0 = 1.0 / 3.0  # uniform-ternary null for p0

CKPT_PATH = "ckpt_ternary_384.npz"


# ---------------------------------------------------------------------------
# Size theory (exact integers)
# ---------------------------------------------------------------------------

def theory_bytes_2bit(n: int) -> int:
    """ceil(n/4) bytes."""
    return (n + 3) // 4


def theory_bytes_5_8(n: int) -> int:
    """ceil(n/5) bytes."""
    return (n + 4) // 5


def container_bits_uniform(n: int) -> int:
    """C_n = minimal P with 3**n <= 2**P = (3**n).bit_length() for n >= 1."""
    if n <= 0:
        return 0
    x = Decimal(n) * _LOG2_3
    return int(x.to_integral_value(rounding=ROUND_FLOOR)) + 1


def theory_bytes_uniform_ideal(n: int) -> int:
    """ceil(C_n / 8) with C_n = (3**n).bit_length()."""
    if n <= 0:
        return 0
    return (container_bits_uniform(n) + 7) // 8


def empirical_H_bits(counts: Sequence[int], n: int) -> float:
    """Shannon entropy H in bits/symbol; 0*log0 := 0. Display float."""
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            h -= p * math.log2(p)
    return h


def adaptive_bound_bytes(n: int, H: float) -> int:
    """ceil(n*H/8) + 2 — static AC bound (within 2 bytes of n*H/8)."""
    if n <= 0:
        return 0
    return math.ceil(n * H / 8.0) + 2


# ---------------------------------------------------------------------------
# Static range coder — same renorm / carry design as fmt_stream, but
# counts-proportional interval split (empirical static distribution).
# ---------------------------------------------------------------------------

def _cum_counts(counts: Sequence[int]) -> Tuple[int, int, int, int]:
    """Return (c0, c1, c2, total) cumulative ends for symbols 0,1,2."""
    c0, c1, c2 = int(counts[0]), int(counts[1]), int(counts[2])
    return c0, c0 + c1, c0 + c1 + c2, c0 + c1 + c2


class _StaticEncoder:
    """Carry-counting range encoder with fixed counts-proportional splits."""

    __slots__ = ("low", "range", "cache", "ffnum", "out", "cum", "total")

    def __init__(self, counts: Sequence[int]) -> None:
        c1, c2, c3, total = _cum_counts(counts)
        if total <= 0:
            raise ValueError("static encoder requires total counts > 0")
        # cum[s] = start of symbol s; cum ends at total
        self.cum = (0, c1, c2, c3)  # c1=count0, c2=count0+count1, c3=total
        self.total = total
        self.low = 0
        self.range = STREAM_FULL
        self.cache = 0
        self.ffnum = 0
        self.out = bytearray()

    def _shift_low(self) -> None:
        top = self.low >> 24
        if top != 0xFF:
            carry = self.low >> 32
            self.out.append((self.cache + carry) & 0xFF)
            if self.ffnum:
                cbyte = (0xFF + carry) & 0xFF
                for _ in range(self.ffnum):
                    self.out.append(cbyte)
                self.ffnum = 0
            self.cache = top & 0xFF
        else:
            self.ffnum += 1
        self.low = (self.low << 8) & STREAM_MASK
        self.range <<= 8

    def encode(self, s: int) -> None:
        """Encode symbol s in {0,1,2} with counts-proportional bounds."""
        total = self.total
        r = self.range
        lo = self.cum[s]
        hi = self.cum[s + 1]
        # Exact integer proportional split; last open interval absorbs rem
        # when hi == total: (r * total) // total == r.
        new_low = self.low + (r * lo) // total
        new_hi = self.low + (r * hi) // total
        self.low = new_low
        self.range = new_hi - new_low
        if self.range <= 0:
            raise RuntimeError(
                f"collapsed range for symbol {s}: lo={lo} hi={hi} r={r} total={total}"
            )
        while self.range < STREAM_TOP:
            self._shift_low()

    def finish(self) -> bytes:
        for _ in range(STREAM_FLUSH_SHIFTS):
            self._shift_low()
        return bytes(self.out)


class _StaticDecoder:
    """Mirror of _StaticEncoder; needs (data, n, counts)."""

    __slots__ = ("data", "pos", "low", "range", "code", "cum", "total")

    def __init__(self, data: bytes, counts: Sequence[int]) -> None:
        c1, c2, c3, total = _cum_counts(counts)
        if total <= 0:
            raise ValueError("static decoder requires total counts > 0")
        self.cum = (0, c1, c2, c3)
        self.total = total
        self.data = data
        self.pos = 0
        self.low = 0
        self.range = STREAM_FULL
        self.code = 0
        for _ in range(STREAM_INIT_BYTES):
            self.code = ((self.code << 8) | self._read()) & STREAM_MASK

    def _read(self) -> int:
        if self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            return b
        return 0

    def decode(self) -> int:
        total = self.total
        r = self.range
        # Unsigned 32-bit distance of code from low.
        v = (self.code - self.low) & STREAM_MASK
        # Find symbol: largest s with (r * cum[s]) // total <= v
        # Three-way: check s=0,1,2 via cumulative thresholds.
        t0 = (r * self.cum[1]) // total  # end of symbol 0
        t1 = (r * self.cum[2]) // total  # end of symbol 1
        if v < t0:
            s = 0
        elif v < t1:
            s = 1
        else:
            s = 2
        lo = self.cum[s]
        hi = self.cum[s + 1]
        self.low = (self.low + (r * lo) // total) & STREAM_MASK
        # range update mirrors encoder; low was masked so recompute carefully.
        # Encoder: low += (r*lo)//total (unmasked until shift); range = (r*hi)//total - (r*lo)//total
        # Decoder keeps low masked; range is identical integer.
        new_range = (r * hi) // total - (r * lo) // total
        self.range = new_range
        if self.range <= 0:
            raise RuntimeError(f"decoder collapsed range for symbol {s}")
        while self.range < STREAM_TOP:
            self.low = (self.low << 8) & STREAM_MASK
            self.code = ((self.code << 8) | self._read()) & STREAM_MASK
            self.range <<= 8
        return s


def weights_to_trits(w: np.ndarray) -> np.ndarray:
    """Map w in {-1,0,1} -> t in {0,1,2} as int64."""
    return w.astype(np.int64) + 1


def trits_to_weights(t: np.ndarray) -> np.ndarray:
    """Map t in {0,1,2} -> w in {-1,0,1} as int8."""
    return (t.astype(np.int64) - 1).astype(np.int8)


def count_trits(trits: np.ndarray) -> Tuple[int, int, int]:
    """Exact counts of symbols 0,1,2."""
    t = np.asarray(trits, dtype=np.int64).reshape(-1)
    # bincount length 3
    bc = np.bincount(t, minlength=3)
    return int(bc[0]), int(bc[1]), int(bc[2])


def count_weights(w: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (n_m1, n_0, n_p1, n) with conservation assert."""
    a = np.asarray(w).reshape(-1)
    if a.dtype != np.int8:
        a = a.astype(np.int8, copy=False)
    if a.size and not (np.all(a >= -1) and np.all(a <= 1)):
        bad = a[(a < -1) | (a > 1)]
        raise ValueError(f"weights must be in {{-1,0,1}}; found e.g. {int(bad[0])}")
    n = int(a.size)
    n_m1 = int(np.sum(a == -1))
    n_0 = int(np.sum(a == 0))
    n_p1 = int(np.sum(a == 1))
    assert n_m1 + n_0 + n_p1 == n, (
        f"conservation counts: {n_m1}+{n_0}+{n_p1} != {n}"
    )
    return n_m1, n_0, n_p1, n


def static_pack(trits: np.ndarray, counts: Sequence[int] | None = None) -> bytes:
    """Encode trits {0,1,2} with static counts-proportional range coder."""
    t = np.asarray(trits, dtype=np.int64).reshape(-1)
    n = int(t.size)
    if n == 0:
        return b""
    if counts is None:
        counts = count_trits(t)
    total = int(counts[0]) + int(counts[1]) + int(counts[2])
    assert total == n, f"counts sum {total} != n {n}"
    enc = _StaticEncoder(counts)
    # Hot loop: pure Python ints (exact).
    tt = t.tolist()  # faster than repeated int(t[i]) for large n
    encode = enc.encode
    for s in tt:
        encode(s)
    return enc.finish()


def static_unpack(data: bytes, n: int, counts: Sequence[int]) -> np.ndarray:
    """Decode n trits; assert counts sum == n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    total = int(counts[0]) + int(counts[1]) + int(counts[2])
    assert total == n, f"counts sum {total} != n {n}"
    if not data:
        raise ValueError(f"static stream: empty data for n={n}")
    dec = _StaticDecoder(data, counts)
    out = np.empty(n, dtype=np.int64)
    decode = dec.decode
    for i in range(n):
        out[i] = decode()
    return out


def static_roundtrip(trits: np.ndarray) -> Tuple[bytes, int]:
    """Pack + unpack; assert exact equality. Returns (payload, nbytes)."""
    t = np.asarray(trits, dtype=np.int64).reshape(-1)
    n = int(t.size)
    counts = count_trits(t)
    assert sum(counts) == n
    data = static_pack(t, counts)
    back = static_unpack(data, n, counts)
    assert back.shape == t.shape
    assert np.array_equal(back, t), "static range coder round-trip failed"
    return data, len(data)


# ---------------------------------------------------------------------------
# Per-tensor / totals analysis
# ---------------------------------------------------------------------------

def analyze_tensor(name: str, w: np.ndarray) -> Dict:
    n_m1, n_0, n_p1, n = count_weights(w)
    counts = (n_m1, n_0, n_p1)  # trit symbols 0,1,2
    H = empirical_H_bits(counts, n)
    b_2bit = theory_bytes_2bit(n)
    b_58 = theory_bytes_5_8(n)
    b_uni = theory_bytes_uniform_ideal(n)
    b_bound = adaptive_bound_bytes(n, H)
    trits = weights_to_trits(np.asarray(w).reshape(-1))
    _, b_real = static_roundtrip(trits)
    # house: real should sit near bound (within a few bytes of nH/8, bound is +2)
    return {
        "name": name,
        "n": n,
        "n_m1": n_m1,
        "n_0": n_0,
        "n_p1": n_p1,
        "H": H,
        "b_2bit": b_2bit,
        "b_58": b_58,
        "b_uni": b_uni,
        "b_bound": b_bound,
        "b_real": b_real,
    }


def load_q_tensors(path: str) -> List[Tuple[str, np.ndarray]]:
    z = np.load(path)
    keys = sorted(k for k in z.files if k.startswith("q__"))
    out: List[Tuple[str, np.ndarray]] = []
    for k in keys:
        out.append((k, z[k]))
    return out


def print_tensor_table(rows: List[Dict]) -> None:
    print("--- 1) PER-TENSOR ---")
    hdr = (
        f"{'tensor':<32} {'n':>10} {'n_m1':>9} {'n_0':>9} {'n_p1':>9} "
        f"{'H':>7} {'2bit':>9} {'5/8':>9} {'uni_id':>9} "
        f"{'ad_bnd':>9} {'ad_real':>9}"
    )
    print(hdr)
    for r in rows:
        short = r["name"]
        if short.startswith("q__"):
            short = short[3:]
        if len(short) > 32:
            short = short[:29] + "..."
        print(
            f"{short:<32} {r['n']:>10d} {r['n_m1']:>9d} {r['n_0']:>9d} {r['n_p1']:>9d} "
            f"{r['H']:>7.4f} {r['b_2bit']:>9d} {r['b_58']:>9d} {r['b_uni']:>9d} "
            f"{r['b_bound']:>9d} {r['b_real']:>9d}"
        )


def print_totals(rows: List[Dict]) -> Dict:
    print()
    print("--- 2) TOTALS (whole checkpoint) ---")
    n = sum(r["n"] for r in rows)
    n_m1 = sum(r["n_m1"] for r in rows)
    n_0 = sum(r["n_0"] for r in rows)
    n_p1 = sum(r["n_p1"] for r in rows)
    # conservation of counts across tensors
    assert n_m1 + n_0 + n_p1 == n, (
        f"global conservation: {n_m1}+{n_0}+{n_p1} != {n}"
    )
    b_2bit = sum(r["b_2bit"] for r in rows)
    b_58 = sum(r["b_58"] for r in rows)
    b_uni = sum(r["b_uni"] for r in rows)
    b_bound = sum(r["b_bound"] for r in rows)
    b_real = sum(r["b_real"] for r in rows)
    # global H from pooled counts (for context)
    H_global = empirical_H_bits((n_m1, n_0, n_p1), n)
    p0 = n_0 / n if n else 0.0

    print(
        f"  tensors={len(rows)}  n={n}  counts (m1,0,p1)=({n_m1},{n_0},{n_p1})"
    )
    print(f"  CONSERVATION: n_m1+n_0+n_p1 == n  ->  True  (exact integer assert)")
    print(
        f"  sizes bytes:  2bit={b_2bit}  5/8={b_58}  uni_ideal={b_uni}  "
        f"ad_bound={b_bound}  ad_real={b_real}"
    )

    # savings rates vs null = 0 savings (adaptive same as baseline)
    def sav_pct(base: int, adapt: int) -> float:
        return 100.0 * (base - adapt) / base if base else 0.0

    sav_58 = sav_pct(b_58, b_real)
    sav_uni = sav_pct(b_uni, b_real)
    null_sav = 0.0  # null: uniform p, adaptive == uniform ideal → 0% save
    print(
        f"  SAVINGS ad_real vs 5/8:     {sav_58:+.4f}%   "
        f"(null={null_sav:.1f}% under uniform p0=1/3)"
    )
    print(
        f"  SAVINGS ad_real vs uni_id:  {sav_uni:+.4f}%   "
        f"(null={null_sav:.1f}% under uniform p0=1/3)"
    )
    print(
        f"  GATE: p0={p0:.6f}  null_p0={NULL_P0:.6f}  "
        f"delta_p0={p0 - NULL_P0:+.6f}  H_global={H_global:.6f}  "
        f"log2(3)={LOG2_3_F:.6f}"
    )
    # bpw for display
    def bpw(nbytes: int) -> float:
        return 8.0 * nbytes / n if n else 0.0

    print(
        f"  bpw:  2bit={bpw(b_2bit):.4f}  5/8={bpw(b_58):.4f}  "
        f"uni={bpw(b_uni):.4f}  ad_real={bpw(b_real):.4f}  H={H_global:.4f}"
    )
    return {
        "n": n,
        "n_m1": n_m1,
        "n_0": n_0,
        "n_p1": n_p1,
        "p0": p0,
        "H_global": H_global,
        "b_2bit": b_2bit,
        "b_58": b_58,
        "b_uni": b_uni,
        "b_bound": b_bound,
        "b_real": b_real,
        "sav_58": sav_58,
        "sav_uni": sav_uni,
    }


def synthetic_row(p0: float, n: int, seed: int) -> Dict:
    """Symmetric p_m1 = p_p1 = (1-p0)/2; real adaptive vs uniform ideal."""
    p_side = (1.0 - p0) / 2.0
    rng = np.random.default_rng(seed)
    # sample trits: 0=-1, 1=0, 2=+1 with probs (p_side, p0, p_side)
    trits = rng.choice(3, size=n, p=[p_side, p0, p_side]).astype(np.int64)
    counts = count_trits(trits)
    assert sum(counts) == n
    H = empirical_H_bits(counts, n)
    b_uni = theory_bytes_uniform_ideal(n)
    b_bound = adaptive_bound_bytes(n, H)
    _, b_real = static_roundtrip(trits)
    bpw_real = 8.0 * b_real / n
    bpw_uni = 8.0 * b_uni / n
    bpw_H = H
    sav = 100.0 * (b_uni - b_real) / b_uni if b_uni else 0.0
    return {
        "p0": p0,
        "n": n,
        "counts": counts,
        "H": H,
        "b_uni": b_uni,
        "b_bound": b_bound,
        "b_real": b_real,
        "bpw_real": bpw_real,
        "bpw_uni": bpw_uni,
        "bpw_H": bpw_H,
        "sav_pct": sav,
    }


def print_sensitivity(ckpt_p0: float) -> None:
    print()
    print("--- 3) CONTEXT + SYNTHETIC SENSITIVITY ---")
    print(
        f"  checkpoint global p0={ckpt_p0:.6f}  "
        f"(null uniform p0={NULL_P0:.6f}; expect ~0.33 → small adaptive savings)"
    )
    print(
        f"  synthetic: n=10^6 seeded trits, p_m1=p_p1=(1-p0)/2; "
        f"real adaptive coder vs uniform-ternary ideal"
    )
    print(
        f"  {'p0':>6}  {'n_m1':>8}  {'n_0':>8}  {'n_p1':>8}  "
        f"{'H':>7}  {'uni_B':>9}  {'ad_B':>9}  {'ad_real':>9}  "
        f"{'bpw_uni':>8}  {'bpw_ad':>8}  {'bpw_H':>7}  {'save%':>8}"
    )
    n = 10**6
    seed_base = 20260804
    for i, p0 in enumerate((0.35, 0.4, 0.5, 0.6, 0.7)):
        row = synthetic_row(p0, n, seed=seed_base + i)
        c0, c1, c2 = row["counts"]
        # conservation already asserted in static_roundtrip
        print(
            f"  {p0:>6.2f}  {c0:>8d}  {c1:>8d}  {c2:>8d}  "
            f"{row['H']:>7.4f}  {row['b_uni']:>9d}  {row['b_bound']:>9d}  "
            f"{row['b_real']:>9d}  {row['bpw_uni']:>8.4f}  {row['bpw_real']:>8.4f}  "
            f"{row['bpw_H']:>7.4f}  {row['sav_pct']:>+7.3f}%"
        )
        # house gate: null savings under p0→1/3 is ~0; at high p0 must be >0
        if p0 >= 0.5:
            assert row["b_real"] < row["b_uni"], (
                f"expected adaptive < uniform at p0={p0}: "
                f"{row['b_real']} vs {row['b_uni']}"
            )
    print(
        f"  null curve: at p0=1/3, bpw_ad ≈ bpw_uni ≈ log2(3)={LOG2_3_F:.4f}; "
        f"adaptivity pays as p0 rises (bpw_ad → H < log2(3))"
    )


# ---------------------------------------------------------------------------
# Selftest + main
# ---------------------------------------------------------------------------

def selftest() -> None:
    print("SELFTEST 10k-trit tensor (static range coder round-trip) ...")
    rng = np.random.default_rng(42)
    # mildly non-uniform so adaptive path is exercised
    trits = rng.choice(3, size=10_000, p=[0.25, 0.40, 0.35]).astype(np.int64)
    counts = count_trits(trits)
    assert sum(counts) == 10_000
    data, nbytes = static_roundtrip(trits)
    H = empirical_H_bits(counts, 10_000)
    bound = adaptive_bound_bytes(10_000, H)
    uni = theory_bytes_uniform_ideal(10_000)
    # bound should upper-bound real within a generous margin; real near nH/8
    nH8 = math.ceil(10_000 * H / 8.0)
    print(f"  n=10000  counts={counts}  H={H:.6f}")
    print(f"  bytes: uni_ideal={uni}  ad_bound={bound}  ad_real={nbytes}  ceil(nH/8)={nH8}")
    assert nbytes > 0
    # real should be within a few dozen bytes of nH/8 for n=10k
    assert abs(nbytes - nH8) <= 16, (
        f"real {nbytes} far from ceil(nH/8)={nH8}"
    )
    # zero-entropy edge: all zeros
    t0 = np.ones(1000, dtype=np.int64)  # symbol 1 only
    d0, n0 = static_roundtrip(t0)
    assert n0 <= 16, f"zero-entropy should be tiny, got {n0}"
    # conservation on a weight vector
    w = trits_to_weights(trits)
    n_m1, n_0, n_p1, n = count_weights(w)
    assert n_m1 + n_0 + n_p1 == n
    assert (n_m1, n_0, n_p1) == counts
    print(f"  zero-entropy n=1000 ad_real={n0} (flush only)")
    print("SELFTEST PASS")
    sys.stdout.flush()


def main() -> None:
    selftest()
    print()
    print(f"LOADING {CKPT_PATH} ...")
    tensors = load_q_tensors(CKPT_PATH)
    print(f"  q__ tensors: {len(tensors)}")
    rows: List[Dict] = []
    for name, arr in tensors:
        rows.append(analyze_tensor(name, arr))
        # progress on large encode
        r = rows[-1]
        print(
            f"  packed {name}: n={r['n']} ad_real={r['b_real']} "
            f"H={r['H']:.4f}",
            flush=True,
        )
    print()
    print_tensor_table(rows)
    totals = print_totals(rows)
    print_sensitivity(totals["p0"])
    print()
    print("ADAPTIVE COMPLETE")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
