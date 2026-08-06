#!/usr/bin/env python3
"""Exact container-format round-trip for ternary weight checkpoints.

HARD RULE (log-frame): every verdict is exact integer arithmetic — no floats
anywhere in pack/unpack/size logic. Floats are allowed ONLY in printed
display ratios (e.g. bytes_fmt / bytes_2bit).

Trit encoding
-------------
Weights w in {-1, 0, +1} map to trits t = w + 1 in {0, 1, 2}.
A tensor is a flat sequence of n trits.

Formats
-------
- fmt_2bit:    4 trits / byte, little-endian 2-bit fields
               (byte = t0 | t1<<2 | t2<<4 | t3<<6).
- fmt_5_8:     blocks of 5 trits -> one byte:
               value = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4  (0..242).
               Tail block of r < 5 trits: same base-3 little-endian value
               in one byte.
- fmt_41_65:   blocks of 41 trits -> integer
               x = sum_i t_i * 3^i   (0 <= x < 3^41 <= 2^65)
               packed as 9 bytes little-endian.
               Tail r < 41: x packed into ceil(P/8) bytes where
               P is the container bit width for r trits (see below).
- fmt_306_485: blocks of 306 trits -> integer < 3^306 <= 2^485
               packed as 61 bytes little-endian. Tail same rule as above.
- fmt_stream:  exact streaming ternary-to-binary carry-counting range coder
               (uniform symbols over {0,1,2}); variable length; zero header.
               See ``fmt_stream`` section below for renorm / flush contract.

Container bit width for a tail of r >= 1 trits
----------------------------------------------
Values occupy the range [0, 3^r). The minimal bit width P satisfying
3^r <= 2^P is ceil(log2(3^r)).

For any integer m > 0, m.bit_length() = floor(log2(m)) + 1.
- If m is not a power of two, ceil(log2(m)) = floor(log2(m)) + 1 = m.bit_length().
- If m = 2^k, ceil(log2(m)) = k = m.bit_length() - 1.

Note: using (3**r - 1).bit_length() would be the bit length of the *largest
value*, which coincides with ceil(log2(3^r)) except when 3^r is an exact
power of two (where it would under-count the capacity form 3^r <= 2^P by
zero only if one is careful — the capacity form is the intended one).

Exact statement used here:
    minimal P with 3^r <= 2^P equals (3**r).bit_length()
    because 3^r is never a power of 2 for r >= 1
    (any positive power of an odd integer > 1 is odd and greater than 1).

Therefore P = (3**r).bit_length() and tail_bytes = (P + 7) // 8.

fmt_stream — exact carry-counting range coder
--------------------------------------------
State (all exact integers; floats only in display):
  low, range with RENORM_BITS = 32 (range lives in ``[1, 2**32]``;
  after a renorm step that fired, ``range >= 2**24``).
  cache byte + pending count of deferred 0xFF bytes (classic carry-counting).
  Output is a raw byte stream, MSB-first (first emitted byte is most
  significant); **zero** fixed header.

Encode symbol t in {0,1,2}::

    range3 = range // 3
    bounds = [0, range3, 2*range3, range]   # last interval absorbs rem < 3
    low   += bounds[t]
    range  = bounds[t+1] - bounds[t]
    while range < 2**24:   # TOP
        emit top byte of low via carry-counter (cache + pending 0xFF run)
        low   = (low << 8) & (2**32 - 1)
        range <<= 8

``finish()`` flush contract (exact, minimal for this precision):
  Perform ``STREAM_FLUSH_SHIFTS = 5`` forced renorm shifts after the last
  symbol. That is enough to push the full 32-bit ``low`` through the one-byte
  delayed cache pipeline (4 bytes of state + 1 delayed cache byte). The
  decoder mirrors by loading the first ``STREAM_INIT_BYTES = 5`` stream bytes
  into ``code``. No extra trailer beyond those forced shifts.

Decoder uses only ``(data, n)``: same bounds arithmetic; symbol = interval
containing ``(code - low) mod 2**32``; renorm reads subsequent bytes.

Instrumentation via ``stream_probe`` (phase claims): finalized bits track
``C_i = (3**i).bit_length()`` within a bounded schedule band; final slack
``8*nbytes - C_n`` is a small exact integer.

Container file layout (*.bin)
-----------------------------
Single self-describing file:

    <one JSON line UTF-8>\\n
    <raw concatenated packed blobs>

JSON schema (compact, one line)::

    {
      "format": "<fmt_id>",
      "tensors": [
        {"name": str, "n": int, "format": str, "offset": int, "nbytes": int},
        ...
      ]
    }

``offset`` / ``nbytes`` index into the raw blob section that begins
immediately after the first newline. Blobs appear in tensor order with
no padding between them.

CLI
---
    python3 pack_ladder.py selftest
    python3 pack_ladder.py pack <ckpt.npz> <out_prefix>
"""

from __future__ import annotations

import json
import sys
from decimal import ROUND_FLOOR, Decimal, getcontext
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Format ids
# ---------------------------------------------------------------------------

FMT_2BIT = "fmt_2bit"
FMT_5_8 = "fmt_5_8"
FMT_41_65 = "fmt_41_65"
FMT_306_485 = "fmt_306_485"
FMT_665_1055 = "fmt_665_1055"
FMT_STREAM = "fmt_stream"

# Fixed-size container formats (measured size == theory size).
# fmt_665_1055 is opt-in (Law B sum as flat rung-block); not in default FIXED_FORMATS.
FIXED_FORMATS: Tuple[str, ...] = (FMT_2BIT, FMT_5_8, FMT_41_65, FMT_306_485)
ALL_FORMATS: Tuple[str, ...] = FIXED_FORMATS + (FMT_665_1055, FMT_STREAM)

# Full-block sizes (bytes) for the big-int formats.
# 3^41 needs 65 bits -> 9 bytes; 3^306 needs 485 bits -> 61 bytes;
# 3^665 needs 1055 bits -> 132 bytes (flat container of Law B 2×306+53).
BLOCK_41 = 41
BYTES_41 = 9
BLOCK_306 = 306
BYTES_306 = 61
BLOCK_665 = 665
BYTES_665 = 132


# ---------------------------------------------------------------------------
# Exact integer size theory (no floats)
# ---------------------------------------------------------------------------

def container_bits_for_r_trits(r: int) -> int:
    """Minimal P with 3^r <= 2^P for r >= 1; 0 if r == 0.

    Exact statement: minimal P with 3^r <= 2^P equals (3**r).bit_length()
    because 3^r is never a power of 2 for r >= 1.
    """
    if r <= 0:
        return 0
    # (3**r - 1).bit_length() coincides for all r>=1 since 3^r is never 2^k,
    # but the capacity form 3^r <= 2^P is exactly (3**r).bit_length() here.
    return (3 ** r).bit_length()


def container_bytes_for_r_trits(r: int) -> int:
    """ceil(P/8) for r trits, exact integer arithmetic."""
    if r <= 0:
        return 0
    p = container_bits_for_r_trits(r)
    return (p + 7) // 8


def theory_bytes_2bit(n: int) -> int:
    """ceil(n/4) bytes."""
    return (n + 3) // 4


def theory_bytes_5_8(n: int) -> int:
    """ceil(n/5) bytes (each full or tail block uses exactly one byte)."""
    return (n + 4) // 5


def theory_bytes_41_65(n: int) -> int:
    full, rem = divmod(n, BLOCK_41)
    return full * BYTES_41 + container_bytes_for_r_trits(rem)


def theory_bytes_306_485(n: int) -> int:
    full, rem = divmod(n, BLOCK_306)
    return full * BYTES_306 + container_bytes_for_r_trits(rem)


def theory_bytes_665_1055(n: int) -> int:
    """Flat Law-B-sum blocks (665 trits → 132 B) + flat bigint rem.

    Distinct from chiral ``fmt_665_frame`` (2×306+53 parts → 133 B @ 665).
    """
    full, rem = divmod(n, BLOCK_665)
    return full * BYTES_665 + container_bytes_for_r_trits(rem)


THEORY: Dict[str, Callable[[int], int]] = {
    FMT_2BIT: theory_bytes_2bit,
    FMT_5_8: theory_bytes_5_8,
    FMT_41_65: theory_bytes_41_65,
    FMT_306_485: theory_bytes_306_485,
    FMT_665_1055: theory_bytes_665_1055,
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _as_flat_weights(trits: np.ndarray) -> np.ndarray:
    """Return flat int8 view/copy of weights in {-1, 0, +1}."""
    a = np.asarray(trits).reshape(-1)
    if a.dtype != np.int8:
        a = a.astype(np.int8, copy=False)
    # Exact membership check without float.
    if a.size and not (np.all(a >= -1) and np.all(a <= 1)):
        bad = a[(a < -1) | (a > 1)]
        raise ValueError(f"weights must be in {{-1,0,1}}; found e.g. {int(bad[0])}")
    return a


def _to_trits(weights: np.ndarray) -> np.ndarray:
    """Map w in {-1,0,1} -> t in {0,1,2} as int64 for arithmetic."""
    return weights.astype(np.int64) + 1


def _from_trits(trits: np.ndarray) -> np.ndarray:
    """Map t in {0,1,2} -> w in {-1,0,1} as int8."""
    return (trits.astype(np.int64) - 1).astype(np.int8)


# ---------------------------------------------------------------------------
# fmt_2bit — vectorized
# ---------------------------------------------------------------------------

def pack_2bit(trits: np.ndarray) -> bytes:
    """Pack weights in {-1,0,1} as 4 little-endian 2-bit trits per byte."""
    w = _as_flat_weights(trits)
    n = int(w.size)
    if n == 0:
        return b""
    t = _to_trits(w).astype(np.uint8)  # 0,1,2
    pad = (-n) % 4
    if pad:
        t = np.concatenate([t, np.zeros(pad, dtype=np.uint8)])
    t = t.reshape(-1, 4)
    # little-endian 2-bit fields: bits [1:0]=t0, [3:2]=t1, [5:4]=t2, [7:6]=t3
    b = (
        t[:, 0].astype(np.uint8)
        | (t[:, 1].astype(np.uint8) << 2)
        | (t[:, 2].astype(np.uint8) << 4)
        | (t[:, 3].astype(np.uint8) << 6)
    )
    return b.astype(np.uint8).tobytes()


def unpack_2bit(data: bytes, n: int) -> np.ndarray:
    """Unpack n weights from fmt_2bit bytes."""
    if n < 0:
        raise ValueError("n must be non-negative")
    need = theory_bytes_2bit(n)
    if len(data) < need:
        raise ValueError(f"fmt_2bit: need {need} bytes for n={n}, got {len(data)}")
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    b = np.frombuffer(memoryview(data)[:need], dtype=np.uint8)
    t0 = b & np.uint8(3)
    t1 = (b >> 2) & np.uint8(3)
    t2 = (b >> 4) & np.uint8(3)
    t3 = (b >> 6) & np.uint8(3)
    t = np.stack([t0, t1, t2, t3], axis=1).reshape(-1)[:n]
    return _from_trits(t)


# ---------------------------------------------------------------------------
# fmt_5_8 — vectorized
# ---------------------------------------------------------------------------

_POW3_5 = np.array([1, 3, 9, 27, 81], dtype=np.int64)


def pack_5_8(trits: np.ndarray) -> bytes:
    """Pack weights as base-3 little-endian, 5 trits per byte (tail r<5 too)."""
    w = _as_flat_weights(trits)
    n = int(w.size)
    if n == 0:
        return b""
    t = _to_trits(w)  # int64 0,1,2
    full, rem = divmod(n, 5)
    parts: List[bytes] = []
    if full:
        blocks = t[: full * 5].reshape(full, 5)
        vals = (blocks * _POW3_5).sum(axis=1)
        # 0..242 fits in uint8
        parts.append(vals.astype(np.uint8).tobytes())
    if rem:
        tail = t[full * 5 :]
        powers = _POW3_5[:rem]
        val = int((tail * powers).sum())
        parts.append(bytes((val,)))
    return b"".join(parts)


def unpack_5_8(data: bytes, n: int) -> np.ndarray:
    """Unpack n weights from fmt_5_8 bytes."""
    if n < 0:
        raise ValueError("n must be non-negative")
    need = theory_bytes_5_8(n)
    if len(data) < need:
        raise ValueError(f"fmt_5_8: need {need} bytes for n={n}, got {len(data)}")
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    b = np.frombuffer(memoryview(data)[:need], dtype=np.uint8).astype(np.int64)
    full, rem = divmod(n, 5)
    out = np.empty(n, dtype=np.int64)
    if full:
        vals = b[:full].copy()
        # successive divmod base 3 → little-endian digits per block
        blocks = np.empty((full, 5), dtype=np.int64)
        for j in range(5):
            blocks[:, j] = vals % 3
            vals = vals // 3
        out[: full * 5] = blocks.reshape(-1)
    if rem:
        val = int(b[full])
        for j in range(rem):
            out[full * 5 + j] = val % 3
            val //= 3
    return _from_trits(out)


# ---------------------------------------------------------------------------
# fmt_41_65 / fmt_306_485 — Python big ints, loop per block
# ---------------------------------------------------------------------------

def _pack_bigint_blocks(
    trits: np.ndarray,
    block: int,
    full_bytes: int,
) -> bytes:
    w = _as_flat_weights(trits)
    n = int(w.size)
    if n == 0:
        return b""
    t = _to_trits(w)  # 0,1,2 as int64
    out = bytearray()
    full, rem = divmod(n, block)
    # Precompute powers of 3 for a full block (reuse for speed on large n).
    pow3_full = [3 ** i for i in range(block)]
    base = 0
    for _ in range(full):
        x = 0
        # sum t_i * 3^i
        for i in range(block):
            x += int(t[base + i]) * pow3_full[i]
        out += x.to_bytes(full_bytes, "little")
        base += block
    if rem:
        x = 0
        p = 1
        for i in range(rem):
            x += int(t[base + i]) * p
            p *= 3
        nb = container_bytes_for_r_trits(rem)
        out += x.to_bytes(nb, "little")
    return bytes(out)


def _unpack_bigint_blocks(
    data: bytes,
    n: int,
    block: int,
    full_bytes: int,
    fmt_name: str,
) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    full, rem = divmod(n, block)
    need = full * full_bytes + container_bytes_for_r_trits(rem)
    if len(data) < need:
        raise ValueError(f"{fmt_name}: need {need} bytes for n={n}, got {len(data)}")
    out = np.empty(n, dtype=np.int64)
    pos = 0
    base = 0
    for _ in range(full):
        x = int.from_bytes(data[pos : pos + full_bytes], "little")
        pos += full_bytes
        for i in range(block):
            out[base + i] = x % 3
            x //= 3
        base += block
    if rem:
        nb = container_bytes_for_r_trits(rem)
        x = int.from_bytes(data[pos : pos + nb], "little")
        for i in range(rem):
            out[base + i] = x % 3
            x //= 3
    return _from_trits(out)


def pack_41_65(trits: np.ndarray) -> bytes:
    return _pack_bigint_blocks(trits, BLOCK_41, BYTES_41)


def unpack_41_65(data: bytes, n: int) -> np.ndarray:
    return _unpack_bigint_blocks(data, n, BLOCK_41, BYTES_41, FMT_41_65)


def pack_306_485(trits: np.ndarray) -> bytes:
    return _pack_bigint_blocks(trits, BLOCK_306, BYTES_306)


def unpack_306_485(data: bytes, n: int) -> np.ndarray:
    return _unpack_bigint_blocks(data, n, BLOCK_306, BYTES_306, FMT_306_485)


def pack_665_1055(trits: np.ndarray) -> bytes:
    return _pack_bigint_blocks(trits, BLOCK_665, BYTES_665)


def unpack_665_1055(data: bytes, n: int) -> np.ndarray:
    return _unpack_bigint_blocks(data, n, BLOCK_665, BYTES_665, FMT_665_1055)


# ---------------------------------------------------------------------------
# fmt_stream — exact carry-counting range coder (uniform ternary)
# ---------------------------------------------------------------------------

# RENORM_BITS = 32: low masked to 32 bits after each emitted byte; range is a
# free Python int held in [1, 2**32] and restored to >= TOP by renorm.
STREAM_RENORM_BITS = 32
STREAM_FULL = 1 << STREAM_RENORM_BITS  # 2**32
STREAM_MASK = STREAM_FULL - 1
STREAM_TOP = 1 << 24  # renorm while range < 2**24

# Flush / decoder-init coupling (see module docstring):
#   finish: STREAM_FLUSH_SHIFTS forced renorm shifts (no extra trailer write)
#   decode: preload STREAM_INIT_BYTES into `code` before the first symbol
STREAM_FLUSH_SHIFTS = 5
STREAM_INIT_BYTES = 5

# Decimal precision for exact C_i = floor(i*log2(3))+1 = (3**i).bit_length()
# (3^i never a power of two). prec=80 is ample for n up to > 10**7.
getcontext().prec = 80
_LOG2_3 = Decimal(3).ln() / Decimal(2).ln()


def _C_i(i: int) -> int:
    """Exact C_i = (3**i).bit_length() for i >= 1; 0 for i <= 0.

    Uses floor(i * log2(3)) + 1 via high-precision Decimal (equivalent to
    ``(3**i).bit_length()`` because 3**i is never a power of two).
    """
    if i <= 0:
        return 0
    x = Decimal(i) * _LOG2_3
    fl = int(x.to_integral_value(rounding=ROUND_FLOOR))
    return fl + 1


class _StreamEncoder:
    """Binary carry-counting range encoder for uniform symbols in {0,1,2}."""

    __slots__ = (
        "low",
        "range",
        "cache",
        "ffnum",
        "out",
        "finalized_bytes",
        "pending_max",
        "e_list",
        "instrument",
    )

    def __init__(self, instrument: bool = False) -> None:
        self.low = 0
        self.range = STREAM_FULL
        self.cache = 0  # delayed output byte (may receive a future +1 carry)
        self.ffnum = 0  # pending 0xFF count after cache (not yet finalized)
        self.out = bytearray()
        self.finalized_bytes = 0
        self.pending_max = 0
        self.instrument = instrument
        self.e_list: List[int] = []

    def _shift_low(self) -> None:
        """Emit the top byte of ``low`` through the carry-counting pipeline."""
        # Top 8 bits of the unrestricted low (bits above 31 are the carry).
        top = self.low >> 24
        if top != 0xFF:
            carry = self.low >> 32  # 0 or 1 in normal operation
            # Resolve cache + pending 0xFF run into FINAL output bytes.
            self.out.append((self.cache + carry) & 0xFF)
            self.finalized_bytes += 1
            if self.ffnum:
                cbyte = (0xFF + carry) & 0xFF
                for _ in range(self.ffnum):
                    self.out.append(cbyte)
                    self.finalized_bytes += 1
                self.ffnum = 0
            self.cache = top & 0xFF
        else:
            self.ffnum += 1
            if self.ffnum > self.pending_max:
                self.pending_max = self.ffnum
        self.low = (self.low << 8) & STREAM_MASK
        self.range <<= 8

    def encode(self, t: int) -> None:
        """Encode one trit t in {0,1,2} with exact integer bounds."""
        range3 = self.range // 3
        # bounds = [0, range3, 2*range3, range]; last interval absorbs rem < 3.
        b0 = 0
        b1 = range3
        b2 = range3 + range3  # 2*range3
        b3 = self.range
        if t == 0:
            # low += 0
            self.range = b1 - b0
        elif t == 1:
            self.low += b1
            self.range = b2 - b1
        else:
            self.low += b2
            self.range = b3 - b2
        while self.range < STREAM_TOP:
            self._shift_low()
        if self.instrument:
            # Finalized bits only: bytes already past the pending 0xFF run.
            self.e_list.append(self.finalized_bytes * 8)

    def finish(self) -> bytes:
        """Flush remaining state.

        Exact flush size policy: ``STREAM_FLUSH_SHIFTS`` (= 5) forced renorm
        shifts. Each shift may write 1 + ffnum bytes when the top byte is not
        0xFF. No separate post-loop cache write — the five shifts drain the
        32-bit ``low`` through the one-byte delayed cache (4 state bytes + 1
        pipeline byte). Decoder preloads the same number of bytes.
        """
        for _ in range(STREAM_FLUSH_SHIFTS):
            self._shift_low()
        return bytes(self.out)


class _StreamDecoder:
    """Mirror of ``_StreamEncoder``; needs only the byte stream and n."""

    __slots__ = ("data", "pos", "low", "range", "code")

    def __init__(self, data: bytes) -> None:
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
        range3 = self.range // 3
        b1 = range3
        b2 = range3 + range3
        # Unsigned 32-bit distance of code from low (interval may wrap mod 2^32).
        v = (self.code - self.low) & STREAM_MASK
        if v < b1:
            t = 0
            self.range = b1  # b1 - 0
        elif v < b2:
            t = 1
            self.low = (self.low + b1) & STREAM_MASK
            self.range = b2 - b1
        else:
            t = 2
            self.low = (self.low + b2) & STREAM_MASK
            self.range = self.range - b2
        while self.range < STREAM_TOP:
            self.low = (self.low << 8) & STREAM_MASK
            self.code = ((self.code << 8) | self._read()) & STREAM_MASK
            self.range <<= 8
        return t


def stream_pack(trits: np.ndarray) -> bytes:
    """Pack weights in {-1,0,1} with the exact streaming ternary range coder."""
    w = _as_flat_weights(trits)
    n = int(w.size)
    if n == 0:
        return b""
    t = _to_trits(w)  # 0,1,2 as int64
    enc = _StreamEncoder(instrument=False)
    # Tight loop over Python ints (exact arithmetic).
    for i in range(n):
        enc.encode(int(t[i]))
    return enc.finish()


def stream_unpack(data: bytes, n: int) -> np.ndarray:
    """Unpack n weights from a fmt_stream byte stream (only data + n)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    if not data:
        raise ValueError(f"fmt_stream: empty data for n={n}")
    dec = _StreamDecoder(data)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = dec.decode()
    return _from_trits(out)


def stream_probe(trits: np.ndarray) -> Dict[str, Any]:
    """Instrument one stream encode for phase / schedule claims.

    Returns a dict of exact integers (plus ``schedule_path`` list)::

        final_bits      – 8 * len(encoded bytes); zero header
        slack           – final_bits - C_n,  C_n = (3**n).bit_length()
        schedule_dev_max– max_i |e_i - C_i| over i = 1..n
        schedule_path   – (e_i - C_i) at i multiples of max(1, n//64)
        pending_max     – max pending-carry run length (bytes) observed
        n, nbytes       – symbol count and encoded byte length
    """
    w = _as_flat_weights(trits)
    n = int(w.size)
    if n == 0:
        return {
            "n": 0,
            "nbytes": 0,
            "final_bits": 0,
            "slack": 0,
            "schedule_dev_max": 0,
            "schedule_path": [],
            "pending_max": 0,
        }
    t = _to_trits(w)
    enc = _StreamEncoder(instrument=True)
    for i in range(n):
        enc.encode(int(t[i]))
    data = enc.finish()
    nbytes = len(data)
    final_bits = 8 * nbytes
    C_n = (3 ** n).bit_length()  # exact
    slack = final_bits - C_n

    # Schedule: e_i recorded after each symbol; C_i = (3**i).bit_length().
    step = max(1, n // 64)
    schedule_path: List[int] = []
    schedule_dev_max = 0
    for i in range(1, n + 1):
        e_i = enc.e_list[i - 1]
        C_i = _C_i(i)
        diff = e_i - C_i
        ad = diff if diff >= 0 else -diff
        if ad > schedule_dev_max:
            schedule_dev_max = ad
        if i % step == 0:
            schedule_path.append(diff)

    return {
        "n": n,
        "nbytes": nbytes,
        "final_bits": final_bits,
        "slack": slack,
        "schedule_dev_max": schedule_dev_max,
        "schedule_path": schedule_path,
        "pending_max": enc.pending_max,
    }


# ---------------------------------------------------------------------------
# Dispatch tables
# ---------------------------------------------------------------------------

PACKERS: Dict[str, Callable[[np.ndarray], bytes]] = {
    FMT_2BIT: pack_2bit,
    FMT_5_8: pack_5_8,
    FMT_41_65: pack_41_65,
    FMT_306_485: pack_306_485,
    FMT_665_1055: pack_665_1055,
    FMT_STREAM: stream_pack,
}

UNPACKERS: Dict[str, Callable[[bytes, int], np.ndarray]] = {
    FMT_2BIT: unpack_2bit,
    FMT_5_8: unpack_5_8,
    FMT_41_65: unpack_41_65,
    FMT_306_485: unpack_306_485,
    FMT_665_1055: unpack_665_1055,
    FMT_STREAM: stream_unpack,
}


# ---------------------------------------------------------------------------
# Container I/O (JSON index line + raw blobs)
# ---------------------------------------------------------------------------

def write_container(
    path: str,
    format_id: str,
    items: Sequence[Tuple[str, bytes, int]],
) -> int:
    """Write self-describing container; return total raw blob bytes.

    items: list of (name, packed_bytes, n).
    """
    tensors = []
    offset = 0
    blobs: List[bytes] = []
    for name, blob, n in items:
        tensors.append(
            {
                "name": name,
                "n": int(n),
                "format": format_id,
                "offset": offset,
                "nbytes": len(blob),
            }
        )
        blobs.append(blob)
        offset += len(blob)
    header = {"format": format_id, "tensors": tensors}
    line = json.dumps(header, separators=(",", ":"), ensure_ascii=True)
    with open(path, "wb") as f:
        f.write(line.encode("utf-8"))
        f.write(b"\n")
        for blob in blobs:
            f.write(blob)
    return offset


def read_container(path: str) -> Tuple[str, List[Tuple[str, int, str, bytes]]]:
    """Read container -> (format_id, [(name, n, format, blob), ...])."""
    with open(path, "rb") as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"empty container: {path}")
        raw = f.read()
    header = json.loads(header_line.decode("utf-8"))
    format_id = header["format"]
    out: List[Tuple[str, int, str, bytes]] = []
    for entry in header["tensors"]:
        off = int(entry["offset"])
        nb = int(entry["nbytes"])
        blob = raw[off : off + nb]
        if len(blob) != nb:
            raise ValueError(
                f"short blob for {entry['name']}: want {nb}, got {len(blob)}"
            )
        out.append((entry["name"], int(entry["n"]), entry["format"], blob))
    return format_id, out


# ---------------------------------------------------------------------------
# CLI: selftest
# ---------------------------------------------------------------------------

SELFTEST_SIZES = [1, 4, 5, 40, 41, 42, 305, 306, 307, 1000, 123457]

# Stream round-trip sizes: all of [1..64] plus selected large n (seeded numpy).
STREAM_SELFTEST_SIZES = list(range(1, 65)) + [305, 306, 307, 999, 1000, 123457, 10**6]
STREAM_SLACK_BOUND = 64
STREAM_SCHEDULE_BOUND = 64


def cmd_selftest() -> int:
    rng = np.random.default_rng(0)
    n_cases = 0

    # Round-trip every *fixed* format × every size (measured == theory).
    for n in SELFTEST_SIZES:
        # uniform over {-1, 0, 1}
        w = rng.integers(-1, 2, size=n, dtype=np.int8)
        for fmt in FIXED_FORMATS:
            packed = PACKERS[fmt](w)
            recovered = UNPACKERS[fmt](packed, n)
            if not np.array_equal(recovered, w):
                # Locate first mismatch for a useful failure message.
                bad = np.flatnonzero(recovered != w)
                i = int(bad[0]) if bad.size else -1
                raise AssertionError(
                    f"round-trip fail {fmt} n={n} at i={i}: "
                    f"got {int(recovered[i]) if i >= 0 else '?'} "
                    f"want {int(w[i]) if i >= 0 else '?'}"
                )
            measured = len(packed)
            theory = THEORY[fmt](n)
            if measured != theory:
                raise AssertionError(
                    f"size fail {fmt} n={n}: measured={measured} theory={theory}"
                )
            n_cases += 1

    # Size table (measured == theory already asserted).
    # Columns: n, bytes per format (measured = theory).
    header = (
        f"{'n':>8}  "
        f"{'2bit':>8}  {'th_2bit':>8}  "
        f"{'5_8':>8}  {'th_5_8':>8}  "
        f"{'41_65':>8}  {'th_41':>8}  "
        f"{'306_485':>8}  {'th_306':>8}"
    )
    print(header)
    print("-" * len(header))
    for n in SELFTEST_SIZES:
        row = [f"{n:>8}"]
        for fmt in FIXED_FORMATS:
            b = THEORY[fmt](n)
            row.append(f"{b:>8}")
            row.append(f"{b:>8}")
        print("  ".join(row))

    # Sanity: full-block byte widths match the stated constants.
    assert container_bytes_for_r_trits(BLOCK_41) == BYTES_41, (
        f"3^{BLOCK_41} should pack into {BYTES_41} bytes, "
        f"got {container_bytes_for_r_trits(BLOCK_41)} "
        f"(P={container_bits_for_r_trits(BLOCK_41)})"
    )
    assert container_bytes_for_r_trits(BLOCK_306) == BYTES_306, (
        f"3^{BLOCK_306} should pack into {BYTES_306} bytes, "
        f"got {container_bytes_for_r_trits(BLOCK_306)} "
        f"(P={container_bits_for_r_trits(BLOCK_306)})"
    )
    assert container_bytes_for_r_trits(BLOCK_665) == BYTES_665, (
        f"3^{BLOCK_665} should pack into {BYTES_665} bytes, "
        f"got {container_bytes_for_r_trits(BLOCK_665)} "
        f"(P={container_bits_for_r_trits(BLOCK_665)})"
    )
    # Flat 665 beats chiral 665-frame bytes at one full block (132 vs 133).
    assert theory_bytes_665_1055(BLOCK_665) == BYTES_665
    assert theory_bytes_665_1055(BLOCK_665) < theory_bytes_5_8(BLOCK_665)

    # -------------------------------------------------------------------
    # fmt_stream: round-trip, SLACK, SCHEDULE
    # -------------------------------------------------------------------
    stream_rng = np.random.default_rng(0)
    slack_max_seen = 0
    slack_max_n = 0
    probe_1e6: Dict[str, Any] | None = None

    for n in STREAM_SELFTEST_SIZES:
        w = stream_rng.integers(-1, 2, size=n, dtype=np.int8)
        packed = stream_pack(w)
        recovered = stream_unpack(packed, n)
        if not np.array_equal(recovered, w):
            bad = np.flatnonzero(recovered != w)
            i = int(bad[0]) if bad.size else -1
            raise AssertionError(
                f"round-trip fail {FMT_STREAM} n={n} at i={i}: "
                f"got {int(recovered[i]) if i >= 0 else '?'} "
                f"want {int(w[i]) if i >= 0 else '?'}"
            )
        n_cases += 1

        # SLACK: 0 <= final_bits - C_n <= 64 (C_n = (3**n).bit_length()).
        # Use stream_probe so schedule instrumentation is exercised too.
        # For huge n, probe is the single encode with instrumentation.
        if n == 10**6:
            probe = stream_probe(w)
            # Cross-check pack path matches probe byte length.
            if probe["nbytes"] != len(packed):
                raise AssertionError(
                    f"stream probe/pack size mismatch n={n}: "
                    f"probe={probe['nbytes']} pack={len(packed)}"
                )
            probe_1e6 = probe
        else:
            # Lightweight slack without full schedule scan for medium n:
            # still exact — re-derive from packed bytes.
            final_bits = 8 * len(packed)
            C_n = (3 ** n).bit_length()
            slack = final_bits - C_n
            probe = {
                "slack": slack,
                "final_bits": final_bits,
                "n": n,
                "nbytes": len(packed),
            }

        slack = int(probe["slack"])
        if slack < 0 or slack > STREAM_SLACK_BOUND:
            print(
                f"SLACK FAIL n={n} slack={slack} "
                f"(bound 0..{STREAM_SLACK_BOUND}) "
                f"final_bits={probe['final_bits']} "
                f"C_n={(3 ** n).bit_length()}"
            )
            raise AssertionError(
                f"SLACK property fail n={n}: slack={slack} "
                f"not in [0, {STREAM_SLACK_BOUND}]"
            )
        if slack > slack_max_seen:
            slack_max_seen = slack
            slack_max_n = n
        n_cases += 1  # slack property case

    print(
        f"STREAM SLACK max_seen={slack_max_seen} at n={slack_max_n} "
        f"(bound {STREAM_SLACK_BOUND})"
    )

    # SCHEDULE: schedule_dev_max <= 64 for n = 10**6.
    assert probe_1e6 is not None
    sdev = int(probe_1e6["schedule_dev_max"])
    print(
        f"STREAM SCHEDULE schedule_dev_max={sdev} "
        f"(bound {STREAM_SCHEDULE_BOUND}) pending_max={probe_1e6['pending_max']}"
    )
    if sdev > STREAM_SCHEDULE_BOUND:
        raise AssertionError(
            f"SCHEDULE property fail n=1000000: schedule_dev_max={sdev} "
            f"> {STREAM_SCHEDULE_BOUND}"
        )
    n_cases += 1  # schedule property case

    # One-line stream summary for n=10**6 (bpw is display float only).
    n1 = 10**6
    nbytes = int(probe_1e6["nbytes"])
    bpw = (8 * nbytes) / n1  # display float
    print(
        f"STREAM n={n1} bpw={bpw:.6f} "
        f"slack={int(probe_1e6['slack'])} "
        f"schedule_dev_max={sdev} "
        f"pending_max={int(probe_1e6['pending_max'])}"
    )

    print(f"SELFTEST PASS n_cases={n_cases}")
    return 0


# ---------------------------------------------------------------------------
# CLI: pack
# ---------------------------------------------------------------------------

def cmd_pack(ckpt_path: str, out_prefix: str) -> int:
    z = np.load(ckpt_path)
    # Stable order: sorted names starting with q__
    names = sorted(k for k in z.files if k.startswith("q__"))
    if not names:
        raise SystemExit(f"no arrays starting with 'q__' in {ckpt_path}")

    sources: List[Tuple[str, np.ndarray]] = []
    for name in names:
        arr = z[name]
        flat = _as_flat_weights(arr)
        sources.append((name, flat))

    totals: Dict[str, int] = {}
    for fmt in ALL_FORMATS:
        items: List[Tuple[str, bytes, int]] = []
        for name, flat in sources:
            blob = PACKERS[fmt](flat)
            # Per-tensor round-trip before writing.
            rec = UNPACKERS[fmt](blob, int(flat.size))
            if not np.array_equal(rec, flat):
                raise AssertionError(f"pack round-trip fail {fmt} tensor={name}")
            items.append((name, blob, int(flat.size)))

        path = f"{out_prefix}.{fmt}.bin"
        total = write_container(path, fmt, items)
        totals[fmt] = total

        # File-level unpack assert.
        _, loaded = read_container(path)
        src_by_name = {n: a for n, a in sources}
        for name, n, fmt_id, blob in loaded:
            assert fmt_id == fmt
            rec = UNPACKERS[fmt](blob, n)
            if not np.array_equal(rec, src_by_name[name]):
                raise AssertionError(
                    f"file round-trip fail {fmt} tensor={name} file={path}"
                )

    base = totals[FMT_2BIT]
    # Display ratios may use floats; packing logic above did not.
    print(f"{'format':<14}  {'bytes':>12}  {'vs_2bit':>10}")
    for fmt in ALL_FORMATS:
        b = totals[fmt]
        if base == 0:
            ratio_s = "n/a"
        else:
            ratio_s = f"{b / base:.6f}"  # display float only
        print(f"{fmt:<14}  {b:>12}  {ratio_s:>10}")
    print("ROUNDTRIP EXACT")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print("Usage:")
        print("  python3 pack_ladder.py selftest")
        print("  python3 pack_ladder.py pack <ckpt.npz> <out_prefix>")
        return 0 if args and args[0] in ("-h", "--help") else 2

    cmd = args[0]
    if cmd == "selftest":
        return cmd_selftest()
    if cmd == "pack":
        if len(args) != 3:
            print("usage: python3 pack_ladder.py pack <ckpt.npz> <out_prefix>", file=sys.stderr)
            return 2
        return cmd_pack(args[1], args[2])

    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
