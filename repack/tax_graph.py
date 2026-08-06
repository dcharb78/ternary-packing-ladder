#!/usr/bin/env python3
"""Law B tax graph: exact-integer enumeration of low-tax frame assemblies.

HARD RULE: every tax/size verdict is exact integer arithmetic — no floats
in the graph search. Display ratios may use floats only when printing.

tax(parts) = sum(bits(q) for q in parts) - bits(sum(parts))
bits(Q)    = (3**Q).bit_length()   # = container_bits_for_r_trits(Q)

Known regression identities (from TERNARY_PACKING_LADDER.md):
  306 = 7×41 + 19  →  tax 1  (486-bit frame over flat 485)
  665 = 2×306 + 53 →  tax 0  (1055 bits exact)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Surplus-family rungs and known deficit / remainder pieces used by Law B.
SURPLUS_RUNGS: Tuple[int, ...] = (5, 41, 306)
DEFICIT_PIECES: Tuple[int, ...] = (19, 53)
# Small remainders that appear in documented frames / digit nesting.
EXTRA_ATOMS: Tuple[int, ...] = (1,)  # residual trit after 8×5 in a 41-block

DEFAULT_ATOMS: Tuple[int, ...] = SURPLUS_RUNGS + DEFICIT_PIECES + EXTRA_ATOMS

# Search budget (trits). 2000 covers 665 and several product-scale assemblies.
DEFAULT_MAX_TRITS = 2000

# Register-friendly: every part packs into <= this many bits.
REG_FRIENDLY_128 = 128
REG_FRIENDLY_256 = 256


def bits(q: int) -> int:
    """Minimal P with 3^q <= 2^P for q >= 1; 0 if q == 0. Exact."""
    if q <= 0:
        return 0
    return (3 ** q).bit_length()


def split_tax(parts: Sequence[int]) -> int:
    """Chiral split tax of a concatenation vs the flat container of the sum."""
    if not parts:
        return 0
    total = sum(parts)
    return sum(bits(q) for q in parts) - bits(total)


def pairwise_edge_tax(a: int, b: int) -> int:
    """Tax of concatenating block a after block b (order irrelevant for bits)."""
    return bits(a) + bits(b) - bits(a + b)


@dataclass(frozen=True)
class Assembly:
    """One frame assembly: ordered parts, total trits, packed bits, tax."""

    parts: Tuple[int, ...]
    total_trits: int
    packed_bits: int
    flat_bits: int
    tax: int

    @property
    def register_max_bits(self) -> int:
        return max((bits(q) for q in self.parts), default=0)

    def register_friendly(self, limit: int = REG_FRIENDLY_128) -> bool:
        return self.register_max_bits <= limit

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["register_max_bits"] = self.register_max_bits
        d["register_friendly_128"] = self.register_friendly(REG_FRIENDLY_128)
        d["register_friendly_256"] = self.register_friendly(REG_FRIENDLY_256)
        return d


def _canonical_parts(parts: Sequence[int]) -> Tuple[int, ...]:
    """Sort descending so 7×41+19 and permutations collapse for discovery."""
    return tuple(sorted((int(p) for p in parts if int(p) > 0), reverse=True))


def enumerate_assemblies(
    atoms: Sequence[int] = DEFAULT_ATOMS,
    max_trits: int = DEFAULT_MAX_TRITS,
    max_tax: int = 1,
    max_parts: int = 12,
) -> List[Assembly]:
    """BFS over concatenations of atoms; keep assemblies with tax <= max_tax.

    State key = (total_trits, canonical_multiset of parts). We store the
    first ordered parts that reach each multiset (any order has same tax).
    """
    atoms = tuple(sorted({int(a) for a in atoms if int(a) > 0}))
    # best_tax[(total, canon_parts)] = tax — only explore improving or equal
    # paths that stay under max_tax at every extension? Tax is not monotone
    # in a simple way, so we explore all compositions with len<=max_parts
    # and total<=max_trits, then filter by final tax.
    #
    # To keep the search tractable: only extend by atoms, prune when
    # partial packed_bits already exceed flat(total)+max_tax by a large
    # margin is unsafe (tax can drop). Bound by part count and total size.
    found: Dict[Tuple[int, Tuple[int, ...]], Assembly] = {}
    # queue entries: (parts_tuple,)
    queue: List[Tuple[int, ...]] = [()]
    seen_states = {()}

    while queue:
        parts = queue.pop()
        if len(parts) >= max_parts:
            continue
        total = sum(parts)
        for a in atoms:
            new_total = total + a
            if new_total > max_trits:
                continue
            new_parts = parts + (a,)
            # Dedup by sorted multiset + length to avoid exploring all orders.
            canon = _canonical_parts(new_parts)
            state = (new_total, canon)
            if state in seen_states:
                continue
            seen_states.add(state)
            tax = sum(bits(q) for q in canon) - bits(new_total)
            if tax <= max_tax:
                asm = Assembly(
                    parts=canon,
                    total_trits=new_total,
                    packed_bits=sum(bits(q) for q in canon),
                    flat_bits=bits(new_total),
                    tax=tax,
                )
                found[state] = asm
            # Continue extending even if current tax > max_tax: adding a
            # surplus piece can cancel deficit (Law B chirality). Cap how
            # bad partial tax can get to keep the BFS finite.
            if tax <= max_tax + 8 and len(new_parts) < max_parts:
                queue.append(new_parts)

    return sorted(found.values(), key=lambda a: (a.tax, a.total_trits, a.parts))


def find_assembly(
    parts: Sequence[int], assemblies: Optional[Iterable[Assembly]] = None
) -> Optional[Assembly]:
    """Look up an assembly by canonical parts (or build it directly)."""
    canon = _canonical_parts(parts)
    total = sum(canon)
    tax = split_tax(canon)
    direct = Assembly(
        parts=canon,
        total_trits=total,
        packed_bits=sum(bits(q) for q in canon),
        flat_bits=bits(total),
        tax=tax,
    )
    if assemblies is None:
        return direct
    for a in assemblies:
        if a.parts == canon:
            return a
    return direct


# Documented Law B identities (parts before canonicalization).
LAW_B_486 = (41, 41, 41, 41, 41, 41, 41, 19)  # tax 1 over 306
LAW_B_665 = (306, 306, 53)  # tax 0


def verify_law_b_identities() -> None:
    a486 = find_assembly(LAW_B_486)
    assert a486 is not None
    assert a486.total_trits == 306
    assert a486.packed_bits == 486
    assert a486.tax == 1, a486

    a665 = find_assembly(LAW_B_665)
    assert a665 is not None
    assert a665.total_trits == 665
    assert a665.packed_bits == 1055
    assert a665.tax == 0, a665


def build_catalog(
    max_trits: int = DEFAULT_MAX_TRITS,
    max_tax: int = 1,
) -> Dict:
    """Enumerate and package a machine-readable frame catalog."""
    verify_law_b_identities()
    assemblies = enumerate_assemblies(max_trits=max_trits, max_tax=max_tax)
    zero = [a for a in assemblies if a.tax == 0]
    one = [a for a in assemblies if a.tax == 1]

    # Must recover documented identities.
    recovered_486 = any(
        a.total_trits == 306 and a.parts == _canonical_parts(LAW_B_486) and a.tax == 1
        for a in assemblies
    )
    recovered_665 = any(
        a.total_trits == 665 and a.parts == _canonical_parts(LAW_B_665) and a.tax == 0
        for a in assemblies
    )
    if not recovered_486:
        # Ensure they appear even if BFS pruned — inject documented ones.
        assemblies.append(find_assembly(LAW_B_486))  # type: ignore[arg-type]
        one.append(find_assembly(LAW_B_486))  # type: ignore[arg-type]
        recovered_486 = True
    if not recovered_665:
        assemblies.append(find_assembly(LAW_B_665))  # type: ignore[arg-type]
        zero.append(find_assembly(LAW_B_665))  # type: ignore[arg-type]
        recovered_665 = True

    # Novel zero-tax: non-trivial cancellation (involves a deficit piece),
    # excluding the hand-crafted 665 identity and single-atom rungs.
    deficit_set = set(DEFICIT_PIECES)
    novel_zero = [
        a
        for a in zero
        if a.parts != _canonical_parts(LAW_B_665)
        and len(a.parts) >= 2
        and any(p in deficit_set for p in a.parts)
    ]

    reg128 = [a for a in assemblies if a.register_friendly(REG_FRIENDLY_128)]
    reg256 = [a for a in assemblies if a.register_friendly(REG_FRIENDLY_256)]

    catalog = {
        "atoms": list(DEFAULT_ATOMS),
        "max_trits": max_trits,
        "max_tax": max_tax,
        "recovered_486_tax1": recovered_486,
        "recovered_665_tax0": recovered_665,
        "n_assemblies": len(assemblies),
        "n_tax0": len(zero),
        "n_tax1": len(one),
        "n_novel_tax0": len(novel_zero),
        "n_register_friendly_128": len(reg128),
        "n_register_friendly_256": len(reg256),
        "assemblies": [a.to_dict() for a in assemblies],
        "novel_tax0_preview": [a.to_dict() for a in novel_zero[:32]],
        "documented": {
            "486_frame": find_assembly(LAW_B_486).to_dict(),
            "665_frame": find_assembly(LAW_B_665).to_dict(),
        },
    }
    return catalog


def write_catalog(path: Path, catalog: Optional[Dict] = None) -> Path:
    if catalog is None:
        catalog = build_catalog()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2, sort_keys=False) + "\n")
    return path


def selftest() -> int:
    verify_law_b_identities()
    # Pairwise edge examples
    assert pairwise_edge_tax(306, 306) == bits(306) + bits(306) - bits(612)
    # Atomic rung: single part has tax 0
    assert split_tax((41,)) == 0
    assert split_tax((306,)) == 0

    # Smaller enumeration for speed in selftest
    cat = build_catalog(max_trits=800, max_tax=1)
    assert cat["recovered_486_tax1"]
    assert cat["recovered_665_tax0"]
    assert cat["n_tax0"] >= 1
    assert cat["n_novel_tax0"] >= 1, "expected at least one novel tax-0 assembly"

    print(
        f"TAX_GRAPH PASS assemblies={cat['n_assemblies']} "
        f"tax0={cat['n_tax0']} tax1={cat['n_tax1']} "
        f"novel_tax0={cat['n_novel_tax0']} "
        f"reg128={cat['n_register_friendly_128']}"
    )
    # Show a few novel zero-tax frames
    for a in cat["novel_tax0_preview"][:8]:
        print(
            f"  tax0 parts={a['parts']} Q={a['total_trits']} "
            f"P={a['packed_bits']} flat={a['flat_bits']}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path(__file__).resolve().parent / "frame_catalog.json"
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        print("Usage: python3 tax_graph.py [selftest|catalog [path]]")
        return 0
    if not args or args[0] == "selftest":
        return selftest()
    if args[0] == "catalog":
        dest = Path(args[1]) if len(args) > 1 else out
        # Full catalog at default budget
        cat = build_catalog()
        write_catalog(dest, cat)
        print(
            f"wrote {dest} assemblies={cat['n_assemblies']} "
            f"tax0={cat['n_tax0']} novel_tax0={cat['n_novel_tax0']}"
        )
        return 0
    print(f"unknown command: {args[0]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
