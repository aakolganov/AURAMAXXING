"""Shared helpers for the DFT input writers (VASP, CP2K).

Backend-agnostic: a recursive config merge that lets a user override or extend our default
tag sets from YAML with one mechanism, plus structure helpers that order atoms
lightest-to-heaviest (so a hand-assembled VASP POTCAR lines up with the POSCAR) while
keeping any FixAtoms selective-dynamics constraint correct.
"""
from __future__ import annotations

import copy

import numpy as np
from ase import Atoms


def deep_merge(base: dict, override: dict | None) -> dict:
    """Return ``base`` with ``override`` applied recursively (``base`` is not mutated).

    Rules, so one YAML block can both adjust and extend our defaults:
      - key in both, both dict  -> merge recursively
      - value is ``None``       -> delete the key from the result
      - otherwise               -> override (or add) the value
    """
    result = copy.deepcopy(base)
    if not override:
        return result
    for key, val in override.items():
        if val is None:
            result.pop(key, None)
        elif isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def fixed_indices(atoms: Atoms) -> set[int]:
    """Indices held fixed by any ASE constraint (e.g. FixAtoms from a POSCAR's Selective
    Dynamics). Mirrors ``AmorphousStruc.fixed_indices`` but works on a bare Atoms object."""
    fixed: set[int] = set()
    for constraint in atoms.constraints:
        if hasattr(constraint, "get_indices"):
            fixed.update(int(i) for i in constraint.get_indices())
    return fixed


def sort_by_Z(atoms: Atoms) -> Atoms:
    """Return a copy of ``atoms`` ordered by ascending atomic number (stable within an
    element). ASE's fancy indexing reindexes any FixAtoms constraint, so a frozen substrate
    stays frozen and species form lightest-to-heaviest blocks -- the order a hand-built VASP
    POTCAR must follow."""
    order = np.argsort(atoms.numbers, kind="stable")
    return atoms[order]


def element_order(atoms: Atoms) -> list[str]:
    """Distinct element symbols in ascending-Z order (POSCAR/POTCAR block order)."""
    return list(dict.fromkeys(sort_by_Z(atoms).get_chemical_symbols()))
