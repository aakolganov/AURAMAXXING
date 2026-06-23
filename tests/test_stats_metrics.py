"""Tests for stats.metrics (pure computation, no matplotlib needed)."""
import numpy as np
import pytest
from ase.constraints import FixAtoms

from base.amorphous_structure import AmorphousStruc_factory
from stats.metrics import analyze_structure, merge_metrics, summarize

CELL = 30.0
CENTER = [CELL / 2] * 3

TETRA = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]      # ideal tetrahedron
SQUARE = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]         # square planar


def _center_with(neighbors, center="Si", bond=1.7):
    pos = [CENTER] + [list(np.array(CENTER) + bond * np.array(d) / np.linalg.norm(d))
                      for d in neighbors]
    return AmorphousStruc_factory(symbols=[center] + ["O"] * len(neighbors), positions=pos,
                                  cell=[CELL] * 3, pbc=True, seed=0)


def test_tau4_tetrahedral_is_one():
    m = analyze_structure(_center_with(TETRA))
    assert m["tau4"]["Si"]["tau4"][0] == pytest.approx(1.0, abs=1e-3)
    assert m["tau4"]["Si"]["tau4_prime"][0] == pytest.approx(1.0, abs=1e-3)


def test_tau4_square_planar_is_zero():
    m = analyze_structure(_center_with(SQUARE))
    assert m["tau4"]["Si"]["tau4"][0] == pytest.approx(0.0, abs=1e-3)
    assert m["tau4"]["Si"]["tau4_prime"][0] == pytest.approx(0.0, abs=1e-3)


def test_coordination_excludes_fixed_subjects():
    # two Si: one mobile (4 O neighbours), one fixed and isolated.
    s = _center_with(TETRA)                       # Si(0) + 4 O
    s.atoms.append(__import__("ase").Atom("Si", position=[1, 1, 1]))  # isolated Si at index 5
    s.atoms.set_constraint(FixAtoms(indices=[5]))
    m = analyze_structure(s)
    # only the mobile, 4-coordinate Si is a subject
    assert m["coordination"]["Si"] == [4]
    assert m["n_fixed"] == 1


def test_homo_distance_flags_bonds_below_cutoff():
    # two O atoms 1.4 Å apart -> below the O-O cutoff (1.8) -> a homo-element bond.
    s = AmorphousStruc_factory(symbols=["O", "O"], positions=[[5, 5, 5], [6.4, 5, 5]],
                               cell=[CELL] * 3, pbc=True, seed=0)
    m = analyze_structure(s)
    homo = m["homo_distance"]["O"]
    assert homo["distances"][0] == pytest.approx(1.4, abs=1e-6)
    assert homo["cutoff"] == pytest.approx(1.8)
    assert homo["homo_bond_count"] == 2          # both O atoms are within the cutoff


def test_element_O_nearest_distance():
    m = analyze_structure(_center_with(TETRA, bond=1.7))
    assert m["element_O_distance"]["Si"][0] == pytest.approx(1.7, abs=1e-6)
    assert "O" not in m["element_O_distance"]    # O is the anion, not a subject here


def test_saturation_oh_counting():
    # Si-O-H chain (one OH on Si) + a bare O with an H (anion H, not an OH "group").
    s = AmorphousStruc_factory(
        symbols=["Si", "O", "H"], positions=[[5, 5, 5], [6.6, 5, 5], [7.5, 5, 5]],
        cell=[CELL] * 3, pbc=True, seed=0)
    m = analyze_structure(s, is_saturation=True)
    assert m["saturation"]["oh_distances"][0] == pytest.approx(0.9, abs=1e-6)
    assert m["saturation"]["oh_per_element"]["Si"] == [1]


def test_edge_cases_do_not_crash():
    # single atom of an element (no homo pairs), no 4-coordinate atoms
    s = AmorphousStruc_factory(symbols=["Si", "O"], positions=[[5, 5, 5], [6.6, 5, 5]],
                               cell=[CELL] * 3, pbc=True, seed=0)
    m = analyze_structure(s)
    assert "Si" not in m["homo_distance"]        # only one Si -> skipped
    assert m["tau4"] == {}                        # nothing is 4-coordinate
    assert summarize(m)["n_atoms"] == 2


def test_merge_pools_distributions():
    a = analyze_structure(_center_with(TETRA))
    b = analyze_structure(_center_with(SQUARE))
    merged = merge_metrics([a, b])
    assert merged["n_structures"] == 2
    assert len(merged["tau4"]["Si"]["tau4"]) == 2      # one centre from each
    assert merged["composition"]["O"] == 8
