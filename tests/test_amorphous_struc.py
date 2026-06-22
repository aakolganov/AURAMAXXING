"""Tests for the AmorphousStruc data structure: mutations, graph, charge, factory."""
import numpy as np
import pytest

from base.amorphous_structure import AmorphousStruc_factory


def test_commit_atom_increases_count_and_updates_graph(make_struct):
    s = make_struct(["Si"], [[10, 10, 10]], cell=(20.0, 20.0, 20.0))
    assert len(s) == 1 and s.get_cn(0) == 0           # builds graph (lone Si)

    s.commit_atom("O", [11.6, 10, 10])                # within Si-O cutoff -> bonded
    assert len(s) == 2
    assert s.get_cn(0) == 1 and s.get_cn(1) == 1


def test_replace_atom_changes_symbol_and_position(make_struct):
    s = make_struct(["Si", "O"], [[10, 10, 10], [11.6, 10, 10]], cell=(20.0, 20.0, 20.0))
    s.replace_atom("Al", [12.0, 10, 10], 0)
    assert len(s) == 2
    assert s.symbols[0] == "Al"
    assert np.allclose(s.atoms.get_positions()[0], [12.0, 10, 10])


def test_remove_atom_preserves_and_remaps_fixatoms(make_struct):
    from ase.constraints import FixAtoms
    s = make_struct(["Si", "O", "O", "H"], [[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]],
                    cell=(20.0, 20.0, 20.0))
    s.atoms.set_constraint(FixAtoms(indices=[1, 3]))
    s.remove_atom(0)
    fixed = sorted(int(i) for c in s.atoms.constraints for i in c.get_indices())
    assert fixed == [0, 2]                             # old 1,3 -> new 0,2


@pytest.mark.parametrize("symbols,expected", [
    (["Si", "O", "O"], 0),                  # SiO2 neutral
    (["Al", "Al", "O", "O", "O"], 0),       # Al2O3 neutral
    (["Si"], 4),
    (["O"], -2),
    (["Si", "O"], 2),
    (["Al", "O", "O"], -1),
])
def test_charge_from_composition(make_struct, symbols, expected):
    pos = [[i * 3.0, 0.0, 0.0] for i in range(len(symbols))]
    s = make_struct(symbols, pos, cell=(40.0, 40.0, 40.0))
    assert s.charge() == expected


def test_factory_requires_cell_for_pbc():
    with pytest.raises(ValueError):
        AmorphousStruc_factory(symbols=["Si"], positions=[[0, 0, 0]], pbc=True)
