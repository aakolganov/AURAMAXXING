"""Tests for the bare-surface generalization beyond Si/Al/O/H.

Covers the data-driven charge() and the sign-of-oxidation attachment rule with a non-legacy
element (Ti). Distances/coordination for the test structures come from the covalent-radii
derivation in base.element_data, fed through a CoordinationConfig.
"""
import numpy as np
import pytest

from ase import Atoms
from base.amorphous_structure import AmorphousStruc, AmorphousStruc_factory
from base.config import CoordinationConfig
from base.element_data import build_element_tables
from helpers.atom_picker import choose_atom_idx_to_attach_to


def _config_for(elements):
    t = build_element_tables(elements)
    return CoordinationConfig(max_cn=t["max_cn"], min_cn=t["min_cn"],
                              cut_offs=t["cut_offs"], oxidation=t["oxidation"])


def _struct(symbols, positions, elements, cell=(20.0, 20.0, 20.0), seed=42):
    return AmorphousStruc_factory(symbols=list(symbols),
                                  positions=np.asarray(positions, dtype=float),
                                  cell=list(cell), pbc=True, seed=seed,
                                  config=_config_for(elements))


def test_charge_data_driven_for_non_legacy_element():
    # TiO2 is charge-neutral (Ti +4, 2 O -2)
    s = _struct(["Ti", "O", "O"], [[10, 10, 10], [12, 10, 10], [8, 10, 10]], ["Ti", "O"])
    assert s.charge() == 0
    # a lone Ti carries +4, a lone O carries -2
    assert _struct(["Ti"], [[10, 10, 10]], ["Ti", "O"]).charge() == 4
    assert _struct(["O"], [[10, 10, 10]], ["Ti", "O"]).charge() == -2


def test_charge_unknown_element_raises_naming_it():
    # default (legacy) oxidation table has no Mg -> clear error, not a bare KeyError
    s = AmorphousStruc(atoms=Atoms("MgO", positions=[[0, 0, 0], [2, 0, 0]], cell=[20, 20, 20], pbc=True))
    with pytest.raises(ValueError, match="Mg"):
        s.charge()


def test_attachment_uses_oxidation_sign_for_non_legacy_element():
    # Ti-O within the derived Ti-O bonding cutoff; adding O must attach to the Ti cation,
    # adding Ti must attach to the O anion -- driven purely by oxidation sign.
    s = _struct(["Ti", "O"], [[10, 10, 10], [12.0, 10, 10]], ["Ti", "O"])
    assert s.get_cn(0) == 1 and s.get_cn(1) == 1     # they are bonded
    assert s.symbols[choose_atom_idx_to_attach_to(s, "O", weight_z=False)] == "Ti"
    assert s.symbols[choose_atom_idx_to_attach_to(s, "Ti", weight_z=False)] == "O"
