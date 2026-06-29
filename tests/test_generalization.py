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
                              cut_offs=t["cut_offs"], oxidation=t["oxidation"],
                              sample_dist=t["sample_dist"], d_min_max=t["d_min_max"])


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


def test_grow_titania_with_covalent_distances(dummy_calc, tmp_path):
    # End-to-end smoke test: grow a small TiO2 slab through the covalent-radii distance
    # tables (no Si/Al/O legacy data involved) with the soft dummy calculator.
    from base.initialize import initialize_structure_blank
    from base.limits import make_limit_flat, fix_limits
    from growth.new_growth import grow_structure

    s = initialize_structure_blank(cell=[20.0, 20.0, 25.0], config=_config_for(["Ti", "O"]))
    s.set_seed(7)
    make_limit_flat(s, z_val=5.0, is_for="bottom")
    make_limit_flat(s, z_val=20.0, is_for="top")
    fix_limits(s.limits, hard_limit="bottom")

    grow_structure(s, target_number_atoms=18, target_ratios={"Ti": 1, "O": 2},
                   calculator=dummy_calc, output_dir=tmp_path / "g")

    assert len(s) == 18
    o_frac = s.symbols.count("O") / len(s)
    assert 0.5 < o_frac < 0.8                                   # tracks the 1:2 Ti:O target
    dm = s.atoms.get_all_distances(mic=True)
    np.fill_diagonal(dm, np.inf)
    assert dm.min() > 1.0                                       # no hard steric clashes
    pos = s.atoms.get_positions()
    assert np.all(pos >= 0.0) and np.all(pos < np.array([20.0, 20.0, 25.0]))


def test_non_orthogonal_cell_rejected():
    # growth's grid assumes an orthogonal box; a meaningfully tilted cell must error at init,
    # not silently produce a wrong grid / out-of-box placements.
    from base.initialize import initialize_structure_blank
    with pytest.raises(ValueError, match="non-orthogonal"):
        initialize_structure_blank(cell=[[20, 0, 0], [2, 20, 0], [0, 0, 30]])


def test_negligible_cell_tilt_accepted():
    # a relaxed "orthogonal" slab can carry tiny numerical off-diagonal entries; the relative
    # tolerance must not reject those.
    from base.initialize import initialize_structure_blank
    s = initialize_structure_blank(cell=[[20, 1e-4, 0], [0, 20, 0], [0, 0, 30]])
    assert s is not None
