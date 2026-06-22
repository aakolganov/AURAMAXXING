"""Tests for growth-time helpers and the grow/finalize entry points."""
import numpy as np
import pytest


# --- C5: a calculator is required (was a silent Ellipsis placeholder) --------------

def test_finalize_requires_calculator(blank_struct):
    from growth.new_growth import finalize_structure
    with pytest.raises(ValueError):
        finalize_structure(blank_struct, calculator=None)


def test_grow_requires_calculator(blank_struct, tmp_path):
    from growth.new_growth import grow_structure
    with pytest.raises(ValueError):
        grow_structure(
            blank_struct,
            target_number_atoms=3,
            target_ratios={"Si": 1, "O": 2},
            calculator=None,
            output_dir=tmp_path / "g",
        )


# --- C5: placed atoms stay inside the periodic cell (candidate wrapping) -----------

def test_placement_stays_in_cell(make_struct):
    from helpers.atom_placing import place_atom_sphere

    # Anchor near the far corner: without wrapping, many candidates land outside the
    # cell and one of them could be committed at a position >= L.
    s = make_struct(["Si"], [[19.7, 19.7, 19.7]], cell=(20.0, 20.0, 20.0))
    placed = sum(
        place_atom_sphere(s, "O", 0, bond_length=1.6, num_samples=200) for _ in range(8)
    )
    assert placed >= 1

    pos = s.atoms.get_positions()
    assert np.all(pos >= 0.0)
    assert np.all(pos < 20.0)


# --- slice_structure must never remove fixed atoms (frozen substrate) --------------
# On a growth jam, slice_structure removed everything outside the z-limits; a frozen
# substrate sits below the growth volume, so it got deleted and the FixAtoms
# constraint collapsed -> the substrate then relaxed in finalize.

def test_slice_structure_keeps_fixed_atoms(make_struct):
    from ase.constraints import FixAtoms
    from base.limits import make_limit_flat
    from helpers.atom_placing import slice_structure

    # two "substrate" atoms low (z=2) + one grown atom inside the limits (z=12)
    s = make_struct(["O", "O", "Si"], [[5, 5, 2], [6, 5, 2], [5, 5, 12]],
                    cell=(20.0, 20.0, 20.0))
    s.atoms.set_constraint(FixAtoms(indices=[0, 1]))   # freeze the low substrate atoms

    # growth volume is z in [10, 18]; the frozen atoms (z=2) are below it
    make_limit_flat(s, z_val=10.0, is_for="bottom")
    make_limit_flat(s, z_val=18.0, is_for="top")

    slice_structure(s)

    # the frozen substrate must survive (and the in-limits Si), constraint intact
    assert len(s.atoms) == 3
    fixed = [int(i) for c in s.atoms.constraints for i in c.get_indices()]
    assert sorted(fixed) == [0, 1]
