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


def test_slice_structure_removes_out_of_bounds(make_struct):
    from base.limits import make_limit_flat
    from helpers.atom_placing import slice_structure

    # one atom inside [10,18], one below, one above -> only the inside one survives
    s = make_struct(["Si", "Si", "Si"], [[5, 5, 14], [5, 5, 2], [5, 5, 25]],
                    cell=(20.0, 20.0, 30.0))
    make_limit_flat(s, z_val=10.0, is_for="bottom")
    make_limit_flat(s, z_val=18.0, is_for="top")

    slice_structure(s)
    assert len(s.atoms) == 1
    assert 10.0 <= s.atoms.get_positions()[0, 2] <= 18.0


# --- Tier A #1: a grown structure must satisfy basic physical invariants ----------

def _build_growth_struct(seed, cell=(20.0, 20.0, 25.0)):
    from base.initialize import initialize_structure_blank
    from base.limits import make_limit_flat, fix_limits
    s = initialize_structure_blank(cell=list(cell))
    s.set_seed(seed)
    make_limit_flat(s, z_val=5.0, is_for="bottom")
    make_limit_flat(s, z_val=20.0, is_for="top")
    fix_limits(s.limits, hard_limit="bottom")
    return s


def test_growth_produces_valid_structure(dummy_calc, tmp_path):
    from growth.new_growth import grow_structure

    target, cell = 30, (20.0, 20.0, 25.0)
    s = _build_growth_struct(seed=1, cell=cell)
    grow_structure(s, target_number_atoms=target, target_ratios={"Si": 1, "O": 2},
                   calculator=dummy_calc, output_dir=tmp_path / "g")

    assert len(s) == target                                  # reaches the requested size

    syms = s.symbols                                          # composition tracks 1:2 Si:O
    o_frac = syms.count("O") / len(syms)
    assert 0.5 < o_frac < 0.8

    dm = s.atoms.get_all_distances(mic=True)                  # no steric clashes
    np.fill_diagonal(dm, np.inf)
    assert dm.min() > 1.4

    pos = s.atoms.get_positions()                             # inside the periodic cell
    assert np.all(pos >= 0.0) and np.all(pos < np.array(cell))


# --- H1: finalize must invalidate the cached coordination graph -------------------
# finalize_structure relaxes atoms.positions in place via the calculator's optimizer.
# The cached graph cannot observe an in-place position change, so without an explicit
# invalidation get_cn()/get_graph() keep returning the pre-relaxation coordination --
# which the saturation/charge stages (which run right after finalize) then read.

class _BondFormingCalc:
    """Minimal CalculatorInterface stand-in whose optimize() deterministically pulls atom 1
    to a bonding distance from atom 0 (mimicking a relaxation that forms a bond). Used so the
    test pins the cache-invalidation contract without depending on the soft LJ dummy's force
    magnitudes (which leave near-bond atoms unmoved at the default fmax)."""
    def optimize(self, atoms, **kwargs):
        atoms.positions[1] = atoms.positions[0] + [1.6, 0.0, 0.0]   # 1.6 A < 2.0 A Si-O cutoff
        return atoms


def test_finalize_invalidates_stale_graph(make_struct):
    from growth.new_growth import finalize_structure

    # Si-O at 2.3 A -- just beyond the 2.0 A Si-O bonding cutoff, so initially unbonded.
    s = make_struct(["Si", "O"], [[10, 10, 10], [12.3, 10, 10]], cell=(20.0, 20.0, 20.0))
    assert s.get_cn().tolist() == [0, 0]              # primes the graph cache (no bond)

    finalize_structure(s, calculator=_BondFormingCalc())   # relaxation -> bonded at 1.6 A

    # Read the cached path FIRST (a force_rebuild would itself refresh the cache and mask
    # the bug). Stale graph -> [0, 0]; correctly invalidated -> [1, 1].
    cached = s.get_cn().tolist()
    truth = [s.get_graph(force_rebuild=True).degree(i) for i in range(len(s))]
    assert truth == [1, 1]                            # the relaxed geometry really is bonded
    assert cached == truth


# --- Tier A #2: seeded growth is reproducible ------------------------------------

def test_growth_is_deterministic(dummy_calc, tmp_path):
    from growth.new_growth import grow_structure

    def run(out):
        s = _build_growth_struct(seed=42)
        grow_structure(s, target_number_atoms=24, target_ratios={"Si": 1, "O": 2},
                       calculator=dummy_calc, output_dir=tmp_path / out)
        return s

    a, b = run("a"), run("b")
    assert a.symbols == b.symbols
    assert np.allclose(a.atoms.get_positions(), b.atoms.get_positions())


# --- growth dumps must be pure observers: dumps-on == dumps-off -------------------
# write_structure_to_file used to sort_atoms() on the LIVE structure per dump frame,
# reordering the indices consumed by subsequent rng draws: the same seed grew a
# DIFFERENT structure with dumps on than with dumps off.

def test_growth_dumps_do_not_change_the_run(dummy_calc, tmp_path):
    from growth.new_growth import grow_structure

    results = {}
    for dumps in (False, True):
        s = _build_growth_struct(seed=3)
        grow_structure(s, target_number_atoms=25, target_ratios={"Si": 1, "O": 2},
                       calculator=dummy_calc, output_dir=tmp_path / f"dumps_{dumps}",
                       write_growth_dumps=dumps)
        results[dumps] = (list(s.symbols), s.atoms.get_positions().copy())

    assert results[False][0] == results[True][0], "dumps changed the grown composition/order"
    assert np.allclose(results[False][1], results[True][1]), "dumps changed the grown geometry"


# --- deposition mode: bottom-seeded, front-following growth ------------------------
# growth.mode="deposition" seeds the first atom just above the bottom boundary and
# weights anchor selection toward the lowest unsaturated sites, so the film densifies
# bottom-up while the volume above is still open (the bottom face of mid-out growth
# is left porous 4-5 A deep; deposition is the generator-level fix).

def test_deposition_seeds_at_the_bottom(make_struct):
    from base.initialize import initialize_structure_blank
    from base.limits import make_limit_flat, fix_limits
    from helpers.atom_placing import place_atom_most_z_space

    zs = {}
    for mode in ("default", "deposition"):
        s = initialize_structure_blank(cell=[20.0, 20.0, 25.0])
        s.set_seed(4)
        make_limit_flat(s, z_val=5.0, is_for="bottom")
        make_limit_flat(s, z_val=20.0, is_for="top")
        fix_limits(s.limits, hard_limit="bottom")
        place_atom_most_z_space(s, "Si", mode=mode)
        zs[mode] = s.atoms.get_positions()[0, 2]
    assert zs["deposition"] == pytest.approx(6.0, abs=0.01)     # lower_lim + 1.0
    assert zs["default"] == pytest.approx(12.5, abs=0.01)       # mid-height


def test_deposition_grows_bottom_up(dummy_calc, tmp_path):
    import numpy as np
    from growth.new_growth import grow_structure

    s = _build_growth_struct(seed=5)
    grow_structure(s, target_number_atoms=30, target_ratios={"Si": 1, "O": 2},
                   calculator=dummy_calc, output_dir=tmp_path / "g", mode="deposition")
    assert len(s) == 30
    z = s.atoms.get_positions()[:, 2]
    # commit order == index order (no dumps): the first third must sit lower than the
    # last third -- the front advanced upward.
    assert z[:10].mean() < z[-10:].mean() - 1.0
    # and everything grew from the bottom region: the earliest atoms hug the wall
    assert z[:5].mean() < 8.0
