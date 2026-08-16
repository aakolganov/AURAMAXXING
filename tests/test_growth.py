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

    # enough atoms for several layers: fill-fraction leveling keeps a sub-monolayer
    # film deliberately flat, so the upward-front signature needs a thicker film.
    s = _build_growth_struct(seed=5)
    grow_structure(s, target_number_atoms=80, target_ratios={"Si": 1, "O": 2},
                   calculator=dummy_calc, output_dir=tmp_path / "g", mode="deposition")
    assert len(s) == 80
    z = s.atoms.get_positions()[:, 2]
    # commit order == index order (no dumps): fill-fraction leveling keeps the film
    # deliberately flat, so the signature of an advancing front is a MONOTONIC rise of
    # the mean height across commit-order quartiles, not a large absolute gap.
    q = [z[i * 20:(i + 1) * 20].mean() for i in range(4)]
    assert q[3] > q[0] + 0.5, f"front did not advance: quartile means {q}"
    # and everything grew from the bottom region: the earliest atoms hug the wall
    assert z[:5].mean() < 8.0


# --- bridge-scoring placement: prefer network-completing (bridging) sites ----------
# With growth.bridge_bias > 0, place_atom_sphere weights valid candidates by
# (1+bias)^(useful partners - saturated contacts): an atom is preferentially born
# BRIDGING two under-coordinated opposite-sign atoms instead of dangling on one.
# bias=0 preserves the historical uniform pick exactly.

def test_bridge_bias_prefers_bridging_site(make_struct):
    from helpers.atom_placing import place_atom_sphere
    # two under-coordinated Si 3.2 A apart: part of the 1.6 A sphere around the anchor
    # is within the Si-O cutoff (2.0) of the second Si -- a huge bias must pick it.
    s = make_struct(["Si", "Si"], [[10, 10, 10], [13.2, 10, 10]], cell=(20.0, 20.0, 20.0))
    assert place_atom_sphere(s, "O", 0, bond_length=1.6, num_samples=200, bridge_bias=1e6)
    g = s.get_graph(force_rebuild=True)
    assert g.has_edge(2, 0) and g.has_edge(2, 1), "O must be born bridging both Si"


def test_bridge_bias_avoids_saturated_bystander(make_struct):
    import numpy as np
    from helpers.atom_placing import place_atom_sphere
    # the bystander Si#1 is fully coordinated (4 O); with a huge bias the new O placed
    # on Si#0 must avoid landing inside the saturated Si's cutoff.
    d = 1.62
    pos = [[10, 10, 10], [13.2, 10, 10],
           [13.2 + d, 10, 10], [13.2, 10 + d, 10], [13.2, 10 - d, 10], [13.2, 10, 10 + d]]
    s = make_struct(["Si", "Si", "O", "O", "O", "O"], pos, cell=(24.0, 24.0, 24.0))
    assert int(s.get_cn(1)) == 4
    placed = 0
    for _ in range(5):
        if place_atom_sphere(s, "O", 0, bond_length=1.6, num_samples=200, bridge_bias=1e6):
            new = len(s) - 1
            g = s.get_graph(force_rebuild=True)
            assert not g.has_edge(new, 1), "cap-avoiding weight must keep O off the saturated Si"
            placed += 1
    assert placed >= 1


def test_bridge_bias_zero_is_deterministic_default(make_struct):
    import numpy as np
    from helpers.atom_placing import place_atom_sphere
    from runner.config import GrowthSpec
    assert GrowthSpec().bridge_bias == 0.0
    outs = []
    for _ in range(2):
        s = make_struct(["Si"], [[10, 10, 10]], cell=(20.0, 20.0, 20.0), seed=11)
        place_atom_sphere(s, "O", 0, bond_length=1.6, num_samples=100, bridge_bias=0.0)
        outs.append(s.atoms.get_positions()[-1].copy())
    assert np.allclose(outs[0], outs[1])


def test_deposition_front_levels_fill_fraction_across_columns(dummy_calc, tmp_path):
    import numpy as np
    from base.initialize import initialize_structure_blank
    from base.limits import make_limit_flat, fix_limits
    from growth.new_growth import grow_structure

    # two-level top: columns with x < 10 may grow to z=20 (allowance 15), columns with
    # x >= 10 only to z=12 (allowance 7). Fill-fraction leveling must give the tall half
    # proportionally more atoms, so both halves reach a similar FRACTION of their own
    # ceiling -- the mechanism that makes deposition follow a roughness envelope.
    s = initialize_structure_blank(cell=[20.0, 20.0, 25.0])
    s.set_seed(6)
    make_limit_flat(s, z_val=5.0, is_for="bottom")
    make_limit_flat(s, z_val=20.0, is_for="top")
    fix_limits(s.limits, hard_limit="bottom")
    half = s.limits.nx // 2
    s.limits.upper_lim[half:, :] = 12.0

    grow_structure(s, target_number_atoms=60, target_ratios={"Si": 1, "O": 2},
                   calculator=dummy_calc, output_dir=tmp_path / "g", mode="deposition")
    pos = s.atoms.get_positions()
    tall = pos[pos[:, 0] < 10.0]
    short = pos[pos[:, 0] >= 10.0]
    assert len(tall) > 5 and len(short) > 5
    # fill leveling => the tall half's surface sits physically higher than the short
    # half's (the envelope shapes the film), without seed-tuned fraction tolerances
    assert np.percentile(tall[:, 2], 90) > np.percentile(short[:, 2], 90) + 0.5


def test_bridge_bias_degenerates_gracefully_without_partners(make_struct):
    from helpers.atom_placing import place_atom_sphere
    # a lone anchor: no opposite-sign partner anywhere, so all bridge weights are equal --
    # the weighted pick must not crash (w / w.sum()) and the placement must succeed.
    s = make_struct(["Si"], [[10, 10, 10]], cell=(20.0, 20.0, 20.0))
    assert place_atom_sphere(s, "O", 0, bond_length=1.6, num_samples=100, bridge_bias=20.0)


def test_bridge_weights_zero_born_overcoordinated_mixed_contacts(make_struct):
    import numpy as np
    from helpers.atom_placing import _bridge_weights, build_placement_cache
    d = 1.62
    # anchor Si#0; useful under-CN Si#1 and SATURATED Si#2 (four O) both near the "over"
    # candidate: born CN there = anchor + useful + saturated = 3 > O max 2, so its weight
    # must be zero even though the mixed (useful - saturated) exponent cancels to 0.
    syms = ["Si", "Si", "Si", "O", "O", "O", "O"]
    pos = [[10, 10, 10], [13.2, 10, 10], [11.6, 12.77, 10],
           [11.6 + d, 12.77, 10], [11.6 - d, 12.77, 10],
           [11.6, 12.77 + d, 10], [11.6, 12.77, 10 + d]]
    s = make_struct(syms, pos, cell=(24.0, 24.0, 24.0))
    assert int(s.get_cn(2)) == 4                      # saturated bystander
    cache = build_placement_cache(s)
    over = np.array([[11.6, 11.2, 10.0]])             # bonds Si#0(2.0), Si#1(2.0), Si#2(1.57)
    clean = np.array([[10.0, 8.4, 10.0]])             # bonds only the anchor
    w = _bridge_weights(s, "O", 0, np.vstack([clean, over]), cache[1], bias=20.0)
    assert w[1] == 0.0, "born-over-CN candidate with mixed contacts must be zeroed"
    assert w[0] > 0


def test_growth_dumps_write_meshes_and_stage_frames(dummy_calc, tmp_path):
    import json
    from ase.io import read
    from growth.new_growth import grow_structure
    from saturation.new_sat import saturate_under_coordinated

    s = _build_growth_struct(seed=9)
    grow_structure(s, target_number_atoms=20, target_ratios={"Si": 1, "O": 2},
                   calculator=dummy_calc, output_dir=tmp_path / "growth",
                   write_growth_dumps=True)
    dumps = list((tmp_path / "growth").glob("dump_*.xyz"))
    meshes = list((tmp_path / "growth").glob("mesh_*.obj"))
    manifest = tmp_path / "growth" / "mesh_manifest.json"
    assert dumps and meshes and manifest.exists()
    entries = json.load(open(manifest))
    assert all((tmp_path / "growth" / e["mesh"]).exists() for e in entries)

    n0 = len(s)
    saturate_under_coordinated(s, dump_dir=tmp_path / "sat")
    frames = read(tmp_path / "sat" / "saturation.xyz", index=":")
    # one frame before capping plus one per capped SITE; each site adds 1 (H) or 2 (-OH)
    # atoms, and the last frame is the final structure
    assert len(frames[0]) == n0 and len(frames[-1]) == len(s)
    deltas = [len(frames[k + 1]) - len(frames[k]) for k in range(len(frames) - 1)]
    assert deltas and all(d in (1, 2) for d in deltas)


# --- 2-ring (second-bridge) veto in place_atom_sphere ------------------------------
# Placing an atom that bonds two neighbours which already share a bonded neighbour
# closes a 2-membered ring (edge-sharing polyhedra). The veto drops such candidates
# before the pick; pairs whose atoms are both edge-sharing-capable (per-atom max CN
# >= EDGE_SHARE_MIN_CN) are exempt.

def _bridge_struct(make_struct, seed=42):
    """Si-O-Si corner bridge: both Si bonded to the shared O (legacy cutoff 2.0)."""
    return make_struct(["Si", "O", "Si"],
                       [[10.0, 10.0, 10.0], [11.35, 10.0, 10.76], [12.7, 10.0, 10.0]],
                       seed=seed)


def test_second_bridge_mask_vetoes_ring_closure(make_struct):
    from helpers.atom_placing import _second_bridge_mask, build_placement_cache

    s = _bridge_struct(make_struct)
    cache = build_placement_cache(s)
    # candidate O bridging both Si (midway, below the existing bridge) vs a clean site
    ring_closer = np.array([[11.35, 10.0, 9.2]])
    clean = np.array([[8.4, 10.0, 10.0]])
    keep = _second_bridge_mask(s, "O", 0, np.vstack([ring_closer, clean]), cache[1])
    assert list(keep) == [False, True]


def test_second_bridge_mask_exempts_edge_sharing_cn(make_struct):
    from helpers.atom_placing import _second_bridge_mask, build_placement_cache

    s = _bridge_struct(make_struct)
    s.max_cn["Si"] = 6           # pretend an edge-sharing-capable (octahedral) cation
    cache = build_placement_cache(s)
    ring_closer = np.array([[11.35, 10.0, 9.2]])
    keep = _second_bridge_mask(s, "O", 0, ring_closer, cache[1])
    assert list(keep) == [True]


def test_placement_never_commits_a_second_bridge(make_struct):
    from helpers.atom_placing import place_atom_sphere, RING_VETO_NEAR

    # Weakened same-element exclusions reproduce the broken derived tables that let the
    # O-O "shield" open a ring-closure window (Si-Si down to 2.62, O-O down to 1.72);
    # only the veto then stands between placement and a second bridge. The committed O
    # must always stay outside the veto radius of the non-anchor Si.
    veto_r = RING_VETO_NEAR * 2.0        # legacy Si-O bonding cutoff
    for seed in range(30):
        s = make_struct(["Si", "O", "Si"],
                        [[10.0, 10.0, 10.0], [11.35, 10.0, 10.76], [12.7, 10.0, 10.0]],
                        seed=seed)
        # rebind (don't mutate in place: the factory default d_min_max is a SHALLOW copy,
        # so writing into the inner dicts would leak into every other test's tables)
        s.d_min_max = {**s.d_min_max,
                       "O": {**s.d_min_max["O"], "O": [0.99, 1.72]},
                       "Si": {**s.d_min_max["Si"], "Si": [1.665, 2.62]}}
        if place_atom_sphere(s, "O", 0, bond_length=1.9, num_samples=400):
            d = s.atoms.get_distance(len(s) - 1, 2, mic=True)
            assert d > veto_r, f"seed {seed}: placed O at {d:.2f} A of the bridged Si"


# --- place_atom_terminal ranks steric-exclusion clashes ----------------------------
# A cap with no bystander BOND could still sit inside another atom's d_min_max
# exclusion (e.g. cap O 2.3 A from a network O) -- exactly the marginal geometry the
# next relax collapses into a homonuclear bond. Such candidates lose to clash-free ones.

def test_terminal_cap_avoids_same_element_exclusion_clash(make_struct):
    from helpers.atom_placing import place_atom_terminal

    # num_samples=2 gives exactly two candidates (the +/-y poles). The network O at
    # (10, 13.95, 10) puts the +y candidate 2.30 A away: inside the O..O exclusion
    # (2.4) but outside bond (1.8) and near (1.25 x 1.8 = 2.25) -- so without the clash
    # rank the two candidates tie and the pick is a coin flip; with it, -y always wins.
    for seed in range(10):
        s = make_struct(["Si", "O"], [[10.0, 10.0, 10.0], [10.0, 13.95, 10.0]], seed=seed)
        place_atom_terminal(s, "O", 0, bond_length=1.65, num_samples=2)
        d = s.atoms.get_distance(len(s) - 1, 1, mic=True)
        assert d >= 2.4, f"seed {seed}: cap O at {d:.2f} A of the network O"
