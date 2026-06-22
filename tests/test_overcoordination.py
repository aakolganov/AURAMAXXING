"""Tests for configurable / per-atom coordination limits and the Al overcoordination
policy (a fraction of Al allowed to reach CN 6)."""
import numpy as np
import pytest

from base.amorphous_structure import AmorphousStruc_factory
from base.config import CoordinationConfig
from base.initialize import initialize_structure_blank


SIRAL = {"Al": {"max_cn": 6, "fraction": 0.2}}


# --- back-compat: no policy behaves exactly as before -------------------------------

def test_max_cn_array_defaults_without_policy(make_struct):
    s = make_struct(["Al", "O", "O"], [[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0]])
    assert "max_cn" not in s.atoms.arrays            # no per-atom array created
    assert s.max_cn_array().tolist() == [4, 2, 2]    # pure element defaults
    assert s.min_cn_array().tolist() == [3, 2, 2]


def test_empty_policy_makes_no_rng_draws(make_struct):
    s = make_struct(["O"], [[0, 0, 0]])
    state_before = s.rng.bit_generator.state
    # No policy entry for any element -> must not touch the rng.
    assert s._assign_max_cn("Al") == 0
    assert s._assign_max_cn("O") == 0
    assert s.rng.bit_generator.state == state_before


# --- policy tagging -----------------------------------------------------------------

def test_policy_tags_about_the_target_fraction():
    cfg = CoordinationConfig(overcoord_policy=SIRAL)
    n = 2000
    s = AmorphousStruc_factory(
        symbols=["Al"] * n,
        positions=np.random.RandomState(0).rand(n, 3) * 18,
        cell=[20.0, 20.0, 20.0], pbc=True, seed=123, config=cfg,
    )
    s.apply_overcoord_policy()
    tags = s.atoms.arrays["max_cn"]
    assert set(np.unique(tags)).issubset({0, 6})       # only default-sentinel or 6
    frac = (tags == 6).mean()
    assert 0.15 < frac < 0.25                          # ~0.2 within binomial tolerance
    # max_cn_array resolves the 0 sentinel to the element default (4)
    assert set(np.unique(s.max_cn_array())).issubset({4, 6})


def test_tagged_al_at_cn5_not_saturated_and_not_overcoordinated(make_struct):
    from helpers.atom_picker import choose_atom_idx_to_attach_to
    from saturation.new_sat import collect_over_or_under_cn_atoms

    # Al with 5 O neighbours -> CN 5.
    s = make_struct(
        ["Al", "O", "O", "O", "O", "O"],
        [[10, 10, 10], [11.8, 10, 10], [8.2, 10, 10],
         [10, 11.8, 10], [10, 8.2, 10], [10, 10, 11.8]],
    )
    assert s.get_cn(0) == 5

    # Untagged (max 4): over-coordinated, and excluded as an attachment anchor.
    assert 0 in collect_over_or_under_cn_atoms(s, do_under=False).get("Al", [])

    # Tag this Al as allowed up to CN 6.
    s.atoms.set_array("max_cn", np.array([6, 0, 0, 0, 0, 0], dtype=int))
    assert 0 not in collect_over_or_under_cn_atoms(s, do_under=False).get("Al", [])
    # It should still be a candidate anchor for another O (not saturated at CN 5).
    chosen = choose_atom_idx_to_attach_to(s, "O", weight_z=False)
    assert chosen == 0


# --- config plumbing ----------------------------------------------------------------

def test_config_plumbs_through_initialize():
    cfg = CoordinationConfig(max_cn={"Al": 6, "Si": 4, "O": 2, "H": 1},
                             overcoord_policy=SIRAL)
    s = initialize_structure_blank(cell=[20.0, 20.0, 20.0], config=cfg)
    assert s.max_cn["Al"] == 6
    assert s.overcoord_policy == SIRAL
    # config dicts are copied, not aliased
    assert s.overcoord_policy is not SIRAL


# --- maintenance through mutations --------------------------------------------------

def test_commit_tags_new_atom_and_overwrites_padding():
    cfg = CoordinationConfig(overcoord_policy=SIRAL)
    s = AmorphousStruc_factory(cell=[20.0, 20.0, 20.0], pbc=True, seed=1, config=cfg)
    for _ in range(50):
        s.commit_atom("Al", np.random.rand(3) * 18)
    tags = s.atoms.arrays["max_cn"]
    assert len(tags) == 50
    assert set(np.unique(tags)).issubset({0, 6})       # never a leftover pad artefact


def test_sort_atoms_keeps_max_cn_aligned():
    cfg = CoordinationConfig(overcoord_policy=SIRAL)
    s = AmorphousStruc_factory(
        symbols=["O", "Al", "O", "Al"],
        positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]],
        cell=[20.0, 20.0, 20.0], pbc=True, seed=2, config=cfg,
    )
    s.apply_overcoord_policy()
    before = {(sym, int(tag)) for sym, tag in
              zip(s.symbols, s.atoms.arrays["max_cn"])}
    s.sort_atoms()
    after = {(sym, int(tag)) for sym, tag in
             zip(s.symbols, s.atoms.arrays["max_cn"])}
    assert before == after


def test_remove_atom_reindexes_max_cn(make_struct):
    cfg = CoordinationConfig(overcoord_policy=SIRAL)
    s = AmorphousStruc_factory(
        symbols=["Al", "O", "Al"], positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0]],
        cell=[20.0, 20.0, 20.0], pbc=True, seed=3, config=cfg,
    )
    s.atoms.set_array("max_cn", np.array([6, 0, 6], dtype=int))
    s.remove_atom(0)                                   # drop first Al
    assert s.atoms.arrays["max_cn"].tolist() == [0, 6]


# --- reproducibility ----------------------------------------------------------------

def test_policy_growth_reproducible(dummy_calc, tmp_path):
    from growth.new_growth import grow_structure
    from base.limits import make_limit_flat, make_limits_fourier, fix_limits

    def run():
        cfg = CoordinationConfig(overcoord_policy=SIRAL)
        s = initialize_structure_blank(cell=[18.0, 18.0, 30.0], config=cfg)
        s.set_seed(7)
        make_limit_flat(s, z_val=10.0, is_for="bottom")
        make_limits_fourier(s, z_av=18.0, alpha=1.0, is_for="top")
        fix_limits(s.limits, hard_limit="bottom")
        grow_structure(s, target_number_atoms=60, target_ratios={"Al": 2, "O": 3},
                       calculator=dummy_calc, output_dir=tmp_path / "g")
        return s.symbols, s.atoms.get_positions(), s.atoms.arrays.get("max_cn")

    sym1, pos1, tag1 = run()
    sym2, pos2, tag2 = run()
    assert sym1 == sym2
    assert np.allclose(pos1, pos2)
    assert np.array_equal(tag1, tag2)
