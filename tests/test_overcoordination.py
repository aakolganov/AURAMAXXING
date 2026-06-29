"""Tests for configurable / per-atom coordination limits and the coordination-number
distribution (cn_distr) -- e.g. a fraction of Al grown at CN 6, or a fraction of Si as a
3+ / CN-3 site."""
import numpy as np

from base.amorphous_structure import AmorphousStruc_factory
from base.config import CoordinationConfig
from base.initialize import initialize_structure_blank


SIRAL = {"Al": {6: 0.2}}                       # ~20% of Al at CN 6, the rest at the default CN 4
# a minority-variant distribution that ALSO changes the oxidation state: ~30% of Si grown as a
# 3+ / CN-3 species instead of the default 4+ / CN-4 (list form carries the per-variant oxidation).
SI_VARIANT = {"Si": [{"cn": 3, "fraction": 0.3, "oxidation": 3}]}


# --- back-compat: no distribution behaves exactly as before -------------------------

def test_max_cn_array_defaults_without_distribution(make_struct):
    s = make_struct(["Al", "O", "O"], [[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0]])
    assert "max_cn" not in s.atoms.arrays            # no per-atom array created
    assert s.max_cn_array().tolist() == [4, 2, 2]    # pure element defaults
    assert s.min_cn_array().tolist() == [3, 2, 2]


def test_empty_distribution_makes_no_rng_draws(make_struct):
    s = make_struct(["O"], [[0, 0, 0]])
    state_before = s.rng.bit_generator.state
    # No cn_distr entry for any element -> must not touch the rng.
    assert s._assign_cn_variant("Al") == (0, 0)
    assert s._assign_cn_variant("O") == (0, 0)
    assert s.rng.bit_generator.state == state_before


# --- distribution tagging -----------------------------------------------------------

def test_distribution_tags_about_the_target_fraction():
    cfg = CoordinationConfig(cn_distr=SIRAL)
    n = 2000
    s = AmorphousStruc_factory(
        symbols=["Al"] * n,
        positions=np.random.RandomState(0).rand(n, 3) * 18,
        cell=[20.0, 20.0, 20.0], pbc=True, seed=123, config=cfg,
    )
    s.apply_cn_distr()
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
    cfg = CoordinationConfig(max_cn={"Al": 6, "Si": 4, "O": 2, "H": 1}, cn_distr=SIRAL)
    s = initialize_structure_blank(cell=[20.0, 20.0, 20.0], config=cfg)
    assert s.max_cn["Al"] == 6
    # cn_distr is stored in the normalized list form
    assert s.cn_distr == {"Al": [{"cn": 6, "fraction": 0.2}]}
    assert s.cn_distr is not SIRAL                      # config dicts are copied, not aliased


def test_single_cn_value_is_the_only_expected_cn():
    # "Si: 4" -> every Si is assigned CN 4 (fraction 1.0); no rng tail to the default.
    cfg = CoordinationConfig(cn_distr={"Si": 4})
    s = AmorphousStruc_factory(symbols=["Si"] * 50, positions=np.random.RandomState(0).rand(50, 3) * 18,
                               cell=[20.0, 20.0, 20.0], pbc=True, seed=5, config=cfg)
    s.apply_cn_distr()
    assert np.all(s.atoms.arrays["max_cn"] == 4)


# --- maintenance through mutations --------------------------------------------------

def test_commit_tags_new_atom_and_overwrites_padding():
    cfg = CoordinationConfig(cn_distr=SIRAL)
    s = AmorphousStruc_factory(cell=[20.0, 20.0, 20.0], pbc=True, seed=1, config=cfg)
    for _ in range(50):
        s.commit_atom("Al", np.random.rand(3) * 18)
    tags = s.atoms.arrays["max_cn"]
    assert len(tags) == 50
    assert set(np.unique(tags)).issubset({0, 6})       # never a leftover pad artefact


def test_sort_atoms_keeps_max_cn_aligned():
    cfg = CoordinationConfig(cn_distr=SIRAL)
    s = AmorphousStruc_factory(
        symbols=["O", "Al", "O", "Al"],
        positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]],
        cell=[20.0, 20.0, 20.0], pbc=True, seed=2, config=cfg,
    )
    s.apply_cn_distr()
    before = {(sym, int(tag)) for sym, tag in
              zip(s.symbols, s.atoms.arrays["max_cn"], strict=True)}
    s.sort_atoms()
    after = {(sym, int(tag)) for sym, tag in
             zip(s.symbols, s.atoms.arrays["max_cn"], strict=True)}
    assert before == after


def test_remove_atom_reindexes_max_cn(make_struct):
    cfg = CoordinationConfig(cn_distr=SIRAL)
    s = AmorphousStruc_factory(
        symbols=["Al", "O", "Al"], positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0]],
        cell=[20.0, 20.0, 20.0], pbc=True, seed=3, config=cfg,
    )
    s.atoms.set_array("max_cn", np.array([6, 0, 6], dtype=int))
    s.remove_atom(0)                                   # drop first Al
    assert s.atoms.arrays["max_cn"].tolist() == [0, 6]


# --- reproducibility ----------------------------------------------------------------

def test_distribution_growth_reproducible(dummy_calc, tmp_path):
    from growth.new_growth import grow_structure
    from base.limits import make_limit_flat, make_limits_fourier, fix_limits

    def run():
        cfg = CoordinationConfig(cn_distr=SIRAL)
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


# --- fractional oxidation/CN variants -----------------------------------------------

def test_oxidation_variant_is_coupled_to_cn_tag():
    cfg = CoordinationConfig(cn_distr=SI_VARIANT)
    n = 2000
    s = AmorphousStruc_factory(
        symbols=["Si"] * n, positions=np.random.RandomState(0).rand(n, 3) * 18,
        cell=[20.0, 20.0, 20.0], pbc=True, seed=123, config=cfg,
    )
    s.apply_cn_distr()
    mx = s.atoms.arrays["max_cn"]
    ox = s.atoms.arrays["oxidation"]
    assert set(np.unique(mx)).issubset({0, 3})
    assert set(np.unique(ox)).issubset({0, 3})
    # the SAME atoms carry the variant CN and the variant oxidation (one shared draw)
    assert np.array_equal(mx == 3, ox == 3)
    assert 0.25 < (ox == 3).mean() < 0.35


def test_oxidation_free_distribution_creates_no_oxidation_array():
    # SIRAL has no oxidation -> the per-atom oxidation array is never materialised, so plain
    # CN distributions stay byte-for-byte as before.
    cfg = CoordinationConfig(cn_distr=SIRAL)
    s = AmorphousStruc_factory(cell=[20.0, 20.0, 20.0], pbc=True, seed=1, config=cfg)
    for _ in range(20):
        s.commit_atom("Al", np.random.rand(3) * 18)
    assert "max_cn" in s.atoms.arrays
    assert "oxidation" not in s.atoms.arrays


def test_charge_honours_per_atom_oxidation_variants():
    cfg = CoordinationConfig(cn_distr=SI_VARIANT)
    s = AmorphousStruc_factory(
        symbols=["Si", "Si", "O", "O", "O", "O"],
        positions=[[i * 3.0, 0, 0] for i in range(6)],
        cell=[40.0, 40.0, 40.0], pbc=True, seed=1, config=cfg,
    )
    assert s.charge() == 0                       # no variant array yet: 2*4 + 4*(-2)
    s.atoms.set_array("oxidation", np.array([3, 0, 0, 0, 0, 0], dtype=int))
    assert s.charge() == -1                      # first Si as 3+: 3 + 4 - 8


def test_oxidation_variant_growth_reproducible(dummy_calc, tmp_path):
    from growth.new_growth import grow_structure
    from base.limits import make_limit_flat, make_limits_fourier, fix_limits

    distr = {"Si": [{"cn": 4, "fraction": 0.25, "oxidation": 3}]}  # CN unchanged, oxidation variant

    def run():
        cfg = CoordinationConfig(cn_distr=distr)
        s = initialize_structure_blank(cell=[18.0, 18.0, 30.0], config=cfg)
        s.set_seed(7)
        make_limit_flat(s, z_val=10.0, is_for="bottom")
        make_limits_fourier(s, z_av=18.0, alpha=1.0, is_for="top")
        fix_limits(s.limits, hard_limit="bottom")
        grow_structure(s, target_number_atoms=60, target_ratios={"Si": 1, "O": 2},
                       calculator=dummy_calc, output_dir=tmp_path / "g")
        return s.symbols, s.atoms.arrays.get("oxidation")

    sym1, ox1 = run()
    sym2, ox2 = run()
    assert sym1 == sym2
    assert ox1 is not None and np.array_equal(ox1, ox2)


# --- M2: a terminal cap must not over-coordinate a bystander atom ------------------------
# correct_charge force-places caps; place_atom_terminal must pick a spot that bonds ONLY the
# intended anchor, so the cap can't bond (and over-coordinate) an unrelated neighbour.

def test_place_atom_terminal_avoids_bystander_overcoordination(make_struct):
    from helpers.atom_placing import place_atom_terminal

    # Si anchor with a bystander O already bonded nearby. Capping the Si with another O must put
    # the new O where it bonds ONLY to Si -- never to the bystander O (an O-O contact would be a
    # defect that over-coordinates the bystander).
    s = make_struct(["Si", "O"], [[10, 10, 10], [11.5, 10, 10]], cell=(20.0, 20.0, 20.0))
    assert s.get_cn().tolist() == [1, 1]              # Si-O bonded

    place_atom_terminal(s, "O", idx_anchor=0, bond_length=1.63, num_samples=200)

    s.get_graph(force_rebuild=True)
    cn = s.get_cn()
    assert len(s) == 3
    assert cn[0] == 2          # Si gained the cap (now bonded to both O)
    assert cn[2] == 1          # the new cap-O bonds ONLY Si
    assert cn[1] == 1          # the bystander O is NOT over-coordinated by the cap
