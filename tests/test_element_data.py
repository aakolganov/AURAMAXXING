"""Tests for the curated element table and covalent-radii distance derivation.

These pin the coherence guarantees the rest of the generalization relies on: every sampled
bond stays inside the bonding cutoff (so a committed bond is always a graph edge) and the
different-element collision floor sits below the bond window (so a bonded neighbour is never
rejected as a clash). Also pins that Si/Al/O/H curated defaults reproduce the legacy tables.
"""
import pytest

from base.element_data import (
    OXIDE_ELEMENTS,
    build_element_tables,
    derive_pair_distances,
    element_record,
)


# a deliberately mixed set: legacy + a high-CN transition metal + a light former
ELEMENTS = ["Si", "Al", "O", "H", "Ti", "B", "Mg"]


def _window(sample_dist, a, b):
    """Bypass RandomSample.__getitem__ (which would draw a sample) and read the frozen
    uniform distribution's [lo, hi) support."""
    frozen = dict.__getitem__(sample_dist[a], b)
    lo, hi = frozen.support()
    return float(lo), float(hi)


def test_si_al_o_h_match_legacy_constants():
    from default_constants import OXIDATION_POS, OXIDATION_NEG, default_max_cn, default_min_cn

    legacy_ox = {**OXIDATION_NEG, **OXIDATION_POS}
    for el, ox in legacy_ox.items():
        rec = element_record(el)
        assert rec["oxidation"] == ox, f"{el} oxidation"
        assert rec["max_cn"] == default_max_cn[el], f"{el} max_cn"
        assert rec["min_cn"] == default_min_cn[el], f"{el} min_cn"


def test_unknown_element_raises_naming_it():
    with pytest.raises(ValueError, match="Xx"):
        element_record("Xx")


def test_coherence_window_inside_cutoff_for_all_pairs():
    sample_dist, d_min_max, pair_cutoffs = derive_pair_distances(ELEMENTS)
    for a in ELEMENTS:
        for b in ELEMENTS:
            lo, hi = _window(sample_dist, a, b)
            cutoff = pair_cutoffs[(a, b)]
            assert lo >= 0, f"{a}-{b} window starts below 0 ({lo})"
            # half-open [lo, hi): a draw is strictly < hi <= cutoff -> always a graph edge
            assert hi <= cutoff, f"{a}-{b} window hi {hi} exceeds cutoff {cutoff}"


def test_collision_floor_below_bond_window_for_all_pairs():
    sample_dist, d_min_max, _ = derive_pair_distances(ELEMENTS)
    for a in ELEMENTS:
        for b in ELEMENTS:
            lo, _hi = _window(sample_dist, a, b)
            dmin = d_min_max[a][b][0]
            # a legitimately bonded neighbour (>= lo) must never be inside the clash radius
            assert dmin < lo, f"{a}-{b} collision floor {dmin} not below window lo {lo}"


def test_dmax_equals_cutoff_and_cutoffs_symmetric():
    sample_dist, d_min_max, pair_cutoffs = derive_pair_distances(ELEMENTS)
    for a in ELEMENTS:
        for b in ELEMENTS:
            assert pair_cutoffs[(a, b)] == pair_cutoffs[(b, a)], f"{a}-{b} cutoff asymmetric"
            assert d_min_max[a][b][1] == pair_cutoffs[(a, b)], f"{a}-{b} dmax != cutoff"


def test_sampled_bond_length_is_inside_the_window():
    sample_dist, _, pair_cutoffs = derive_pair_distances(ELEMENTS)
    lo, hi = _window(sample_dist, "Si", "O")
    for _ in range(200):
        d = sample_dist["Si"]["O"]          # RandomSample.__getitem__ draws a sample
        assert lo <= d < hi
        assert d < pair_cutoffs[("Si", "O")]


def test_distance_knob_validation_enforces_coherence():
    # M4: the coherence invariants are enforced in code, not just asserted in tests.
    with pytest.raises(ValueError, match="cutoff_pad"):
        derive_pair_distances(["Si", "O"], cutoff_pad=-0.1)
    with pytest.raises(ValueError, match="collision_factor"):
        derive_pair_distances(["Si", "O"], collision_factor=1.0, bond_factor=1.0)
    with pytest.raises(ValueError, match="bond_factor"):
        derive_pair_distances(["Si", "O"], bond_factor=0.0)
    # a collision floor that would reach into the (small) H-H bond window is rejected per-pair
    with pytest.raises(ValueError, match="collision floor"):
        derive_pair_distances(["H"], collision_factor=0.95, bond_factor=1.0, bond_dev=0.15)


def test_spectator_elements_get_distances_only():
    # M2: a present-but-not-grown element (e.g. a foreign substrate) gets distance/cutoff rows
    # from covalent radii but is NOT required to be curated and is absent from the per-element
    # tables (the per-atom CN arrays tolerate it).
    t = build_element_tables(["Si", "O"], spectator_elements=["C"])   # C is not a curated oxide former
    assert ("C", "O") in t["cut_offs"] and ("O", "C") in t["cut_offs"]
    assert "C" in t["d_min_max"] and "O" in t["d_min_max"]["C"]
    assert "C" in t["sample_dist"]
    assert "C" not in t["oxidation"] and "C" not in t["max_cn"] and "C" not in t["min_cn"]
    # a spectator already among the grown elements is not duplicated/treated specially
    t2 = build_element_tables(["Si", "O"], spectator_elements=["O"])
    assert set(t2["oxidation"]) == {"Si", "O"}


def test_build_element_tables_shapes_and_overrideable_knobs():
    t = build_element_tables(["Ti", "O"])
    assert set(t) == {"oxidation", "max_cn", "min_cn", "sample_dist", "d_min_max", "cut_offs"}
    assert t["oxidation"] == {"Ti": 4, "O": -2}
    assert t["max_cn"]["Ti"] == OXIDE_ELEMENTS["Ti"]["max_cn"]
    # a larger bond_factor pushes the window (and cutoff) out
    base = build_element_tables(["Ti", "O"], distance_knobs={"bond_factor": 1.0})
    longer = build_element_tables(["Ti", "O"], distance_knobs={"bond_factor": 1.3})
    assert longer["cut_offs"][("Ti", "O")] > base["cut_offs"][("Ti", "O")]
