"""Tests for the YAML run-config schema and loader (runner.config)."""
import pytest

from runner.config import load_config, RunConfig


def _minimal():
    return {
        "structure": {"cell": [20.0, 20.0, 30.0]},
        "composition": {"target_ratios": {"Si": 1, "O": 2}, "target_number_atoms": 100},
        "limits": {"bottom": {"type": "flat", "z": 10.0},
                   "top": {"type": "fourier", "z_av": 18.0, "alpha": 0.3}, "fix": "bottom"},
        "calculators": {"growth": {"type": "lammps", "dump_path": "d"}},
    }


def test_load_minimal_applies_defaults():
    cfg = load_config(_minimal())
    assert isinstance(cfg, RunConfig)
    # pipeline knobs fall back to the same defaults as the functions
    assert cfg.growth.per_anchor_attempts == 100
    assert cfg.finalize.fmax == 0.1
    assert cfg.saturation.enabled is False
    assert cfg.run.seeds == [0] and cfg.run.roughness is None
    # coordination is derived for the run's element set (SiO2 -> {Si, O, H}); curated CN.
    assert cfg.coordination.max_cn["Si"] == 4
    assert cfg.coordination.oxidation["Si"] == 4 and cfg.coordination.oxidation["O"] == -2
    assert cfg.coordination.cn_distr == {}


def test_coordination_merges_onto_defaults():
    d = _minimal()
    d["coordination"] = {"max_cn": {"Al": 6}, "cn_distr": {"Al": {6: 0.2}}}
    cfg = load_config(d)
    assert cfg.coordination.max_cn["Al"] == 6     # overridden
    assert cfg.coordination.max_cn["Si"] == 4     # default preserved
    # cn_distr is stored normalized to the list form
    assert cfg.coordination.cn_distr == {"Al": [{"cn": 6, "fraction": 0.2}]}


def test_cn_distr_accepts_forms_and_oxidation():
    d = _minimal()
    d["coordination"] = {"cn_distr": {
        "Si": 4,                                              # single CN
        "Al": {4: 0.8, 6: 0.2},                              # {cn: fraction} mapping
        "Fe": [{"cn": 6, "fraction": 0.9}, {"cn": 4, "fraction": 0.1, "oxidation": 2}],  # list + ox
    }}
    cfg = load_config(d)
    assert cfg.coordination.cn_distr["Si"] == [{"cn": 4, "fraction": 1.0}]
    assert cfg.coordination.cn_distr["Al"] == [{"cn": 4, "fraction": 0.8}, {"cn": 6, "fraction": 0.2}]
    assert cfg.coordination.cn_distr["Fe"][1] == {"cn": 4, "fraction": 0.1, "oxidation": 2}


def test_cn_distr_rejects_fractions_over_one():
    d = _minimal()
    d["coordination"] = {"cn_distr": {"Al": {4: 0.8, 6: 0.5}}}   # sums to 1.3
    with pytest.raises(ValueError, match="cn_distr"):
        load_config(d)


def test_malformed_cut_off_key_gives_clean_error():
    d = _minimal()
    d["coordination"] = {"cut_offs": {"Si-O-x": 2.0}}            # not 'Element-Element'
    with pytest.raises(ValueError, match="Element-Element"):
        load_config(d)


def test_composition_rejects_mismatched_total_key():
    d = _minimal()
    d["composition"] = {"ratio": {"Si": 1, "O": 2}, "target_number_atoms": 99}  # wrong total key for 'ratio'
    with pytest.raises(ValueError, match="total_atoms|target_number_atoms"):
        load_config(d)
    d2 = _minimal()
    d2["composition"] = {"counts": {"Si": 1, "O": 2}, "total_atoms": 99}        # counts takes no total
    with pytest.raises(ValueError, match="total"):
        load_config(d2)


def test_cut_offs_parse_to_symmetric_tuples():
    d = _minimal()
    d["coordination"] = {"cut_offs": {"Si-O": 1.95}}
    cfg = load_config(d)
    # the override applies symmetrically over the override pair
    assert cfg.coordination.cut_offs[("Si", "O")] == 1.95
    assert cfg.coordination.cut_offs[("O", "Si")] == 1.95
    # other pairs keep their derived covalent-radii cutoff (run element set is {Si, O, H})
    assert cfg.coordination.cut_offs[("Si", "Si")] > 0
    assert cfg.coordination.cut_offs[("O", "H")] == cfg.coordination.cut_offs[("H", "O")]


# --- composition input forms + charge-neutral adjustment ---------------------------

def _with_composition(block):
    d = _minimal()
    d["composition"] = block
    return load_config(d)


def _net_charge(counts):
    from base.element_data import OXIDE_ELEMENTS
    return sum(n * OXIDE_ELEMENTS[e]["oxidation"] for e, n in counts.items())


def test_composition_formula_and_ratio_forms():
    assert _with_composition({"formula": "SiO2", "total_atoms": 99}).composition.target_counts \
        == {"Si": 33, "O": 66}
    cfg = _with_composition({"ratio": {"Si": 1, "O": 2}, "total_atoms": 99})
    assert cfg.composition.target_counts == {"Si": 33, "O": 66}
    assert cfg.composition.target_number_atoms == 99


def test_composition_counts_form_passthrough_when_neutral():
    cfg = _with_composition({"counts": {"Si": 160, "O": 320}})
    assert cfg.composition.target_counts == {"Si": 160, "O": 320}
    assert cfg.composition.adjusted is False


def test_composition_mix_dict_and_colon_string_agree():
    a = _with_composition({"mix": {"SiO2": 7, "Al2O3": 1}, "total_atoms": 200}).composition.target_counts
    b = _with_composition({"mix": "SiO2:Al2O3=7:1", "total_atoms": 200}).composition.target_counts
    assert a == b
    assert set(a) == {"Si", "Al", "O"}
    assert _net_charge(a) == 0


def test_composition_mix_percent_string():
    c = _with_composition({"mix": "70% SiO2 + 30% Al2O3", "total_atoms": 200}).composition.target_counts
    assert set(c) == {"Si", "Al", "O"}
    assert _net_charge(c) == 0


def test_composition_adjusts_to_charge_neutral():
    cfg = _with_composition({"counts": {"Si": 3, "O": 5}})
    assert cfg.composition.adjusted is True
    assert _net_charge(cfg.composition.target_counts) == 0


def _brute_nearest_neutral(x, ox, K=60):
    """Reference: exhaustively find the least-squares-closest neutral integer composition."""
    from itertools import product as _product
    elems = list(x)
    n0 = [max(0, round(x[e])) for e in elems]
    best = None
    for d in _product(range(-K, K + 1), repeat=len(elems)):
        n = [n0[i] + d[i] for i in range(len(elems))]
        if any(c < 0 for c in n) or sum(n) == 0:
            continue
        if sum(n[i] * ox[elems[i]] for i in range(len(elems))) != 0:
            continue
        cost = sum((n[i] - x[elems[i]]) ** 2 for i in range(len(elems)))
        if best is None or cost < best[0]:
            best = (cost, dict(zip(elems, n, strict=True)))
    return best


def test_nearest_neutral_returns_the_closest_composition():
    # M1: off-stoichiometry Al:O = 2:1 -- the old fixed (4,8,16) ladder stopped at the first
    # neutral box (Al24/O36, L2 cost 512); the true nearest neutral is Al22/O33 (cost 493).
    from runner.config import _nearest_neutral_counts
    x, ox = {"Al": 40.0, "O": 20.0}, {"Al": 3, "O": -2}
    counts, _, adjusted = _nearest_neutral_counts(dict(x), ox, "test")
    assert adjusted is True
    assert sum(n * ox[e] for e, n in counts.items()) == 0            # neutral
    cost = sum((counts[e] - x[e]) ** 2 for e in x)
    assert cost == _brute_nearest_neutral(x, ox)[0]                  # provably closest
    assert counts == {"Al": 22, "O": 33}


def test_nearest_neutral_no_false_single_sign_error_for_distant_composition():
    # M1: a reachable-but-distant composition (both ion signs present) must NOT raise the
    # "single charge sign" error just because the neutral point needs a large adjustment.
    from runner.config import _nearest_neutral_counts
    x, ox = {"Al": 20.0, "O": 80.0}, {"Al": 3, "O": -2}
    counts, _, _ = _nearest_neutral_counts(dict(x), ox, "test")
    assert sum(n * ox[e] for e, n in counts.items()) == 0
    assert sum((counts[e] - x[e]) ** 2 for e in x) == _brute_nearest_neutral(x, ox)[0]


def test_nearest_neutral_well_formed_oxides_unchanged():
    # Regression guard: well-formed oxide ratios still pass through exactly (cost 0, no adjust).
    from runner.config import _nearest_neutral_counts
    for x, ox in (({"Si": 144.0, "O": 288.0}, {"Si": 4, "O": -2}),
                  ({"Al": 80.0, "O": 120.0}, {"Al": 3, "O": -2})):
        counts, _, adjusted = _nearest_neutral_counts(dict(x), ox, "test")
        assert adjusted is False
        assert counts == {e: int(v) for e, v in x.items()}


def test_composition_unknown_element_raises():
    with pytest.raises(ValueError, match="Re"):
        _with_composition({"ratio": {"Re": 1, "O": 2}, "total_atoms": 30})


def test_composition_all_cation_cannot_neutralize():
    with pytest.raises(ValueError, match="neutral"):
        _with_composition({"counts": {"Si": 2, "Al": 2}})


def test_lammps_with_non_silica_composition_rejected_at_load():
    d = _minimal()
    d["composition"] = {"formula": "TiO2", "total_atoms": 96}   # growth calc is lammps
    with pytest.raises(ValueError, match="BKS|lammps"):
        load_config(d)


def test_lammps_saturation_with_h_rejected_at_load():
    # M5: saturation adds H, which BKS cannot parameterize -> reject a lammps saturation calc.
    d = _minimal()
    d["saturation"] = {"enabled": True}        # saturation calc defaults to reuse growth (lammps)
    with pytest.raises(ValueError, match="saturation"):
        load_config(d)
    # an MLIP saturation calculator is fine
    d["calculators"]["saturation"] = {"type": "mace", "mace_model_path": "m.model"}
    assert load_config(d).saturation.enabled is True
    # saturation disabled -> a lammps-only run still loads
    d2 = _minimal()
    d2["saturation"] = {"enabled": False}
    assert load_config(d2).saturation.enabled is False


def test_loaded_substrate_elements_become_spectators(tmp_path):
    # M2: a loaded substrate's foreign element gets distance/cutoff rows (so growth placement
    # doesn't KeyError on it) but is not required to be curated and stays out of the per-element
    # oxidation/CN tables.
    from ase import Atoms
    from ase.io import write
    sub = Atoms("CSiO", positions=[[1, 1, 1], [4, 1, 1], [7, 1, 1]], cell=[20, 20, 20], pbc=True)
    path = tmp_path / "substrate.vasp"
    write(str(path), sub, format="vasp")

    d = _minimal()
    d["structure"] = {"from_file": str(path)}                 # XOR: no 'cell'
    d["calculators"] = {"growth": {"type": "mace", "mace_model_path": "m.model"}}
    cfg = load_config(d)
    assert ("C", "O") in cfg.coordination.cut_offs           # spectator distance present
    assert "C" in cfg.coordination.d_min_max
    assert "C" not in cfg.coordination.oxidation             # not forced to be curated
    assert "C" not in cfg.coordination.max_cn


def test_composition_multiple_forms_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        _with_composition({"formula": "SiO2", "ratio": {"Si": 1, "O": 2}, "total_atoms": 99})


def test_legacy_target_ratios_still_loads():
    cfg = _with_composition({"target_ratios": {"Si": 1, "O": 2}, "target_number_atoms": 99})
    assert cfg.composition.target_counts == {"Si": 33, "O": 66}


def test_unknown_top_level_key_raises():
    d = _minimal()
    d["nonsense"] = 1
    with pytest.raises(ValueError, match="nonsense"):
        load_config(d)


def test_missing_required_section_raises():
    d = _minimal()
    del d["composition"]
    with pytest.raises(ValueError, match="composition"):
        load_config(d)


def test_structure_requires_exactly_one_source():
    both = _minimal()
    both["structure"] = {"cell": [1, 1, 1], "from_file": "x"}
    with pytest.raises(ValueError, match="exactly one"):
        load_config(both)

    neither = _minimal()
    neither["structure"] = {}
    with pytest.raises(ValueError, match="exactly one"):
        load_config(neither)


def test_missing_from_file_raises_at_load():
    d = _minimal()
    d["structure"] = {"from_file": "/no/such/structure_file.vasp"}
    with pytest.raises(FileNotFoundError, match="from_file"):
        load_config(d)


def test_fourier_limit_requires_alpha():
    d = _minimal()
    d["limits"]["top"] = {"type": "fourier", "z_av": 18.0}   # no alpha
    with pytest.raises(ValueError, match="alpha"):
        load_config(d)


# --- saturation fragments (capping modes a/b/c) -----------------------------------------

def _minimal_mace():
    # saturation/relabel needs an MLIP backend (BKS can't relax H/F/Na), so grow with mace.
    d = _minimal()
    d["calculators"] = {"growth": {"type": "mace", "mace_model_path": "medium"}}
    return d


def test_saturation_fragment_defaults():
    cfg = load_config(_minimal_mace())
    assert cfg.saturation.positive_fragment == ("H",)
    assert cfg.saturation.negative_fragment == ("O", "H")
    # no extra cap elements registered for the default OH/H
    assert "F" not in cfg.coordination.oxidation and "Na" not in cfg.coordination.oxidation


def test_anion_cap_registers_fragment_element_terminal():
    d = _minimal_mace()
    d["saturation"] = {"enabled": True, "negative_fragment": "F"}
    cfg = load_config(d)
    assert cfg.saturation.negative_fragment == ("F",)
    # F gets an oxidation and a terminal CN of 1 so a placed cap isn't flagged under-coordinated
    assert cfg.coordination.oxidation["F"] == -1
    assert cfg.coordination.max_cn["F"] == 1 and cfg.coordination.min_cn["F"] == 1


def test_cation_cap_forces_terminal_cn_over_curated():
    d = _minimal_mace()
    d["saturation"] = {"enabled": True, "positive_fragment": "Na"}
    cfg = load_config(d)
    assert cfg.saturation.positive_fragment == ("Na",)
    assert cfg.coordination.oxidation["Na"] == 1
    # Na is a curated network former (CN 4-6); as a cap it must be terminal (1)
    assert cfg.coordination.max_cn["Na"] == 1 and cfg.coordination.min_cn["Na"] == 1


def test_fragment_must_be_one_valent():
    d = _minimal_mace()
    d["saturation"] = {"enabled": True, "negative_fragment": "O"}   # O alone is -2
    with pytest.raises(ValueError, match="1-valent"):
        load_config(d)


def test_non_default_fragment_must_be_single_atom():
    d = _minimal_mace()
    d["saturation"] = {"enabled": True, "negative_fragment": ["F", "F"]}
    with pytest.raises(ValueError, match="single element"):
        load_config(d)


def test_mode_inconsistent_with_fragments_raises():
    d = _minimal_mace()
    d["saturation"] = {"enabled": True, "mode": "anion_cap", "positive_fragment": "Na"}
    with pytest.raises(ValueError, match="anion_cap"):
        load_config(d)


def test_cation_cap_mode_rejects_non_default_negative():
    d = _minimal_mace()
    d["saturation"] = {"enabled": True, "mode": "cation_cap", "negative_fragment": "F"}
    with pytest.raises(ValueError, match="cation_cap"):
        load_config(d)


def test_neutral_mode_allows_both_fragments():
    d = _minimal_mace()
    d["saturation"] = {"enabled": True, "mode": "neutral",
                       "negative_fragment": "F", "positive_fragment": "Na"}
    cfg = load_config(d)
    assert cfg.saturation.negative_fragment == ("F",)
    assert cfg.saturation.positive_fragment == ("Na",)
    assert cfg.coordination.oxidation["F"] == -1 and cfg.coordination.oxidation["Na"] == 1


def test_uncurated_cap_element_needs_overrides():
    d = _minimal_mace()
    d["saturation"] = {"enabled": True, "negative_fragment": "Br"}
    with pytest.raises(ValueError, match="Br"):
        load_config(d)


def test_saturation_with_lammps_rejected_for_any_fragment():
    d = _minimal()   # growth is lammps
    d["saturation"] = {"enabled": True, "negative_fragment": "F"}
    with pytest.raises(ValueError, match="lammps|BKS|saturation"):
        load_config(d)


def test_scalar_fragment_gives_clear_error_not_typeerror():
    d = _minimal_mace()
    d["saturation"] = {"enabled": True, "positive_fragment": 5}
    with pytest.raises(ValueError, match="element symbol"):   # not a raw TypeError
        load_config(d)


def test_cn_distr_oxidation_on_cap_element_rejected_when_saturation_on():
    d = _minimal_mace()
    d["coordination"] = {"cn_distr": {"O": [{"cn": 2, "fraction": 0.5, "oxidation": -1}]}}
    d["saturation"] = {"enabled": True}
    with pytest.raises(ValueError, match="cap"):
        load_config(d)


def test_cn_distr_opposite_sign_oxidation_rejected():
    d = _minimal_mace()
    d["composition"] = {"formula": "Fe2O3", "total_atoms": 40}
    d["coordination"] = {"cn_distr": {"Fe": [{"cn": 6, "fraction": 0.1, "oxidation": -1}]}}
    with pytest.raises(ValueError, match="opposite sign"):
        load_config(d)


def test_cn_distr_same_sign_multivalent_oxidation_ok():
    d = _minimal_mace()
    d["composition"] = {"formula": "Fe2O3", "total_atoms": 40}
    d["coordination"] = {"cn_distr": {"Fe": [{"cn": 6, "fraction": 0.1, "oxidation": 2}]}}
    cfg = load_config(d)                                       # +2 vs base +3: same sign, allowed
    assert cfg.coordination.cn_distr["Fe"][0]["oxidation"] == 2


def test_bond_factor_threaded_to_coordination():
    # D1: a custom distance bond_factor is carried on CoordinationConfig so caps use it too.
    d = _minimal_mace()
    d["coordination"] = {"distance": {"bond_factor": 1.2}}
    cfg = load_config(d)
    assert cfg.coordination.bond_factor == 1.2


def test_disabled_saturation_ignores_misconfigured_fragment():
    # C2: a disabled saturation block must not validate/wire its fragments (consistent with the
    # saturation BKS guard, which is gated on enabled). O alone is -2 (not 1-valent) but ignored.
    d = _minimal_mace()
    d["saturation"] = {"enabled": False, "negative_fragment": "O"}
    cfg = load_config(d)
    assert cfg.saturation.enabled is False
