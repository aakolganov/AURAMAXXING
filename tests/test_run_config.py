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
    assert cfg.coordination.overcoord_policy == {}


def test_coordination_merges_onto_defaults():
    d = _minimal()
    d["coordination"] = {"max_cn": {"Al": 6}, "overcoord_policy": {"Al": {"max_cn": 6, "fraction": 0.2}}}
    cfg = load_config(d)
    assert cfg.coordination.max_cn["Al"] == 6     # overridden
    assert cfg.coordination.max_cn["Si"] == 4     # default preserved
    assert cfg.coordination.overcoord_policy == {"Al": {"max_cn": 6, "fraction": 0.2}}


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


def test_composition_unknown_element_raises():
    with pytest.raises(ValueError, match="Re"):
        _with_composition({"ratio": {"Re": 1, "O": 2}, "total_atoms": 30})


def test_composition_all_cation_cannot_neutralize():
    with pytest.raises(ValueError, match="neutral"):
        _with_composition({"counts": {"Si": 2, "Al": 2}})


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


def test_fourier_limit_requires_alpha():
    d = _minimal()
    d["limits"]["top"] = {"type": "fourier", "z_av": 18.0}   # no alpha
    with pytest.raises(ValueError, match="alpha"):
        load_config(d)


def test_bad_calculator_type_raises():
    d = _minimal()
    d["calculators"]["growth"] = {"type": "vasp"}
    with pytest.raises(ValueError, match="type"):
        load_config(d)
