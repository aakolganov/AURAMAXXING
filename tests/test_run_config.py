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
    # coordination defaults to the constants
    assert cfg.coordination.max_cn["Al"] == 4
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
    assert cfg.coordination.cut_offs[("Si", "O")] == 1.95
    assert cfg.coordination.cut_offs[("O", "Si")] == 1.95
    assert cfg.coordination.cut_offs[("Al", "O")] == 2.1   # untouched default


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
