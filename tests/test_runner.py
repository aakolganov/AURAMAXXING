"""Tests for the config runner (runner.runner) and CLI."""
import numpy as np
import pytest
from ase.io import read

from runner.config import load_config, CalculatorSpec
from runner.runner import resolve_plan, build_calculator, run_from_config


def _blank_cfg(tmp_path, **run):
    return load_config({
        "structure": {"cell": [16.0, 16.0, 30.0]},
        "composition": {"target_ratios": {"Si": 1, "O": 2}, "target_number_atoms": 45},
        "limits": {"bottom": {"type": "flat", "z": 10.0},
                   "top": {"type": "fourier", "z_av": 18.0, "alpha": 1.0}, "fix": "bottom"},
        "calculators": {"growth": {"type": "lammps"}},
        "run": {"output_dir": str(tmp_path / "out"), **run},
    })


# --- sweep expansion ---------------------------------------------------------------

def test_resolve_plan_sweeps_seeds_and_roughness(tmp_path):
    cfg = _blank_cfg(tmp_path, seeds=[0, 1], roughness=[0.01, 1.0])
    plan = resolve_plan(cfg)
    assert len(plan) == 4
    assert {e["seed"] for e in plan} == {0, 1}
    assert {e["alpha"] for e in plan} == {0.01, 1.0}
    # multiple combos -> per-combo subdirectories
    assert all("seed" in p["output_path"].parent.name for p in plan)


def test_resolve_plan_single_combo_writes_flat(tmp_path):
    cfg = _blank_cfg(tmp_path, seeds=[0])
    plan = resolve_plan(cfg)
    assert len(plan) == 1
    assert plan[0]["output_path"].name == "structure.vasp"
    assert plan[0]["output_path"].parent.name == "out"   # no seed/alpha subdir


def test_roughness_without_fourier_top_raises(tmp_path):
    cfg = load_config({
        "structure": {"cell": [16.0, 16.0, 30.0]},
        "composition": {"target_ratios": {"Si": 1, "O": 2}, "target_number_atoms": 45},
        "limits": {"bottom": {"type": "flat", "z": 10.0},
                   "top": {"type": "flat", "z": 18.0}, "fix": "bottom"},
        "calculators": {"growth": {"type": "lammps"}},
        "run": {"output_dir": str(tmp_path), "roughness": [0.1, 1.0]},
    })
    with pytest.raises(ValueError, match="fourier"):
        resolve_plan(cfg)


# --- calculator factory (no backend instantiation in the fast tier) ----------------

def test_build_calculator_requires_model_path():
    with pytest.raises(ValueError, match="mace_model_path"):
        build_calculator(CalculatorSpec(type="mace", options={}))
    with pytest.raises(ValueError, match="uma_model_path"):
        build_calculator(CalculatorSpec(type="uma", options={}))


# --- end-to-end with the LJ dummy backend ------------------------------------------

def test_run_from_config_end_to_end(tmp_path, monkeypatch):
    from tests.dummy_interface import DummyInterface
    # Replace the calculator factory so no LAMMPS/MACE is needed.
    monkeypatch.setattr("runner.runner.build_calculator",
                        lambda spec: DummyInterface(dump_path=str(tmp_path / "dump")))

    cfg = _blank_cfg(tmp_path, seeds=[0, 1])      # 2-structure sweep (no roughness)
    written = run_from_config(cfg)

    assert len(written) == 2
    for path in written:
        assert path.exists()
        atoms = read(str(path))
        assert len(atoms) == 45                    # grew to the requested size
        pos = atoms.get_positions()
        assert np.all(np.isfinite(pos))


# --- CLI ---------------------------------------------------------------------------

def test_cli_dry_run(tmp_path, capsys):
    from runner.__main__ import main
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "structure: {cell: [16.0, 16.0, 30.0]}\n"
        "composition: {target_ratios: {Si: 1, O: 2}, target_number_atoms: 45}\n"
        "limits: {bottom: {type: flat, z: 10.0}, top: {type: fourier, z_av: 18.0, alpha: 0.3}, fix: bottom}\n"
        "calculators: {growth: {type: lammps}}\n"
        f"run: {{seeds: [0, 1], output_dir: {tmp_path / 'out'}}}\n"
    )
    rc = main(["--dry-run", str(cfg_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Resolved 2 structure(s)" in out


def test_cli_bad_config_returns_error(tmp_path, capsys):
    from runner.__main__ import main
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("structure: {cell: [1,1,1]}\n")   # missing required sections
    rc = main(["--dry-run", str(cfg_path)])
    assert rc == 2
