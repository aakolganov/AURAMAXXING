"""Tests for the config runner (runner.runner) and CLI."""
import numpy as np
import pytest
from ase.io import read

from runner.config import load_config, CalculatorSpec
from runner.runner import resolve_plan, build_calculator, run_from_config, pool_from_config, _combo_seed


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
        # statistics are on by default -> per-structure plots + stats.json next to it
        assert (path.parent / "coordination.png").exists()
        assert (path.parent / "stats.json").exists()
    # pooled report for the >1-structure sweep, at the run output_dir
    assert (tmp_path / "out" / "stats_pooled.json").exists()


def test_statistics_per_structure_off_keeps_pooled(tmp_path, monkeypatch):
    from tests.dummy_interface import DummyInterface
    monkeypatch.setattr("runner.runner.build_calculator",
                        lambda spec: DummyInterface(dump_path=str(tmp_path / "dump")))
    cfg = _blank_cfg(tmp_path, seeds=[0, 1])
    cfg.statistics.per_structure = False    # skip per-structure plots; keep the pooled set
    written = run_from_config(cfg)

    assert len(written) == 2
    for path in written:                     # no per-structure plots/json next to structures
        assert not (path.parent / "coordination.png").exists()
        assert not (path.parent / "stats.json").exists()
    assert (tmp_path / "out" / "stats_pooled.json").exists()    # pooled still produced
    assert (tmp_path / "out" / "coordination.png").exists()


# --- Phase 0: per-combo seeding, debug dumps, manifest -----------------------------

def test_combo_seed_decorrelates_and_reproduces():
    # Same (seed, alpha) -> identical stream; differing in either -> different stream.
    draw = lambda rng: rng.integers(0, 2**31, size=8).tolist()
    assert draw(_combo_seed(0, 1.0)) == draw(_combo_seed(0, 1.0))      # reproducible
    assert draw(_combo_seed(0, 0.01)) != draw(_combo_seed(0, 1.0))     # alpha decorrelates
    assert draw(_combo_seed(0, None)) != draw(_combo_seed(1, None))    # seed decorrelates


def test_debug_dumps_off_by_default_then_isolated_per_slab(tmp_path, monkeypatch):
    from tests.dummy_interface import DummyInterface
    # All calculators share ONE dump_path: the worst case for file collisions.
    shared = tmp_path / "shared_dump"
    monkeypatch.setattr("runner.runner.build_calculator",
                        lambda spec: DummyInterface(dump_path=str(shared)))

    cfg = _blank_cfg(tmp_path, seeds=[0, 1])
    cfg.statistics.enabled = False
    run_from_config(cfg)                          # debug defaults: everything off
    out = tmp_path / "out"
    assert not list(out.rglob("final_opt.xyz"))   # no trajectories
    assert not list(out.rglob("traj.xyz"))
    assert not list(out.rglob("growth/dump_*"))   # no per-atom growth snapshots

    cfg2 = _blank_cfg(tmp_path, seeds=[0, 1], output_dir=str(tmp_path / "out2"))
    cfg2.statistics.enabled = False
    cfg2.debug.write_trajectories = True
    run_from_config(cfg2)
    out2 = tmp_path / "out2"
    trajs = list(out2.rglob("final_opt.xyz"))
    assert len(trajs) == 2                                   # one per slab
    assert all("seed" in t.parent.name for t in trajs)       # in each slab's own dir
    assert not (shared / "final_opt.xyz").exists()           # not the shared dump


def test_pool_from_config_rebuilds_from_disk(tmp_path, monkeypatch):
    # Parallel pattern: generation leaves a metrics.json per slab on disk; a single
    # reduce step (pool_from_config) builds the pooled report from those files. This is
    # exactly how a sharded SLURM run pools after all array tasks finish.
    import json
    from tests.dummy_interface import DummyInterface
    monkeypatch.setattr("runner.runner.build_calculator",
                        lambda spec: DummyInterface(dump_path=str(tmp_path / "dump")))
    cfg = _blank_cfg(tmp_path, seeds=[0, 1])
    run_from_config(cfg)

    pooled = tmp_path / "out" / "stats_pooled.json"
    assert pooled.exists()
    pooled.unlink()                       # simulate "slabs generated, pooled not yet built"
    pool_from_config(cfg)                 # reduce: gather metrics.json off disk
    assert pooled.exists()
    assert json.loads(pooled.read_text())["summary"]["n_structures"] == 2


def test_manifest_records_every_combo(tmp_path, monkeypatch):
    import json
    from tests.dummy_interface import DummyInterface
    monkeypatch.setattr("runner.runner.build_calculator",
                        lambda spec: DummyInterface(dump_path=str(tmp_path / "dump")))
    cfg = _blank_cfg(tmp_path, seeds=[0, 1])
    cfg.statistics.enabled = False
    run_from_config(cfg)

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["total"] == 2 and manifest["succeeded"] == 2 and manifest["failed"] == 0
    assert {c["seed"] for c in manifest["combos"]} == {0, 1}
    assert all(c["status"] == "ok" for c in manifest["combos"])


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
