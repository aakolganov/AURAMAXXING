"""Tests for the config runner (runner.runner) and CLI."""
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from runner.config import load_config, CalculatorSpec
from runner.runner import (resolve_plan, build_calculator, run_from_config, pool_from_config,
                           _combo_seed, _shard_plan)


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


# --- substrate-loaded growth (load a fixed surface, grow + saturate on top) -----------

def test_substrate_loaded_growth_keeps_substrate_frozen(tmp_path, monkeypatch):
    # Load a small fixed Al/O substrate (Selective Dynamics -> FixAtoms), anchor the growth
    # region just above it, grow + saturate on top with the LJ dummy, and assert the substrate
    # atoms never move (frozen) while new atoms are added above. Exercises the same loaded-surface
    # path as the production gamma-Al2O3 runs, cheaply. (Charge-correction off so the atom order
    # is preserved -- correct_charge sorts -- making the per-index frozen check exact.)
    import numpy as np
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.io import write, read
    from tests.dummy_interface import DummyInterface

    sub = Atoms("Al4O4",
                positions=[[3, 3, 4.0], [9, 3, 4.0], [3, 9, 4.0], [9, 9, 4.0],
                           [6, 3, 5.5], [3, 6, 5.5], [9, 6, 5.5], [6, 9, 5.5]],
                cell=[12.0, 12.0, 24.0], pbc=True)
    sub.set_constraint(FixAtoms(indices=list(range(len(sub)))))
    poscar = tmp_path / "substrate.vasp"
    write(str(poscar), sub, format="vasp")
    sub_pos = sub.get_positions().copy()

    monkeypatch.setattr("runner.runner.build_calculator",
                        lambda spec: DummyInterface(dump_path=str(tmp_path / "d")))
    cfg = load_config({
        "structure": {"from_file": str(poscar)},
        "composition": {"ratio": {"Si": 1, "O": 2}, "total_atoms": len(sub) + 9},
        "limits": {"bottom": {"type": "surface"},                       # at the substrate top
                   "top": {"type": "fourier", "offset": 6.0, "alpha": 1.0}, "fix": "bottom"},
        "calculators": {"growth": {"type": "mace", "mace_model_path": "m"},
                        "saturation": {"type": "mace", "mace_model_path": "m"}},
        "finalize": {"fmax": 0.5, "max_steps": 10},
        "saturation": {"enabled": True},
        "charge_correction": {"enabled": False},
        "statistics": {"enabled": False},
        "run": {"seeds": [0], "output_dir": str(tmp_path / "out")},
    })
    written = run_from_config(cfg)
    assert len(written) == 1
    atoms = read(str(written[0]))
    assert len(atoms) > len(sub)                                        # grew on top
    # the substrate (first len(sub) atoms) is frozen -- positions unchanged
    assert np.allclose(atoms.get_positions()[:len(sub)], sub_pos, atol=1e-6)
    # and stayed Al/O (not overwritten by grown Si)
    assert atoms.get_chemical_symbols()[:len(sub)] == list(sub.get_chemical_symbols())


# --- saturation capping-mode wiring --------------------------------------------------

def _sat_cfg(tmp_path, sat):
    # growth=lammps + a mace saturation calc passes the BKS guard; build_calculator is
    # monkeypatched to the LJ dummy in the tests below so no real backend is needed.
    return load_config({
        "structure": {"cell": [16.0, 16.0, 30.0]},
        "composition": {"target_ratios": {"Si": 1, "O": 2}, "target_number_atoms": 45},
        "limits": {"bottom": {"type": "flat", "z": 10.0},
                   "top": {"type": "fourier", "z_av": 18.0, "alpha": 1.0}, "fix": "bottom"},
        "calculators": {"growth": {"type": "lammps"},
                        "saturation": {"type": "mace", "mace_model_path": "m"}},
        "saturation": sat,
        "charge_correction": {"enabled": True},
        "run": {"output_dir": str(tmp_path / "out"), "seeds": [0]},
    })


def test_runner_threads_fragments_to_relabel(tmp_path, monkeypatch):
    # The runner must pass the configured fragments to relabel_caps. Stub the saturation engine
    # (so the test doesn't depend on correct_charge converging on LJ-relaxed geometry) and
    # record the relabel call.
    import runner.runner as RR
    from tests.dummy_interface import DummyInterface
    monkeypatch.setattr(RR, "build_calculator",
                        lambda spec: DummyInterface(dump_path=str(tmp_path / "dump")))
    monkeypatch.setattr(RR, "saturate_under_coordinated", lambda *a, **k: None)
    monkeypatch.setattr(RR, "correct_charge", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(RR, "relabel_caps", lambda struct, **kw: calls.append(kw))

    RR.run_from_config(_sat_cfg(tmp_path, {"enabled": True, "negative_fragment": "F"}))
    assert len(calls) == 1
    assert calls[0]["negative_fragment"] == ("F",)
    assert calls[0]["positive_fragment"] == ("H",)


def test_runner_skips_relabel_for_default_caps(tmp_path, monkeypatch):
    # Default -OH/H fragments -> relabel is a no-op and must not be invoked (or finalised again).
    import runner.runner as RR
    from tests.dummy_interface import DummyInterface
    monkeypatch.setattr(RR, "build_calculator",
                        lambda spec: DummyInterface(dump_path=str(tmp_path / "dump")))
    monkeypatch.setattr(RR, "saturate_under_coordinated", lambda *a, **k: None)
    monkeypatch.setattr(RR, "correct_charge", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(RR, "relabel_caps", lambda struct, **kw: calls.append(kw))

    RR.run_from_config(_sat_cfg(tmp_path, {"enabled": True}))   # default OH/H
    assert calls == []


# --- Phase 0: per-combo seeding, debug dumps, manifest -----------------------------

def test_combo_seed_decorrelates_and_reproduces():
    # Same (seed, alpha) -> identical stream; differing in either -> different stream.
    def draw(rng):
        return rng.integers(0, 2**31, size=8).tolist()
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


def test_resume_skips_existing_outputs(tmp_path, monkeypatch):
    import json
    from tests.dummy_interface import DummyInterface
    monkeypatch.setattr("runner.runner.build_calculator",
                        lambda spec: DummyInterface(dump_path=str(tmp_path / "dump")))
    cfg = _blank_cfg(tmp_path, seeds=[0, 1])
    cfg.statistics.enabled = False
    run_from_config(cfg)
    out = tmp_path / "out"
    paths = sorted(out.glob("seed*/structure.vasp"))
    assert len(paths) == 2
    survivor_mtime = paths[1].stat().st_mtime_ns

    paths[0].unlink()                       # simulate one slab missing (failed/interrupted)
    run_from_config(cfg, resume=True)       # only the missing one should regenerate
    man = json.loads((out / "manifest.json").read_text())
    assert man["skipped"] == 1 and man["succeeded"] == 1
    assert paths[0].exists()                                  # regenerated
    assert paths[1].stat().st_mtime_ns == survivor_mtime      # untouched


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


# --- Phase 1: sharding + in-node parallel pool -------------------------------------

def test_shard_plan_is_disjoint_and_exhaustive():
    plan = list(range(10))
    shards = [_shard_plan(plan, i, 3) for i in range(3)]
    flat = sorted(x for s in shards for x in s)
    assert flat == plan                                  # union == whole plan
    assert sum(len(s) for s in shards) == len(plan)      # no overlap
    assert _shard_plan(plan, None, None) == plan         # no sharding -> untouched


def test_shard_plan_validation():
    with pytest.raises(ValueError):
        _shard_plan([1, 2], 0, None)        # shard without num_shards
    with pytest.raises(ValueError):
        _shard_plan([1, 2], None, 2)        # num_shards without shard
    with pytest.raises(ValueError):
        _shard_plan([1, 2], 3, 3)           # shard out of range


def test_sharded_runs_then_pool_reduce(tmp_path, monkeypatch):
    # Each shard generates its slice serially (workers=1 keeps the dummy monkeypatch in
    # process), writes its own manifest, and skips pooling. The reduce merges manifests
    # and pools all structures off disk -- the cross-shard collection guarantee.
    import json
    from tests.dummy_interface import DummyInterface
    monkeypatch.setattr("runner.runner.build_calculator",
                        lambda spec: DummyInterface(dump_path=str(tmp_path / "dump")))
    cfg = _blank_cfg(tmp_path, seeds=[0, 1, 2, 3])
    run_from_config(cfg, shard=0, num_shards=2)
    run_from_config(cfg, shard=1, num_shards=2)

    out = tmp_path / "out"
    assert {p.name for p in out.glob("manifest.shard*of*.json")} == \
        {"manifest.shard0of2.json", "manifest.shard1of2.json"}
    assert not (out / "stats_pooled.json").exists()      # sharded runs defer pooling

    pool_from_config(cfg)                                # reduce step
    merged = json.loads((out / "manifest.json").read_text())
    assert merged["total"] == 4 and merged["succeeded"] == 4
    assert json.loads((out / "stats_pooled.json").read_text())["summary"]["n_structures"] == 4


def test_worker_functions_build_and_run(tmp_path, monkeypatch):
    # In-process check of the pool's worker functions (no subprocess): the initializer
    # builds the calculators into _WORKER and _worker_run produces a correct record.
    import runner.runner as RR
    from tests.dummy_interface import DummyInterface
    monkeypatch.setattr("runner.runner.build_calculator",
                        lambda spec: DummyInterface(dump_path=str(tmp_path / "dump")))
    cfg = _blank_cfg(tmp_path, seeds=[0])
    cfg.statistics.enabled = False
    entry = resolve_plan(cfg)[0]

    RR._worker_init(cfg, threads=1, calc_provider=RR._default_calc_provider)
    assert RR._WORKER["growth"] is not None
    record = RR._worker_run(entry)
    assert record["status"] == "ok"
    assert Path(record["output_path"]).exists()


def test_run_from_config_parallel_pool_matches_serial(tmp_path):
    # Real ProcessPoolExecutor under the platform default (spawn). The picklable
    # calc_provider lets workers build the LJ dummy without LAMMPS/torch.
    import json
    from tests.dummy_interface import dummy_calc_provider

    cfg = _blank_cfg(tmp_path, seeds=[0, 1, 2, 3])
    cfg.statistics.enabled = False
    written = run_from_config(cfg, workers=2, threads=1, calc_provider=dummy_calc_provider)

    assert len(written) == 4
    assert all(p.exists() for p in written)
    man = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert man["total"] == 4 and man["succeeded"] == 4


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
