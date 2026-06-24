"""Tests for the scenario-3 remote evaluator (interfaces.remote_calculator + remote_run).

The GPU evaluator's device is configurable, so these run the whole IPC path on CPU with an
LJ-backed evaluator -- no GPU, torch or MACE required.
"""
import json
import multiprocessing as mp

import numpy as np

from interfaces.remote_calculator import RemoteCalculator, evaluator_loop
from runner.config import load_config
from runner.remote_run import run_remote_from_config
from tests.dummy_interface import lj_remote_factory


def _remote_cfg(tmp_path, **run):
    # growth uses MACE so the remote run routes it to the evaluator; the LJ factory means the
    # model path is never actually loaded.
    return load_config({
        "structure": {"cell": [16.0, 16.0, 30.0]},
        "composition": {"target_ratios": {"Si": 1, "O": 2}, "target_number_atoms": 24},
        "limits": {"bottom": {"type": "flat", "z": 10.0},
                   "top": {"type": "fourier", "z_av": 18.0, "alpha": 1.0}, "fix": "bottom"},
        "calculators": {"growth": {"type": "mace", "mace_model_path": "unused.model"}},
        "statistics": {"enabled": False},
        "run": {"output_dir": str(tmp_path / "out"), **run},
    })


def test_remote_evaluator_roundtrip(make_struct, tmp_path):
    # A worker-side RemoteCalculator drives a local LBFGS whose forces come from a separate
    # evaluator process. Verifies the proxy round-trip end to end.
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    request_q = manager.Queue()
    response_q = manager.Queue()
    evaluator = ctx.Process(target=evaluator_loop,
                            args=(request_q, [response_q], lj_remote_factory, "cpu", 1),
                            daemon=True)
    evaluator.start()
    try:
        rc = RemoteCalculator(request_q, response_q, 0, dump_path=str(tmp_path / "d"))
        s = make_struct(["Si", "O", "O"], [[5, 5, 5], [6.6, 5, 5], [3.4, 5, 5]],
                        cell=(15.0, 15.0, 15.0))
        rc.optimize(s.atoms, fmax=1.0, max_steps=3, workdir=str(tmp_path / "d"))
        assert np.isfinite(s.atoms.get_potential_energy())
        assert np.all(np.isfinite(s.atoms.get_forces()))
    finally:
        request_q.put(None)
        evaluator.join(timeout=30)


def test_run_remote_end_to_end(tmp_path):
    # Full scenario-3 path: one evaluator process + a 2-worker CPU pool, on CPU via LJ.
    cfg = _remote_cfg(tmp_path, seeds=[0, 1])
    written = run_remote_from_config(cfg, workers=2, worker_threads=1, device="cpu",
                                     calc_factory=lj_remote_factory)
    assert len(written) == 2
    for path in written:
        assert path.exists()
    man = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert man["total"] == 2 and man["succeeded"] == 2
