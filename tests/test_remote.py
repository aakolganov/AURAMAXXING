"""Tests for the scenario-3 remote evaluator (interfaces.remote_calculator + remote_run).

The GPU evaluator's device is configurable, so these run the whole IPC path on CPU with an
LJ-backed evaluator -- no GPU, torch or MACE required.
"""
import json
import multiprocessing as mp

import numpy as np
import pytest

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


def test_remote_timeout_fails_fast(tmp_path):
    # No evaluator consumes the request, so the response never arrives: the proxy must raise
    # (not hang) once the timeout elapses, turning a dead evaluator into a failed slab.
    import time
    import pytest
    from ase import Atoms
    from interfaces.remote_calculator import RemoteCalculator

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    rc = RemoteCalculator(manager.Queue(), manager.Queue(), 0,
                          dump_path=str(tmp_path / "d"), timeout=0.5)
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.6, 0, 0]], cell=[10, 10, 10], pbc=True)
    atoms.calc = rc.calc
    t = time.time()
    with pytest.raises(RuntimeError, match="did not respond"):
        atoms.get_potential_energy()
    assert time.time() - t < 5.0       # failed fast, did not hang


def test_remote_calc_discards_stale_reply():
    # H4: after a worker times out on a slow request, the evaluator may still deliver that
    # request's (now-stale) reply later. The proxy must recognise it by token and discard it,
    # not read it as the NEXT request's result -- which would silently drive the optimizer with
    # energy/forces computed for a different configuration. Pure CPU, no evaluator process.
    import queue
    import threading
    from ase import Atoms
    from interfaces.remote_calculator import RemoteASECalculator

    req_q, resp_q = queue.Queue(), queue.Queue()
    calc = RemoteASECalculator(req_q, resp_q, worker_id=0, timeout=5.0)

    def fake_evaluator():
        req = req_q.get(timeout=5.0)
        n = len(req["numbers"])
        # A late reply from a previously-abandoned request (wrong token) arrives FIRST ...
        resp_q.put({"token": req["token"] + 9999, "energy": -111.0, "forces": np.zeros((n, 3))})
        # ... then the correct reply for THIS request.
        resp_q.put({"token": req["token"], "energy": 42.0, "forces": np.ones((n, 3))})

    t = threading.Thread(target=fake_evaluator)
    t.start()
    try:
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10], pbc=True)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        assert energy == 42.0                              # matched reply, not the stale -111.0
        assert np.allclose(atoms.get_forces(), 1.0)        # forces from the matched reply
    finally:
        t.join(timeout=5.0)


@pytest.mark.heavy
def test_mace_batched_matches_sequential():
    # #4 correctness: the batched forward pass must equal per-config evaluation (the basis for
    # the batched path being a pure speed optimization). Device-independent, so run on CPU.
    pytest.importorskip("mace")
    from ase import Atoms
    from interfaces.MACE_interface import MACEInterface
    from interfaces.remote_calculator import _mace_batched_eval, _eval_one, _supports_batching

    calc = MACEInterface(mace_model_path="small", device="cpu").calc
    assert _supports_batching(calc)

    def req(symbols, coords):
        a = Atoms(symbols, positions=coords, cell=[12, 12, 12], pbc=True)
        return {"numbers": a.get_atomic_numbers(), "positions": a.get_positions(),
                "cell": np.asarray(a.get_cell()), "pbc": a.get_pbc()}

    # deliberately varied atom counts to exercise variable-size batching
    reqs = [
        req(["Si", "O", "O"], [[3, 3, 3], [4.6, 3, 3], [3, 4.6, 3]]),
        req(["Si", "O", "O", "O", "Si"], [[3, 3, 3], [4.6, 3, 3], [3, 4.6, 3], [3, 3, 4.6], [6, 6, 6]]),
        req(["O", "Si"], [[5, 5, 5], [6.6, 5, 5]]),
    ]
    batched = _mace_batched_eval(calc, reqs)
    for r, b in zip(reqs, batched, strict=True):
        seq = _eval_one(calc, r)
        assert abs(seq["energy"] - b["energy"]) < 1e-6
        assert np.abs(seq["forces"] - b["forces"]).max() < 1e-6


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
