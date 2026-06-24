"""Remote (out-of-process) force/energy evaluation.

Scenario 3: one process owns the model on the GPU; many CPU worker processes run the
placement/optimization control flow and ship each configuration they need evaluated to that
process, blocking for the energy + forces. The worker never touches CUDA.

``RemoteCalculator`` is a drop-in ``CalculatorInterface``: ``optimize``/``anneal`` run the
existing ASE ``LBFGS``/``AnnealingLangevin`` locally, but the attached ASE calculator is a
thin proxy (``RemoteASECalculator``) that round-trips each ``calculate`` to the evaluator.
``evaluator_loop`` is the server side. IPC is via ``multiprocessing.Manager`` queues (their
proxies are picklable, so they survive the spawn start method that CUDA requires).
"""
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from interfaces.base_interface import CalculatorInterface


class RemoteASECalculator(Calculator):
    """ASE calculator proxy: ships (numbers, positions, cell, pbc) to the evaluator process
    and blocks on this worker's response queue for the energy + forces."""
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, request_q, response_q, worker_id: int, **kwargs):
        super().__init__(**kwargs)
        self._request_q = request_q
        self._response_q = response_q
        self._worker_id = worker_id

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._request_q.put({
            "id": self._worker_id,
            "numbers": atoms.get_atomic_numbers(),
            "positions": atoms.get_positions(),
            "cell": np.asarray(atoms.get_cell()),
            "pbc": np.asarray(atoms.get_pbc()),
        })
        result = self._response_q.get()
        if "error" in result:
            raise RuntimeError(f"remote evaluator failed: {result['error']}")
        energy = float(result["energy"])
        self.results = {"energy": energy, "free_energy": energy, "forces": result["forces"]}


class RemoteCalculator(CalculatorInterface):
    """A ``CalculatorInterface`` whose optimize/anneal run locally (ASE driver) but whose
    force/energy evaluations are served by a remote evaluator process. Inherits the base
    optimize/anneal unchanged -- they only ever touch ``self.calc``, which is the proxy."""

    def __init__(self, request_q, response_q, worker_id: int, dump_path: str = "dump"):
        self.calc = RemoteASECalculator(request_q, response_q, worker_id)
        self.dump_path = Path(dump_path)
        self.dump_path.mkdir(parents=True, exist_ok=True)


def evaluator_loop(request_q, response_qs, calc_factory, device: str = "cuda",
                   threads=None) -> None:
    """Server side, run in a dedicated process: build the model on ``device`` via
    ``calc_factory(device)`` (a picklable callable returning an ASE calculator) and serve
    energy+forces for configurations arriving on ``request_q``, replying on the requester's
    response queue. A ``None`` request stops the loop.

    Serves FIFO -- one request at a time -- which already keeps a single GPU busy when many
    CPU workers feed it. Dynamic batching (coalescing same-shaped requests) can be layered on
    later without changing the worker side.
    """
    if threads is not None:
        from runner.threads import configure_threads
        configure_threads(threads)
    calc = calc_factory(device)
    while True:
        req = request_q.get()
        if req is None:
            break
        wid = req["id"]
        try:
            atoms = Atoms(numbers=req["numbers"], positions=req["positions"],
                          cell=req["cell"], pbc=req["pbc"])
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            response_qs[wid].put({"energy": float(energy), "forces": np.asarray(forces)})
        except Exception as exc:   # report back so the worker raises instead of hanging
            response_qs[wid].put({"error": f"{type(exc).__name__}: {exc}"})
