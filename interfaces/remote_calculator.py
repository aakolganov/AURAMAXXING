"""Remote (out-of-process) force/energy evaluation.

Scenario 3: one process owns the model on the GPU; many CPU worker processes run the
placement/optimization control flow and ship each configuration they need evaluated to that
process, blocking for the energy + forces. The worker never touches CUDA.

``RemoteCalculator`` is a drop-in ``CalculatorInterface``: ``optimize``/``anneal`` run the
existing ASE ``LBFGS``/``AnnealingLangevin`` locally, but the attached ASE calculator is a
thin proxy (``RemoteASECalculator``) that round-trips each ``calculate`` to the evaluator.
``evaluator_loop`` is the server side; it can batch several queued requests into one forward
pass. IPC uses plain ``multiprocessing`` queues (passed to the worker/evaluator processes at
creation), avoiding the extra hop of a Manager proxy.
"""
import queue as _queue
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from interfaces.base_interface import CalculatorInterface

# Default per-call ceiling: how long a worker waits for the evaluator before giving up.
# Generous (a single MACE eval is seconds at most); its job is to turn a dead/wedged
# evaluator into a failed slab instead of a job that hangs until SLURM wall-time.
DEFAULT_RESPONSE_TIMEOUT = 600.0


class RemoteASECalculator(Calculator):
    """ASE calculator proxy: ships (numbers, positions, cell, pbc) to the evaluator process
    and blocks on this worker's response queue for the energy + forces."""
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, request_q, response_q, worker_id: int,
                 timeout: float = DEFAULT_RESPONSE_TIMEOUT, **kwargs):
        super().__init__(**kwargs)
        self._request_q = request_q
        self._response_q = response_q
        self._worker_id = worker_id
        self._timeout = timeout
        # Per-worker monotonic request counter. Each request carries a token echoed back in
        # its reply; this worker only ever has one request outstanding, so the token lets us
        # tell the current reply from a late one for a request we already gave up on.
        self._seq = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._seq += 1
        token = self._seq
        self._request_q.put({
            "id": self._worker_id,
            "token": token,
            "numbers": atoms.get_atomic_numbers(),
            "positions": atoms.get_positions(),
            "cell": np.asarray(atoms.get_cell()),
            "pbc": np.asarray(atoms.get_pbc()),
        })
        # Wait for THIS request's reply. A reply whose token doesn't match belongs to an
        # earlier request this worker abandoned on timeout (the evaluator never drops a
        # request, only the worker does); discard it and keep waiting, otherwise it would be
        # read as this request's result and silently drive the optimizer with forces computed
        # for a stale configuration. Tokens are monotonic and never reused, so a stale reply's
        # token is always < token and can never spuriously match.
        while True:
            try:
                result = self._response_q.get(timeout=self._timeout)
            except _queue.Empty:
                raise RuntimeError(
                    f"remote evaluator did not respond within {self._timeout:.0f}s "
                    f"(evaluator dead or wedged)") from None
            if result.get("token") == token:
                break
        if "error" in result:
            raise RuntimeError(f"remote evaluator failed: {result['error']}")
        energy = float(result["energy"])
        self.results = {"energy": energy, "free_energy": energy, "forces": result["forces"]}


class RemoteCalculator(CalculatorInterface):
    """A ``CalculatorInterface`` whose optimize/anneal run locally (ASE driver) but whose
    force/energy evaluations are served by a remote evaluator process. Inherits the base
    optimize/anneal unchanged -- they only ever touch ``self.calc``, which is the proxy."""

    def __init__(self, request_q, response_q, worker_id: int, dump_path: str = "dump",
                 timeout: float = DEFAULT_RESPONSE_TIMEOUT):
        self.calc = RemoteASECalculator(request_q, response_q, worker_id, timeout=timeout)
        self.dump_path = Path(dump_path)
        self.dump_path.mkdir(parents=True, exist_ok=True)


def _atoms_from_req(req) -> Atoms:
    return Atoms(numbers=req["numbers"], positions=req["positions"],
                 cell=req["cell"], pbc=req["pbc"])


def _eval_one(calc, req) -> dict:
    """Evaluate one config through the ASE calculator. Captures errors as a reply so a single
    bad config fails just its own slab, not the whole batch."""
    try:
        atoms = _atoms_from_req(req)
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        return {"energy": energy, "forces": np.asarray(atoms.get_forces())}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _supports_batching(calc) -> bool:
    """True for a single-model MACE calculator whose own preprocessing we can reuse to build
    a multi-graph batch. Anything else (ensembles, non-MACE) takes the per-config path."""
    models = getattr(calc, "models", None)
    return hasattr(calc, "_atoms_to_batch") and isinstance(models, (list, tuple)) and len(models) == 1


def _mace_real_graph(calc, atoms):
    """Build the single-config ``AtomicData`` exactly as ``MACECalculator._atoms_to_batch``
    does (same key spec, z-table, cutoff, dtype), so a batch of these reproduces per-config
    results bit-for-bit."""
    from mace import data as mace_data
    from mace.tools import torch_tools

    arrays_keys = dict(calc.arrays_keys)
    arrays_keys.update({calc.charges_key: "charges"})
    keyspec = mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=arrays_keys)
    with torch_tools.default_dtype(calc.default_dtype):
        config = mace_data.config_from_atoms(atoms, key_specification=keyspec, head_name=calc.head)
        return mace_data.AtomicData.from_config(
            config, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)


def _mace_batched_eval(calc, reqs: list) -> list:
    """Evaluate several configs in ONE MACE forward pass: build each config's graph the way the
    calculator does, collate them with MACE's own ``DataLoader``, run the model once, then split
    energy/forces per graph (via the batch ``ptr``) and apply the calculator's unit conversions.
    Matches per-config output."""
    import torch
    from mace.tools import torch_geometric

    model = calc.models[0]
    graphs = [_mace_real_graph(calc, _atoms_from_req(r)) for r in reqs]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=graphs, batch_size=len(graphs), shuffle=False, drop_last=False)
    batch = next(iter(loader)).to(calc.device)
    model_dtype = next(model.parameters()).dtype
    for key in batch.keys:
        value = batch[key]
        if torch.is_tensor(value) and torch.is_floating_point(value):
            batch[key] = value.to(dtype=model_dtype)
    out = model(batch.to_dict(), training=False, compute_force=True, compute_stress=False)

    e_conv = calc.energy_units_to_eV
    f_conv = calc.energy_units_to_eV / calc.length_units_to_A
    energies = out["energy"].detach().double().cpu().numpy()
    forces = out["forces"].detach().double().cpu().numpy()
    ptr = batch.ptr.detach().cpu().numpy()
    return [{"energy": float(energies[i]) * e_conv,
             "forces": np.asarray(forces[ptr[i]:ptr[i + 1]] * f_conv)}
            for i in range(len(reqs))]


def evaluator_loop(request_q, response_qs, calc_factory, device: str = "cuda",
                   threads=None, batch_size: int = 1) -> None:
    """Server side, run in a dedicated process: build the model on ``device`` via
    ``calc_factory(device)`` (a picklable callable returning an ASE calculator) and serve
    energy+forces for configurations arriving on ``request_q``, replying on each requester's
    response queue. A ``None`` request stops the loop.

    With ``batch_size > 1`` it greedily drains up to that many requests already waiting and
    evaluates them in a single batched forward pass (when the calculator is a single-model
    MACE), so the GPU does one larger pass instead of N small ones. The batched path falls
    back to per-config evaluation on any error, so it can never change results -- only speed.
    """
    if threads is not None:
        from runner.threads import configure_threads
        configure_threads(threads)
    calc = calc_factory(device)
    can_batch = batch_size > 1 and _supports_batching(calc)

    stop = False
    while not stop:
        first = request_q.get()
        if first is None:
            break
        batch = [first]
        while len(batch) < batch_size:                 # coalesce whatever is already queued
            try:
                nxt = request_q.get_nowait()
            except _queue.Empty:
                break
            if nxt is None:
                stop = True
                break
            batch.append(nxt)

        if can_batch and len(batch) > 1:
            try:
                replies = _mace_batched_eval(calc, batch)
            except Exception:                          # never let batching break a result
                replies = [_eval_one(calc, b) for b in batch]
        else:
            replies = [_eval_one(calc, b) for b in batch]

        for req, reply in zip(batch, replies, strict=True):
            # Echo the request's token so the requester can match this reply to the request it
            # is currently blocked on (and discard a late reply for an abandoned one).
            reply["token"] = req.get("token")
            response_qs[req["id"]].put(reply)
