"""Scenario 3 orchestration: a GPU evaluator process + a pool of CPU worker processes.

One process owns the MACE model on the GPU (``interfaces.remote_calculator.evaluator_loop``);
the worker pool runs the placement/optimization control flow on CPU and routes every MACE
force/energy call to that process via a ``RemoteCalculator``. Stages whose calculator is not
MACE (e.g. LAMMPS/BKS growth) are built locally in each worker and run on CPU as usual.

Reuses the per-slab unit (`_run_entry`), sharding, manifest and pooling from ``runner.runner``
so the output/collection behaviour matches the other run paths exactly.
"""
from __future__ import annotations

import functools
import multiprocessing as mp
import queue as _queue
from pathlib import Path
from typing import Optional, Union

from .config import RunConfig, load_config
from .runner import (resolve_plan, build_calculator, _shard_plan, _partition_resume,
                     _skipped_record, _run_entry, _write_manifest, _pool_stats)
from interfaces.remote_calculator import (RemoteCalculator, evaluator_loop,
                                          DEFAULT_RESPONSE_TIMEOUT)


def _build_mace_calc(device: str, *, model_path: str, **opts):
    """Picklable factory body: build a MACE calculator on ``device`` and return its ASE calc.
    Runs inside the evaluator process so the model is loaded once, on the GPU."""
    from interfaces.MACE_interface import MACEInterface
    return MACEInterface(mace_model_path=model_path, device=device, **opts).calc


def _default_remote_factory(cfg: RunConfig):
    """A picklable ``calc_factory(device)`` derived from the config's MACE calculator (the
    saturation one if present, else growth)."""
    spec = cfg.calculators.saturation or cfg.calculators.growth
    if spec.type != "mace":
        raise ValueError("remote runs need a MACE calculator: set calculators.saturation "
                         "(or growth) to type: mace")
    opts = dict(spec.options)
    model_path = opts.pop("mace_model_path")
    opts.pop("device", None)
    opts.pop("dump_path", None)
    return functools.partial(_build_mace_calc, model_path=model_path, **opts)


def _stage_calcs(cfg: RunConfig, remote_calc: RemoteCalculator):
    """Pick the calculator for each stage: the remote (GPU) one for MACE stages, a locally
    built CPU calculator for the rest. One evaluator/model backs all MACE stages."""
    def pick(spec):
        return remote_calc if spec.type == "mace" else build_calculator(spec)
    growth = pick(cfg.calculators.growth)
    sat_spec = cfg.calculators.saturation
    sat = pick(sat_spec) if sat_spec else growth
    return growth, sat


def _worker_main(wid, cfg, threads, request_q, response_q, task_q, result_q, timeout) -> None:
    """One CPU worker process: pull plan entries from ``task_q``, generate each slab (routing
    MACE evaluations to the shared evaluator through a RemoteCalculator), and put the manifest
    record on ``result_q``. A ``None`` task is the stop signal."""
    from .threads import configure_threads
    configure_threads(threads)
    remote_calc = RemoteCalculator(request_q, response_q, wid, timeout=timeout)
    growth, sat = _stage_calcs(cfg, remote_calc)
    while True:
        entry = task_q.get()
        if entry is None:
            break
        result_q.put(_run_entry(cfg, entry, growth, sat))


def run_remote_from_config(source: Union[str, Path, dict, RunConfig], *,
                           workers: int = 1, worker_threads: Optional[int] = 1,
                           device: str = "cuda", evaluator_threads: Optional[int] = None,
                           batch_size: Optional[int] = None, response_timeout: Optional[float] = None,
                           calc_factory=None, shard: Optional[int] = None,
                           num_shards: Optional[int] = None, resume: bool = False) -> list[Path]:
    """Run the sweep with MACE evaluated on ``device`` by one evaluator process feeding a pool
    of ``workers`` CPU processes. ``worker_threads`` caps each worker's threads (they only do
    placement, so 1 is usually right); ``evaluator_threads`` caps the evaluator's.
    ``batch_size`` is how many queued requests the evaluator may coalesce into one forward pass
    (default = ``workers``, the most that can be outstanding at once). ``resume`` skips
    combinations whose output already exists. Sharding, manifest and pooling behave exactly as
    in ``run_from_config``.

    IPC uses plain ``multiprocessing`` queues passed to the worker/evaluator processes at
    creation (no Manager-proxy hop). Workers pull from a shared task queue, so the slowest
    slabs don't stall the others.
    """
    cfg = source if isinstance(source, RunConfig) else load_config(source)
    plan = _shard_plan(resolve_plan(cfg), shard, num_shards)
    plan, done = _partition_resume(plan, resume)
    if done:
        print(f"[runner] resume: skipping {len(done)} already-written structures")
    factory = calc_factory or _default_remote_factory(cfg)
    workers = max(1, workers)
    if batch_size is None:
        batch_size = workers
    timeout = response_timeout if response_timeout is not None else DEFAULT_RESPONSE_TIMEOUT

    # CUDA cannot be initialised after fork, so the evaluator and workers all use spawn. Plain
    # ctx.Queue()s are passed at process creation (they aren't picklable through a pool's
    # initargs, which is why this uses explicit Processes rather than ProcessPoolExecutor).
    ctx = mp.get_context("spawn")
    request_q = ctx.Queue()
    result_q = ctx.Queue()
    task_q = ctx.Queue()
    response_qs = [ctx.Queue() for _ in range(workers)]
    for entry in plan:
        task_q.put(entry)
    for _ in range(workers):
        task_q.put(None)                       # one poison pill per worker, after all tasks

    evaluator = ctx.Process(target=evaluator_loop,
                            args=(request_q, response_qs, factory, device,
                                  evaluator_threads, batch_size), daemon=True)
    evaluator.start()
    procs = [ctx.Process(target=_worker_main,
                         args=(i, cfg, worker_threads, request_q, response_qs[i],
                               task_q, result_q, timeout), daemon=True)
             for i in range(workers)]
    for p in procs:
        p.start()

    records: list = []
    try:
        while len(records) < len(plan):
            try:
                records.append(result_q.get(timeout=5))
            except _queue.Empty:
                if not any(p.is_alive() for p in procs):
                    while True:                # drain anything still buffered, then stop
                        try:
                            records.append(result_q.get(timeout=2))
                        except _queue.Empty:
                            break
                    break
    finally:
        for p in procs:
            p.join(timeout=30)
        request_q.put(None)                    # stop the evaluator
        evaluator.join(timeout=30)
        if evaluator.is_alive():
            evaluator.terminate()
        for q in (request_q, result_q, task_q, *response_qs):
            q.cancel_join_thread()             # consumers are done; don't block at exit

    records += [_skipped_record(e) for e in done]
    written = [Path(r["output_path"]) for r in records if r["status"] in ("ok", "skipped")]
    _write_manifest(cfg, records, shard=shard, num_shards=num_shards)
    if cfg.statistics.enabled and cfg.statistics.pooled:
        if num_shards is None:
            _pool_stats(cfg)
        else:
            print(f"[runner] sharded run: skipping pooled statistics; run `--pool-only` "
                  f"over {cfg.run.output_dir} after all shards finish")
    return written
