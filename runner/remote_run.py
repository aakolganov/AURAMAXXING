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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Union

from .config import RunConfig, load_config
from .runner import (resolve_plan, build_calculator, _shard_plan, _run_entry,
                     _write_manifest, _pool_stats)
from interfaces.remote_calculator import RemoteCalculator, evaluator_loop


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


# Per-worker state for the CPU pool.
_REMOTE: dict = {}


def _remote_worker_init(cfg, threads, request_q, response_qs, counter, lock) -> None:
    from .threads import configure_threads
    configure_threads(threads)
    with lock:
        idx = counter.value
        counter.value = idx + 1
    if idx >= len(response_qs):
        raise RuntimeError("more pool workers than response queues (a worker respawned?); "
                           "set workers to the pool size used at start")
    remote_calc = RemoteCalculator(request_q, response_qs[idx], idx)
    growth, sat = _stage_calcs(cfg, remote_calc)
    _REMOTE.update(cfg=cfg, growth=growth, sat=sat)


def _remote_worker_run(entry: dict) -> dict:
    return _run_entry(_REMOTE["cfg"], entry, _REMOTE["growth"], _REMOTE["sat"])


def run_remote_from_config(source: Union[str, Path, dict, RunConfig], *,
                           workers: int = 1, worker_threads: Optional[int] = 1,
                           device: str = "cuda", evaluator_threads: Optional[int] = None,
                           calc_factory=None, shard: Optional[int] = None,
                           num_shards: Optional[int] = None) -> list[Path]:
    """Run the sweep with MACE evaluated on ``device`` by one evaluator process feeding a pool
    of ``workers`` CPU processes. ``worker_threads`` caps each worker's threads (they only do
    placement, so 1 is usually right); ``evaluator_threads`` caps the evaluator's. Sharding,
    manifest and pooling behave exactly as in ``run_from_config``.
    """
    cfg = source if isinstance(source, RunConfig) else load_config(source)
    plan = _shard_plan(resolve_plan(cfg), shard, num_shards)
    factory = calc_factory or _default_remote_factory(cfg)

    # CUDA cannot be initialised after fork, so the evaluator and the pool both use spawn.
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    request_q = manager.Queue()
    response_qs = [manager.Queue() for _ in range(max(1, workers))]
    counter = manager.Value("i", 0)
    lock = manager.Lock()

    evaluator = ctx.Process(target=evaluator_loop,
                            args=(request_q, response_qs, factory, device, evaluator_threads),
                            daemon=True)
    evaluator.start()
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_remote_worker_init,
                                 initargs=(cfg, worker_threads, request_q, response_qs,
                                           counter, lock)) as ex:
            records = list(ex.map(_remote_worker_run, plan))
    finally:
        request_q.put(None)          # stop the evaluator
        evaluator.join(timeout=30)
        if evaluator.is_alive():
            evaluator.terminate()

    written = [Path(r["output_path"]) for r in records if r["status"] == "ok"]
    _write_manifest(cfg, records, shard=shard, num_shards=num_shards)
    if cfg.statistics.enabled and cfg.statistics.pooled:
        if num_shards is None:
            _pool_stats(cfg)
        else:
            print(f"[runner] sharded run: skipping pooled statistics; run `--pool-only` "
                  f"over {cfg.run.output_dir} after all shards finish")
    return written
