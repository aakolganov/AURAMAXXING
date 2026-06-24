"""Thread/oversubscription control for parallel runs.

The rule that governs throughput on HPC: with ``W`` worker processes each spawning ``T``
compute threads, keep ``W * T <= cores_per_node`` or the workers fight over cores and
everything slows down. The two deployment shapes want opposite settings:

- Scenario 1 (in-node LAMMPS pool): many single-threaded workers -> ``T = 1``.
- Scenario 2 (one MACE slab per node): one worker using every core -> ``T = cores``.

``configure_threads(n)`` applies both controls for the current process:

* the BLAS/OpenMP env vars (``OMP_NUM_THREADS`` etc.), which LAMMPS and the BLAS
  backends read when first initialised. In this pipeline the heavy backends import
  lazily (inside ``runner.build_calculator``), so calling this early in a process --
  before generation starts, or first thing in a pool worker -- sets them in time.
* ``torch.set_num_threads`` for MACE, which takes effect at runtime.

For backends that initialise their threading eagerly at interpreter start (e.g. numpy's
BLAS), the authoritative place is still the launch script: export the env vars in the
sbatch script before ``python`` runs.
"""
import os

_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def set_thread_env(n_threads: int) -> None:
    """Set the BLAS/OpenMP thread-count env vars to ``n_threads`` for this process.

    Effective for backends that read them at import (LAMMPS, BLAS) only when called
    before that import -- so call this early in a process or from the launch script.
    """
    n = str(int(n_threads))
    for var in _THREAD_ENV:
        os.environ[var] = n


def set_torch_threads(n_threads: int) -> None:
    """Limit torch intra-op threads (effective at runtime). No-op if torch is absent."""
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(int(n_threads))


def configure_threads(n_threads: int) -> None:
    """Apply both the env-var and torch thread limits for this process.

    Call once per process: early in the CLI before generation, or as the first thing a
    pool worker does (before it builds its calculator).
    """
    if n_threads is None:
        return
    if int(n_threads) < 1:
        raise ValueError(f"n_threads must be >= 1, got {n_threads}")
    set_thread_env(n_threads)
    set_torch_threads(n_threads)
