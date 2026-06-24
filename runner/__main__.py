"""Command-line entry point: ``python -m runner config.yaml`` (or the ``auramaxxing``
console script once installed)."""
import argparse
import sys
from typing import Optional

from .config import load_config
from .runner import resolve_plan, run_from_config, pool_from_config, _shard_plan


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="auramaxxing",
        description="Generate amorphous oxide surfaces from a YAML config file.",
    )
    parser.add_argument("config", help="path to the YAML configuration file")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the config and print the per-structure plan without running")
    parser.add_argument("--pool-only", action="store_true",
                        help="skip generation; gather the per-structure metrics.json already "
                             "under the output dir and (re)write the pooled report. Run once "
                             "after a parallel/sharded sweep finishes.")
    parser.add_argument("--threads", type=int, default=None, metavar="N",
                        help="limit compute threads per process to N (OMP/BLAS env + torch). "
                             "Use 1 for an in-node pool of many workers; cores-per-node for "
                             "one slab per node. Keep workers*threads <= cores_per_node.")
    parser.add_argument("--workers", type=int, default=1, metavar="W",
                        help="in-node process pool size: run W slabs at once (default 1 = serial). "
                             "Each worker builds its own calculator.")
    parser.add_argument("--num-shards", type=int, default=None, metavar="N",
                        help="split the plan into N disjoint shards (one SLURM array task each). "
                             "Requires --shard.")
    parser.add_argument("--shard", type=int, default=None, metavar="I",
                        help="which shard [0, N) this task runs. Requires --num-shards.")
    parser.add_argument("--resume", action="store_true",
                        help="skip combinations whose output file already exists (re-run an "
                             "interrupted/failed sweep without redoing finished slabs).")
    parser.add_argument("--remote", action="store_true",
                        help="scenario 3: evaluate the MACE calculator in one GPU process and "
                             "run --workers CPU processes that proxy their force/energy calls "
                             "to it. Use --device to pick the evaluator device.")
    parser.add_argument("--device", default="cuda", metavar="DEV",
                        help="device for the --remote evaluator's model (default cuda).")
    parser.add_argument("--batch-size", type=int, default=None, metavar="B",
                        help="max requests the --remote evaluator coalesces into one forward "
                             "pass (default = --workers). 1 disables batching.")
    args = parser.parse_args(argv)

    # Apply before generation: the heavy backends (LAMMPS, torch/MACE) import lazily in
    # build_calculator, so the env vars/torch limit set here take effect in time.
    if args.threads is not None:
        from .threads import configure_threads
        configure_threads(args.threads)

    try:
        cfg = load_config(args.config)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.pool_only:
        pool_from_config(cfg)
        return 0

    try:
        plan = _shard_plan(resolve_plan(cfg), args.shard, args.num_shards)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    shard_note = "" if args.num_shards is None else f" (shard {args.shard}/{args.num_shards})"
    print(f"Resolved {len(plan)} structure(s) from {args.config}{shard_note}:")
    for entry in plan:
        alpha = entry["alpha"]
        alpha_str = f"{alpha:.3f}" if alpha is not None else "flat"
        print(f"  seed={entry['seed']:<4} alpha={alpha_str:<6} -> {entry['output_path']}")

    if args.dry_run:
        return 0

    if args.remote:
        from .remote_run import run_remote_from_config
        run_remote_from_config(cfg, workers=args.workers, worker_threads=args.threads,
                               device=args.device, batch_size=args.batch_size,
                               shard=args.shard, num_shards=args.num_shards, resume=args.resume)
    else:
        run_from_config(cfg, workers=args.workers, threads=args.threads,
                        shard=args.shard, num_shards=args.num_shards, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
