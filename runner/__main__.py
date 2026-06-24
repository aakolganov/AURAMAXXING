"""Command-line entry point: ``python -m runner config.yaml`` (or the ``auramaxxing``
console script once installed)."""
import argparse
import sys
from typing import Optional

from .config import load_config
from .runner import resolve_plan, run_from_config, pool_from_config


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

    plan = resolve_plan(cfg)
    print(f"Resolved {len(plan)} structure(s) from {args.config}:")
    for entry in plan:
        alpha = entry["alpha"]
        alpha_str = f"{alpha:.3f}" if alpha is not None else "flat"
        print(f"  seed={entry['seed']:<4} alpha={alpha_str:<6} -> {entry['output_path']}")

    if args.dry_run:
        return 0

    run_from_config(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
