"""Standalone stats: ``python -m stats structure.vasp [--saturation] [--out DIR]``.

Analyses one or more saved structure files. Fixed atoms are recovered from a POSCAR's
Selective Dynamics (ASE reads them into a FixAtoms constraint); the coordination graph is
rebuilt with the default cutoffs. ``--saturation`` enables the O-H / OH-per-cation plots
(it cannot be inferred from a structure file).
"""
import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stats",
        description="Plot structural statistics (coordination, homo-element & element-O "
                    "distances, tau4/tau4', and OH stats) for saved structure files.")
    parser.add_argument("structures", nargs="+", help="structure file(s) ASE can read (POSCAR, .vasp, .xyz, ...)")
    parser.add_argument("--out", default="stats", help="output directory (default: ./stats)")
    parser.add_argument("--saturation", action="store_true", help="also emit O-H / OH-per-cation plots")
    parser.add_argument("--no-pooled", action="store_true", help="skip the pooled report when given several files")
    args = parser.parse_args(argv)

    from ase.io import read
    from base.amorphous_structure import AmorphousStruc_factory
    from .report import write_report, write_pooled_report

    out_root = Path(args.out)
    all_metrics = []
    for path in args.structures:
        try:
            struct = AmorphousStruc_factory(atoms=read(path))
        except Exception as exc:
            print(f"error: cannot read {path}: {exc}", file=sys.stderr)
            return 2
        sub = out_root / Path(path).stem if len(args.structures) > 1 else out_root
        m = write_report(struct, sub, is_saturation=args.saturation, label=Path(path).name)
        all_metrics.append(m)
        print(f"[stats] wrote {sub}")

    if len(all_metrics) > 1 and not args.no_pooled:
        write_pooled_report(all_metrics, out_root, is_saturation=args.saturation)
        print(f"[stats] wrote pooled report to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
