"""Standalone stats: ``python -m stats structure.vasp [--saturation] [--out DIR]``.

Analyses one or more saved structure files. Fixed atoms are recovered from a POSCAR's
Selective Dynamics (ASE reads them into a FixAtoms constraint); the coordination graph is
rebuilt with covalent-radii cutoffs derived for the file's actual elements (so e.g. F/Cl/Na
caps are bonded and counted, not just the default Si/Al/O/H). ``--saturation`` enables the
cap-distance / caps-per-cation plots (it cannot be inferred from a structure file).
"""
import argparse
import sys
from pathlib import Path


def _coordination_for(atoms):
    """A CoordinationConfig covering the file's elements (curated CN/oxidation + covalent-radii
    cutoffs), so a re-read F/Cl/Na cap is bonded and counted -- the module-default tables only
    know Si/Al/O/H. Unknown elements become spectators (distance rows, no curated CN). Returns
    ``None`` (use the default tables) when no element is curated. Note: an alkali cap with a
    multivalent curated CN (e.g. Na, max_cn 6) is not flagged terminal on bare re-read -- the
    in-pipeline run, which registers caps at CN 1, counts it correctly."""
    from base.config import CoordinationConfig
    from base.element_data import build_element_tables, OXIDE_ELEMENTS
    elems = sorted(set(atoms.get_chemical_symbols()))
    curated = [e for e in elems if e in OXIDE_ELEMENTS]
    if not curated:
        return None
    spectators = [e for e in elems if e not in OXIDE_ELEMENTS]
    t = build_element_tables(curated, spectator_elements=spectators)
    return CoordinationConfig(max_cn=t["max_cn"], min_cn=t["min_cn"], cut_offs=t["cut_offs"],
                              sample_dist=t["sample_dist"], d_min_max=t["d_min_max"],
                              oxidation=t["oxidation"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stats",
        description="Plot structural statistics (coordination, homo-element & element-anion "
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
            atoms = read(path)
            struct = AmorphousStruc_factory(atoms=atoms, config=_coordination_for(atoms))
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
