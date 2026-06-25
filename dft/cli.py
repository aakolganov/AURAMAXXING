"""Standalone DFT-input generator: turn already-generated structure files into VASP/CP2K
geometry-optimization inputs without re-running the growth pipeline.

    auramaxxing-dft PATH [--package {vasp,cp2k,both}] [--config CONFIG.yaml] [--out DIR] [--slurm]

PATH is a single structure file (POSCAR/.vasp/.xyz/...) or a directory; for a directory we
process every ``structure.vasp`` found beneath it (falling back to top-level ``*.vasp``/
``*.xyz``). With ``--config`` we read the ``dft:`` block (tag overrides) from a run config;
otherwise the built-in defaults are used. ``--slurm`` also drops a ready-to-edit SLURM array
submission script per package covering all processed structures.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ase.io import read as ase_read

from dft.api import write_dft_inputs
from runner.config import DFTSpec, load_config


def _resolve_spec(config_path, packages) -> DFTSpec:
    spec = load_config(config_path).dft if config_path is not None else DFTSpec()
    spec.enabled = True
    spec.packages = packages
    return spec


def _iter_structures(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.rglob("structure.vasp"))
        if not files:
            files = sorted(path.glob("*.vasp")) + sorted(path.glob("*.xyz"))
        return files
    return [path]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="auramaxxing-dft",
        description="Generate VASP/CP2K DFT-refinement inputs from existing structure files.",
    )
    parser.add_argument("path", help="a structure file or a directory of structure.vasp files")
    parser.add_argument("--package", choices=["vasp", "cp2k", "both"], default="both",
                        help="which DFT inputs to write (default: both)")
    parser.add_argument("--config", default=None, metavar="YAML",
                        help="optional run config to read the dft: tag overrides from")
    parser.add_argument("--out", default=None, metavar="DIR",
                        help="output dir (default: a 'dft' dir beside each structure)")
    parser.add_argument("--slurm", action="store_true",
                        help="also write a ready-to-edit SLURM array submission script per package")
    args = parser.parse_args(argv)

    packages = ["vasp", "cp2k"] if args.package == "both" else [args.package]
    try:
        spec = _resolve_spec(args.config, packages)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    path = Path(args.path)
    structures = _iter_structures(path)
    if not structures:
        print(f"error: no structure files found at {path}", file=sys.stderr)
        return 2

    dft_dirs = []
    for struct_file in structures:
        atoms = ase_read(str(struct_file))
        if args.out is not None:
            out_dir = Path(args.out) / struct_file.stem if len(structures) > 1 else Path(args.out)
        else:
            out_dir = struct_file.parent / "dft"
        write_dft_inputs(atoms, spec, out_dir)
        dft_dirs.append(out_dir)
        print(f"[auramaxxing-dft] wrote {', '.join(packages)} inputs -> {out_dir}")

    if args.slurm:
        from dft.slurm import write_slurm_scripts
        slurm_root = Path(args.out) if args.out else (path if path.is_dir() else path.parent)
        project = spec.cp2k.get("project", "amorphous_oxide")
        for script in write_slurm_scripts(slurm_root, dft_dirs, packages, cp2k_project=project):
            print(f"[auramaxxing-dft] wrote SLURM array script -> {script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
