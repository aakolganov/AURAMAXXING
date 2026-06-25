"""Dispatch: write the requested DFT input sets for one structure.

``dft_spec`` is any object exposing ``packages`` (subset of {"vasp", "cp2k"}), ``vasp`` and
``cp2k`` option dicts -- the runner passes a ``DFTSpec``; the CLI passes the same.
"""
from __future__ import annotations

from pathlib import Path

from ase import Atoms

from .cp2k import write_cp2k_inputs
from .vasp import write_vasp_inputs


def write_dft_inputs(atoms: Atoms, dft_spec, out_dir) -> list[Path]:
    """Write a ``vasp/`` and/or ``cp2k/`` input directory under ``out_dir``."""
    out_dir = Path(out_dir)
    written: list[Path] = []
    for pkg in dft_spec.packages:
        if pkg == "vasp":
            written.append(write_vasp_inputs(atoms, dft_spec.vasp, out_dir / "vasp"))
        elif pkg == "cp2k":
            written.append(write_cp2k_inputs(atoms, dft_spec.cp2k, out_dir / "cp2k"))
    return written
