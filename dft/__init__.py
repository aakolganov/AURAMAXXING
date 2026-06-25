"""DFT-refinement input generation (pipeline step 3).

Turn a generated amorphous-oxide structure into a ready-to-run geometry-optimization input
set for VASP and/or CP2K (PBE-D3BJ, fixed cell, spin non-polarised). py4vasp is intentionally
not used here -- it only reads VASP output; ASE writes the structures.
"""
from .api import write_dft_inputs
from .cp2k import DEFAULT_CP2K_BASIS
from .vasp import DEFAULT_VASP_INCAR, DEFAULT_VASP_KPOINTS

__all__ = [
    "write_dft_inputs",
    "DEFAULT_VASP_INCAR",
    "DEFAULT_VASP_KPOINTS",
    "DEFAULT_CP2K_BASIS",
]
