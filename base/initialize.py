from typing import Optional
import os
from .amorphous_structure import AmorphousStruc, AmorphousStruc_factory


def initialize_structure_file(
        initial_file_name: Optional[os.PathLike],
        ase_read_kwargs: dict[str, any],
    ) -> AmorphousStruc:
    if initial_file_name is not None and os.path.exists(initial_file_name):
        from ase.io import read
        initial_atoms = read(initial_file_name, **ase_read_kwargs)
        return AmorphousStruc_factory(atoms=initial_atoms)
        
    return None


def initialize_structure_blank(
        cell: Optional[list[float]] = None,
        pbc: Optional[list[bool]] = None,
    ) -> AmorphousStruc:
    return AmorphousStruc_factory(cell=cell, pbc=pbc)