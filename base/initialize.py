from typing import Optional
import os

import numpy as np

from .amorphous_structure import AmorphousStruc, AmorphousStruc_factory
from .config import CoordinationConfig


def _assert_orthogonal_cell(struct: AmorphousStruc, source: str) -> AmorphousStruc:
    """Growth's limit grid and periodic wrapping assume an orthogonal (diagonal) cell -- a
    tilted cell would silently produce a wrong grid and out-of-box placements. Reject it with
    a clear error instead."""
    cell = np.array(struct.atoms.cell)
    if cell.shape == (3, 3):
        off = cell - np.diag(np.diagonal(cell))
        if np.any(np.abs(off) > 1e-6):
            raise ValueError(
                f"{source}: the cell is non-orthogonal (tilted); growth assumes an orthogonal "
                f"box (zero off-diagonal cell entries). Off-diagonal entries:\n{off}")
    return struct


def initialize_structure_file(
        initial_file_name: Optional[os.PathLike],
        ase_read_kwargs: dict[str, any],
        config: Optional[CoordinationConfig] = None,
    ) -> Optional[AmorphousStruc]:
    if initial_file_name is None:
        return None
    if not os.path.exists(initial_file_name):
        raise FileNotFoundError(
            f"initialize_structure_file: structure file not found: {initial_file_name}")
    from ase.io import read
    initial_atoms = read(initial_file_name, **ase_read_kwargs)
    struct = AmorphousStruc_factory(atoms=initial_atoms, config=config)
    return _assert_orthogonal_cell(struct, f"from_file {str(initial_file_name)!r}")


def initialize_structure_blank(
        cell: Optional[list[float]] = None,
        pbc: Optional[list[bool]] = None,
        config: Optional[CoordinationConfig] = None,
    ) -> AmorphousStruc:
    struct = AmorphousStruc_factory(cell=cell, pbc=pbc, config=config)
    return _assert_orthogonal_cell(struct, "structure.cell")
