"""Shared pytest fixtures for AURAMAXXING.

All structure-building goes through seeded RNGs so tests are deterministic.
Heavy calculators are never imported here; use the ``dummy_calc`` fixture instead.
"""
import numpy as np
import pytest

from base.amorphous_structure import AmorphousStruc_factory


@pytest.fixture
def make_struct():
    """Factory: build a seeded AmorphousStruc from explicit symbols + positions.

    Usage:
        s = make_struct(["Si", "O"], [[10, 10, 10], [11.6, 10, 10]])
    """
    def _make(symbols, positions, cell=(20.0, 20.0, 20.0), pbc=True, seed=42):
        return AmorphousStruc_factory(
            symbols=list(symbols),
            positions=np.asarray(positions, dtype=float),
            cell=list(cell),
            pbc=pbc,
            seed=seed,
        )
    return _make


@pytest.fixture
def blank_struct():
    """An empty, seeded, periodic structure in a 20x20x20 cell."""
    from base.initialize import initialize_structure_blank
    s = initialize_structure_blank(cell=[20.0, 20.0, 20.0])
    s.set_seed(42)
    return s


@pytest.fixture
def dummy_calc(tmp_path):
    """A DummyInterface (LJ-backed) writing into a temp dir. No LAMMPS/MACE needed."""
    from tests.dummy_interface import DummyInterface
    return DummyInterface(dump_path=str(tmp_path / "dump_dummy"))
