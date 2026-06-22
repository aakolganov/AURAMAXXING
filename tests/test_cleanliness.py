"""Output-cleanliness fixes: clean LAMMPS exit env, and velocity-free finalized
geometry."""
import numpy as np


def test_lammps_interface_sets_fi_provider():
    import os
    import interfaces.LAMMPS_Interface  # noqa: F401  (import sets FI_PROVIDER)

    # conda-forge LAMMPS aborts at MPI_Finalize on macOS unless a safe OFI provider
    # is selected; importing the interface sets one (default "tcp").
    assert os.environ.get("FI_PROVIDER")


def test_finalize_strips_velocities(make_struct, dummy_calc):
    from growth.new_growth import finalize_structure

    s = make_struct(["Si", "O", "O"], [[5, 5, 5], [6.6, 5, 5], [3.4, 5, 5]],
                    cell=(15.0, 15.0, 15.0))
    s.atoms.set_momenta(np.ones((len(s), 3)))   # simulate leftover MD velocities
    assert s.atoms.get_velocities() is not None

    finalize_structure(s, calculator=dummy_calc)

    # the finalized structure is static -> no momenta -> no velocity block on write
    assert "momenta" not in s.atoms.arrays
