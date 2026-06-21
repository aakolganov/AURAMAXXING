"""Tests for the calculator interfaces (LAMMPS/BKS path)."""
import numpy as np
import pytest


# --- P1: the anneal must actually cool from T_ini down to T_fin ------------------
# Regression for the bug where LMPInterface.anneal held the target temperature
# constant at T_fin (cooling phase used T_initial=T_fin, T_final=T_fin), so the
# structure was heated and never quenched.

def test_lammps_anneal_schedule_cools(tmp_path):
    from interfaces.LAMMPS_Interface import LMPInterface

    calc = LMPInterface(dump_path=str(tmp_path / "d"))
    temps = calc._target_temperatures(steps=100, T_ini=2000.0, T_fin=300.0)

    assert len(temps) == 100
    assert temps[0] == pytest.approx(2000.0, abs=1e-6)          # starts hot
    assert temps[-1] < 600.0                                     # genuinely quenched
    # monotonically non-increasing (a real cooling ramp, not a constant hold)
    assert all(b <= a + 1e-9 for a, b in zip(temps, temps[1:]))


def test_lmp_interface_is_a_calculator_interface(tmp_path):
    # C3: LMPInterface should satisfy the CalculatorInterface contract.
    from interfaces.LAMMPS_Interface import LMPInterface
    from interfaces.base_interface import CalculatorInterface

    calc = LMPInterface(dump_path=str(tmp_path / "d"))
    assert isinstance(calc, CalculatorInterface)


def test_uma_interface_module_imports():
    # C2: UMA interface must import from the real interfaces package (it previously
    # imported a non-existent auramaxxing.interfaces.base_interface). Needs torch.
    pytest.importorskip("torch")
    from interfaces.UMA_interface import UMAInterface
    from interfaces.base_interface import CalculatorInterface

    assert issubclass(UMAInterface, CalculatorInterface)


@pytest.mark.lammps
def test_lammps_anneal_runs_end_to_end(make_struct):
    """Real BKS MD: anneal completes and leaves finite geometry/energy."""
    pytest.importorskip("lammps")
    from interfaces.LAMMPS_Interface import LMPInterface

    s = make_struct(
        ["Si", "O", "O", "Si", "O"],
        [[5, 5, 5], [6.6, 5, 5], [3.4, 5, 5], [8, 5, 5], [9.6, 5, 5]],
        cell=(15.0, 15.0, 15.0),
    )
    calc = LMPInterface(dump_path="dump_lmp_anneal_test")
    calc.anneal(s.atoms, T_ini=2000.0, T_fin=300.0, steps=20)

    assert np.all(np.isfinite(s.atoms.get_positions()))
    assert np.isfinite(s.atoms.get_potential_energy())


@pytest.mark.lammps
def test_lammps_optimize_runs(make_struct):
    pytest.importorskip("lammps")
    from interfaces.LAMMPS_Interface import LMPInterface

    s = make_struct(
        ["Si", "O", "O"],
        [[5, 5, 5], [6.6, 5, 5], [3.4, 5, 5]],
        cell=(15.0, 15.0, 15.0),
    )
    calc = LMPInterface(dump_path="dump_lmp_opt_test")
    calc.optimize(s.atoms, fmax=1.0, max_steps=10)
    assert np.isfinite(s.atoms.get_potential_energy())
