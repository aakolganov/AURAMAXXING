"""Interface internals: BKS command setup, anneal ramp, trajectory writing.

These use only ASE (no LAMMPS binary / no torch): the LAMMPS calculator object is
constructed but never run, and the anneal/optimize use the LJ-backed DummyInterface.
"""
import pytest
from ase import units


def _lmpcmds(calc):
    # ASE stores constructor kwargs in .parameters
    return calc.parameters["lmpcmds"]


def test_lmp_calculator_builds_bks_commands(make_struct, tmp_path):
    from interfaces.LAMMPS_Interface import LMPInterface

    s = make_struct(["Si", "O", "Al"], [[5, 5, 5], [6.6, 5, 5], [8.2, 5, 5]],
                    cell=(15.0, 15.0, 15.0))
    calc = LMPInterface(dump_path=str(tmp_path / "d"))._init_lmp_calculator(s.atoms)

    joined = "\n".join(_lmpcmds(calc))
    assert "buck/coul/long" in joined          # BKS buckingham
    assert "kspace_style ewald" in joined       # long-range Coulomb
    assert "lj/cut" in joined                   # lj overlay
    # a charge is set for each present element
    assert joined.count("set type") >= 3


def test_annealing_langevin_ramps_temperature_down(make_struct, dummy_calc):
    from interfaces.base_interface import AnnealingLangevin

    s = make_struct(["Si", "O", "O"], [[5, 5, 5], [6.6, 5, 5], [3.4, 5, 5]],
                    cell=(15.0, 15.0, 15.0))
    s.atoms.calc = dummy_calc.calc

    recorded = []
    dyn = AnnealingLangevin(s.atoms, timestep=1.0 * units.fs, T_ini=2000.0,
                            T_fin=300.0, total_steps=10, friction=0.01)
    real_set = dyn.set_temperature
    dyn.set_temperature = lambda temperature_K=None, **kw: (
        recorded.append(temperature_K), real_set(temperature_K=temperature_K, **kw))[1]

    dyn.run(10)

    assert recorded[0] == pytest.approx(2000.0, abs=1.0)        # starts hot
    assert recorded[-1] < 600.0                                  # ends cold
    assert all(b <= a + 1e-6 for a, b in zip(recorded, recorded[1:], strict=False))  # monotonic down


def test_attach_trajectory_writes_into_dump_path(make_struct, dummy_calc):
    s = make_struct(["Si", "O", "O"], [[5, 5, 5], [6.6, 5, 5], [3.4, 5, 5]],
                    cell=(15.0, 15.0, 15.0))
    dummy_calc.optimize(s.atoms, fmax=0.01, max_steps=5,
                        traj_name="opt", traj_fmt="xyz", interval=1)

    traj = dummy_calc.dump_path / "opt.xyz"
    assert traj.exists()
    frames = sum(1 for line in traj.read_text().splitlines() if line.strip() == "3")
    assert frames >= 1
