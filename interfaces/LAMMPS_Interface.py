from typing import Optional, Any
import os
from ase.calculators.lammpslib import LAMMPSlib
from ase import Atoms
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.units import fs, kB
import numpy as np
from pathlib import Path

from interfaces.base_interface import CalculatorInterface
from default_constants import (
    bks_params,
    bks_lj,
    bks_charges,
    default_masses,
    ev_to_kcal,
)

# conda-forge LAMMPS is built with MPI; on macOS its libfabric/OFI provider can
# fail while probing a network interface at MPI_Finalize, aborting the process on
# exit (exit code 143) even though the calculation finished. The TCP provider uses
# loopback and tears down cleanly. setdefault so an explicit user setting wins.
os.environ.setdefault("FI_PROVIDER", "tcp")

# helper functions to setup annealing procedure


def linear_temperature_schedule(step, total_steps, T_initial, T_final):
    """Linear temperature decrease from T_initial to T_final"""
    return T_initial - (T_initial - T_final) * step / total_steps


class TemperatureController:
    """Temperature controller using velocity rescaling"""

    def __init__(self, atoms, target_temp, tau=50):
        self.atoms = atoms
        self.target_temp = target_temp
        self.tau = tau  # coupling strength (lower = stronger coupling)

    def set_temperature(self, temp):
        """Set target temperature"""
        self.target_temp = temp

    def apply(self):
        """Apply Berendsen-like temperature coupling"""
        if len(self.atoms) == 0:
            return
        
        # Calculate current temperature
        ekin = self.atoms.get_kinetic_energy()
        current_temp = 2 * ekin / (3 * kB * len(self.atoms))

        if current_temp > 0:
            # Berendsen thermostat scaling factor
            scale_factor = np.sqrt(1 + (self.target_temp / current_temp - 1) / self.tau)

            # Scale velocities
            velocities = self.atoms.get_velocities()
            self.atoms.set_velocities(velocities * scale_factor)


class LMPInterface(CalculatorInterface):
    """
    Interface of geometry optimization with LAMMPS.
    """

    def __init__(self, dump_path: str = "dump"):
        """
        Parameters
        ----------
        None needed for initialization
        """
        self.dump_path = Path(dump_path)
        if not self.dump_path.exists():
            os.makedirs(self.dump_path)
        # Cache the built LAMMPS calculator, invalidated when the structure's element
        # set or atom count changes (see _init_lmp_calculator).
        self._cached_calc = None
        self._cached_key = None

    def _init_lmp_calculator(self, atoms: Atoms) -> LAMMPSlib:
        """
        Helper function for the initialization of the LAMMPS simulation. Currently only supports BKS
        parameters for Al, Si, and O.

        Parameters
        ----------
        atoms : ase.Atoms
           Structure to optimize

        Returns
        -------
        lmp_calc : LAMMPSlib
           Initialized LAMMPS calculator
        """
        # Reuse the cached calculator when the element set and atom count are
        # unchanged (e.g. the repeated finalize/optimize calls on one structure).
        # We key on atom count too because a reused LAMMPSlib instance is only safe
        # for a fixed system size, so growth (which adds atoms) always rebuilds.
        cache_key = (tuple(sorted(set(atoms.get_chemical_symbols()))), len(atoms))
        if self._cached_calc is not None and self._cached_key == cache_key:
            return self._cached_calc

        unique_atom_types = np.unique(atoms.get_chemical_symbols())
        atom_types = {atom_type: i+1 for i, atom_type in enumerate(unique_atom_types)}
        atom_type_masses = {atom_type: default_masses[atom_type] for atom_type in atom_types.keys()}
        lmp_cmds = [
            "pair_style hybrid/overlay buck/coul/long 5.5 8.0 lj/cut 1.2",
            "kspace_style ewald 1.0e-4",
        ]

        # Parse the coefficeitns for the buckingham portion of the CMD
        for (a, b), (A_eV, rho_inv, C_eV) in bks_params.items():
            if a in atom_types and b in atom_types:
                i, j = atom_types[a], atom_types[b]
                A = A_eV * ev_to_kcal
                rho = 1/rho_inv
                C = C_eV * ev_to_kcal
                lmp_cmds.append(
                    f"pair_coeff {i:>2d} {j:>2d} buck/coul/long {A:.6g} {rho:.6g} {C:.6g}"
                )
        # Parse the coefficients for the lj portion of the CMD
        for (a, b), (eps_eV, sigma, cutoff) in bks_lj.items():
            if a in atom_types and b in atom_types:
                i, j = atom_types[a], atom_types[b]
                eps = eps_eV * ev_to_kcal
                lmp_cmds.append(
                    f"pair_coeff {i:>2d} {j:>2d} lj/cut {eps:.6g} {sigma:.6g} {cutoff:.6g} # {a}-{b}"
                )
        # Parse the charges for the CMD
        for a, q in bks_charges.items():
            if a in atom_types:
                t = atom_types[a]
                lmp_cmds.append(f"set type {t} charge {q:.6g}   # {a}")

        lammps_header = [
            "units real",
            "atom_style charge",
            "atom_modify map array sort 0 0",
        ]
        lmp_calc = LAMMPSlib(
            lammps_header=lammps_header,
            lmpcmds=lmp_cmds,
            atom_types=atom_types,
            atom_type_masses=atom_type_masses,
        )
        self._cached_calc = lmp_calc
        self._cached_key = cache_key
        return lmp_calc


    def optimize(self, atoms: Atoms, fmax: float = 2.0, max_steps: int = 50,
                 logfile: str = "log.log", traj_name: Optional[str] = None,
                 traj_fmt: Optional[str] = None,
                 frozen_atoms: Optional[list] = None,
                 **kwargs: Any) -> Atoms:
        """
        Optimize the geometry with the BKS/LAMMPS potential. The LAMMPS calculator
        is rebuilt for this structure (atom_types depend on which elements are
        present), then the shared LBFGS/trajectory machinery of the base class runs.
        """
        self.calc = self._init_lmp_calculator(atoms)
        if frozen_atoms is not None:
            atoms.set_constraint(FixAtoms(indices=frozen_atoms))
        return super().optimize(
            atoms, fmax=fmax, max_steps=max_steps, logfile=logfile,
            traj_name=traj_name, traj_fmt=traj_fmt, **kwargs,
        )


    def _target_temperatures(self, steps: int, T_ini: float, T_fin: float) -> list:
        """Per-step thermostat target temperatures for a monotonic anneal ramp.

        Annealing means starting hot and quenching, so call with T_ini > T_fin.
        Returns one target per MD step, linearly interpolated from T_ini to T_fin.
        """
        return [linear_temperature_schedule(step, steps, T_ini, T_fin) for step in range(steps)]

    def anneal(self, atoms: Atoms, T_ini: float, T_fin: float,
            steps: int = 500, dt: float = 1.0,
            logfile: str = "log.log", traj_name: Optional[str] = None, traj_fmt: Optional[str] = None,
            **kwargs: Any) -> Atoms:
        """
        Anneal the structure with the BKS/LAMMPS potential: initialise velocities at
        T_ini and ramp the thermostat target linearly down to T_fin over `steps` MD
        steps. Use T_ini > T_fin for a melt-and-quench.
        """
        print("Starting Anneal")
        traj_interval = kwargs.pop('interval', 1)

        atoms.calc = self._init_lmp_calculator(atoms)

        # Initialise velocities at the (hot) starting temperature; remove net COM drift
        MaxwellBoltzmannDistribution(atoms, temperature_K=T_ini)
        atoms.set_momenta(
            atoms.get_momenta() - atoms.get_momenta().sum(axis=0) / len(atoms)
        )

        md = VelocityVerlet(atoms, dt * fs)
        if traj_name:
            self._attach_trajectory(md, atoms, traj_name, traj_fmt, interval=traj_interval)

        # Single monotonic schedule hot -> cold. The previous implementation held the
        # target at T_fin for the whole "cooling" phase, so the structure was heated
        # and never actually quenched.
        controller = TemperatureController(atoms, T_ini, tau=50)
        for target_temp in self._target_temperatures(steps, T_ini, T_fin):
            controller.set_temperature(target_temp)
            md.run(1)
            controller.apply()

        return atoms
