from typing import Optional, Any
import os
from os import system as sys
from typing import Union
from ase.optimize import LBFGS, FIRE2
from ase.calculators.lammpslib import LAMMPSlib
from ase import Atoms
from ase.constraints import FixAtoms

from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.units import fs, kB
import numpy as np
from helpers.files_io import add_dump_to_traj
from pathlib import Path

from default_constants import (
    bks_params,
    bks_lj,
    bks_charges,
    default_masses,
    ev_to_kcal,
)

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


class LMPInterface:
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
        return lmp_calc


    def _attach_trajectory(self, run, atoms: Atoms, 
                           filename: str, fmt: str = "xyz",
                           interval: int = 1):
        """
        Attaches a modular trajectory writer to an optimizer or MD engine.
        Handles both native ASE .traj files and appended text formats (like .xyz).
        """
        if not Path(self.dump_path).exists():
            os.makedirs(self.dump_path)

        full_filename = f"{filename}.{fmt}"
        filepath = self.dump_path / full_filename
        if Path(filepath).exists():
            os.remove(filepath)
        
        def write_frame():
            write(filepath, atoms, append=True, format=fmt)
        run.attach(write_frame, interval=interval)


    # def optimize(self, atoms: Atoms, max_steps: int = 450, frozen_atoms: Optional[list[int]]=None) -> float:
    #     """
    #     Optimization of the structure using MACE.

    #     Parameters
    #     ----------
    #     atoms : ase.Atoms
    #        Structure to optimize
    #     max_steps : int, optional
    #        max iterations, by default 150.

    #     Returns
    #     -------
    #     energy : float
    #         electronic energy of the optimized structure.
    #     atoms: ase.Atoms
    #         optimized geometry
    #     """
    #     calc = self._init_lmp_calculator(atoms)

    #     # 1) Appointing MACE calculator:
    #     atoms.calc = calc
    #     if frozen_atoms is not None:
    #         atoms.set_constraint(FixAtoms(indices=frozen_atoms))

    #     # 2) Running L-BFGS optimizer:
    #     opt = LBFGS(atoms, logfile=None)  # logfile=None,
    #     opt.run(fmax=0.1, steps=max_steps)  # correct fmax if needed

    #     # 3) getting the energy of the optimized structure:
    #     try:
    #         energy = atoms.get_potential_energy()  # output MACE-calculated energy
    #     except Exception:
    #         energy = float("nan")

    #     return energy, atoms
    
    def optimize(self, atoms: Atoms, fmax: float = 2.0, max_steps: int = 50,
                 logfile: str = "log.log", traj_name: str | None = None, traj_fmt: str | None = None,
                 frozen_atoms: Optional[list[int]]=None,
                 **kwargs: Any) -> Atoms:
        """
        Optimize the geometry of the structure using BFGS.
        """
        print("Starting Optimization")
        traj_interval = kwargs.pop('interval', 1)

        atoms.calc = self._init_lmp_calculator(atoms)
        if frozen_atoms is not None:
            atoms.set_constraint(FixAtoms(indices=frozen_atoms))

        opt = LBFGS(atoms, logfile=self.dump_path/logfile, **kwargs)
        if traj_name:
            self._attach_trajectory(opt, atoms, traj_name, traj_fmt, interval=traj_interval)

        opt.run(fmax=fmax, steps=max_steps)
        return atoms
    

    def anneal(self, atoms: Atoms, T_ini: float, T_fin: float, 
            steps: int = 500, dt: float = 1.0, 
            logfile: str = "log.log", traj_name: str | None = None, traj_fmt: str | None = None,
            **kwargs: Any) -> Atoms:
        """
        Annealing of the structure using MACE.
        """
        print("Starting Anneal")

        # 0) Clear any existing dump file
        if os.path.exists("dump.xyz"):
            os.remove("dump.xyz")

        # 1) Appointing MACE calculator:
        calc = self._init_lmp_calculator(atoms)
        atoms.calc = calc

        # 2) Heating phase: High temperature equilibration
        MaxwellBoltzmannDistribution(atoms, temperature_K=T_ini)

        # Remove center of mass motion
        atoms.set_momenta(
            atoms.get_momenta() - atoms.get_momenta().sum(axis=0) / len(atoms)
        )

        # Set up MD integrator
        md = VelocityVerlet(atoms, dt * fs)

        temp_controller = TemperatureController(atoms, T_fin, tau=50)

        for step in range(steps):
            md.run(1)
            temp_controller.apply()

            # write dump file
            if step % 10 == 0 and step != 0:
                atoms.wrap()
                atoms.write(
                    "dump.xyz",
                    format="xyz",
                    append=True,
                    comment=f"Step_heating: {step}",
                )
            
        # 3) Cooling phase:
        for step in range(steps):
            # Calculate target temperature for this step
            target_temp = linear_temperature_schedule(
                step, steps, T_initial=T_fin, T_final=T_fin
            )
            temp_controller.set_temperature(target_temp)

            md.run(1)
            if step % 1 == 0 and step != 0:
                atoms.wrap()
                atoms.write(
                    "dump.xyz",
                    format="xyz",
                    append=True,
                    comment=f"Step_cooling: {step}",
                )
            temp_controller.apply()
            
        return atoms
