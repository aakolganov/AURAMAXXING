from mace.calculators import mace_mp
from typing import Optional
import os
from os import system as sys
from typing import Union
from ase.optimize import LBFGS, FIRE
from ase import Atoms
import torch
#import torch_dftd
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.units import fs, kB
import numpy as np
from helpers.files_io import add_dump_to_traj
from pathlib import Path

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


os.environ['PYTORCH_MPS_PREFER_FLOAT32'] = '1'

class MACEInterface:
    """
    Interface of geometry optimization with MACE.
    """
    def __init__(self, mace_model_path:str, device: str = "mps"):
        """
        Parameters
        ----------
        mace_model_path : str
            path to mace model file.
        device : str, optional
            which device to use, by default "mps" - apple silicon GPU.
        """
        if not os.path.exists(mace_model_path):
            raise FileNotFoundError(f"MACE model wasn't found: {mace_model_path}")
        self.mace_model_path = mace_model_path
        self.device = device
        self.calculator: Union[mace_mp, None] = None

        # loading calculator once
        self._load_mace_calculator()

    def _load_mace_calculator(self):
        """
        Loading MACE calculator.
        """
        model=torch.load(self.mace_model_path, map_location="cpu")
        model_path = self.mace_model_path
        model = model.to(torch.float32)
        new_model_path = model_path[:-6] + "float32.model"
        torch.save(model, new_model_path)

        self.calculator = mace_mp(
            model=new_model_path,
            #dispersion=True,
            device=self.device,            # GPU
            default_dtype="float32"  # float32
        )

    # def add_dump_to_trajectory(self, traj_file: Optional[str] = None):
    #     """Add MACE dump.xyz to the main trajectory file"""
    #     if traj_file and os.path.exists("dump.xyz"):
    #         dump_xyz_path = Path("dump.xyz")
    #         add_dump_to_traj(dump_xyz_path, traj_file)

    def optimize(self, atoms: Atoms, max_steps: int = 450) -> float:
        """
        Optimization of the structure using MACE.

        Parameters
        ----------
        atoms : ase.Atoms
           Structure to optimize
        max_steps : int, optional
           max iterations, by default 150.

        Returns
        -------
        energy : float
            electronic energy of the optimized structure.
        atoms: ase.Atoms
            optimized geometry
        """
        # 1) Appointing MACE calculator:
        atoms.set_calculator(self.calculator)

        # 2) Running L-BFGS optimizer:
        opt = LBFGS(atoms)  # logfile=None,
        opt.run(fmax=0.1, steps=max_steps)  # correct fmax if needed

        # 3) getting the energy of the optimized structure:
        try:
            energy = atoms.get_potential_energy()  # output MACE-calculated energy
        except Exception:
            energy = float('nan')

        return energy, atoms

    def anneal(self,
               atoms: Atoms,
               n_steps_heating: int,
               n_steps_cooling: int,
               start_T: float,
               final_T: float,
               timestep_fs: float = 1.25,
               traj_file: Optional[str] = None):
        """
        Annealing of the structure using MACE.
        """

        #0) Clear any existing dump file
        if os.path.exists("dump.xyz"):
            os.remove("dump.xyz")


        # 1) Appointing MACE calculator:
        atoms.set_calculator(self.calculator)


        #2) Heating phase: High temperature equilibration
        MaxwellBoltzmannDistribution(atoms, temperature_K=start_T)

        # Remove center of mass motion
        atoms.set_momenta(atoms.get_momenta() -
                          atoms.get_momenta().sum(axis=0) / len(atoms))

        # Set up MD integrator
        md = VelocityVerlet(atoms, timestep_fs*fs)

        temp_controller = TemperatureController(atoms, start_T, tau=50)

        for step in range(n_steps_heating):
            md.run(1)
            temp_controller.apply()

            #write dump file
            if step % 1  == 0 and step != 0:
                atoms.wrap()
                atoms.write( "dump.xyz", format="xyz", append=True, comment=f"Step_heating: {step}")

        # 3) Cooling phase:
        for step in range(n_steps_cooling):
            # Calculate target temperature for this step
            target_temp = linear_temperature_schedule(step, n_steps_cooling, T_initial=start_T, T_final= final_T)
            temp_controller.set_temperature(target_temp)

            md.run(1)
            if step%1 ==0 and step != 0:
                atoms.wrap()
                atoms.write( "dump.xyz", format="xyz", append=True, comment=f"Step_cooling: {step}")
            temp_controller.apply()

        return atoms

    def set_task(self,
                 atoms: Atoms,
                 type_opt: str = "minimize",
                 start_T: Optional[float] = None,
                 final_T: Optional[float] = None,
                 n_steps_heating: Optional[int] = None,
                 n_steps_cooling: Optional[int] = None
                 ) -> Atoms:  # Now consistently returns Atoms only
        if type_opt == "minimize":
            _, atoms = self.optimize(atoms, 250)
            return atoms  # Return just atoms

        elif type_opt == "anneal":
            assert isinstance(n_steps_heating, int) and isinstance(n_steps_cooling, int), (
                "For anneal, n_steps_heating and n_steps_cooling are needed"
            )
            assert start_T is not None and final_T is not None, (
                "For anneal, start_T and final_T are needed"
            )
            atoms = self.anneal(
                atoms=atoms,
                n_steps_heating=n_steps_heating,
                n_steps_cooling=n_steps_cooling,
                start_T=start_T,
                final_T=final_T
            )
            return atoms  # Return just atoms

        else:
            raise ValueError(f"Unknown opt type {type_opt!r}, allowed: 'minimize', 'anneal'.")


if __name__ == "__main__":

    atoms = read("../examples/Generation/Siral_10_generation/Siral_10/alpha_0_010/POSCAR_final_struc", format="vasp")

    model_path = "../examples/Saturation/Siral_10_saturation/2024-01-07-mace-128-L2_epoch-199.model"

    mace_if = MACEInterface(model_path, device="mps")

    new_atoms = mace_if.set_task(
        atoms=atoms,
        type_opt="anneal",
        n_steps_heating=1000,
        n_steps_cooling=1000,
        start_T=1000,
        final_T=298,
    )

    new_atoms.write("mace_fix_anneal_test.cif", format="cif")
