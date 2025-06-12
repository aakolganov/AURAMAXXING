from abc import ABC, abstractmethod
from ase import Atoms
from pathlib import Path
import os
from helpers.files_io import add_dump_to_traj
from interfaces.LAMMPS_Interface import LammpsInterface
from interfaces.MACE_interface import MACEInterface
from ase.constraints import FixAtoms

class BaseOptimizer(ABC):
    """Abstract base class for optimizers"""
    @abstractmethod
    def optimize(self, atoms: Atoms, opt_type: str, **kwargs) -> Atoms:
        pass
    @staticmethod
    def process_dump(traj_file: str, dump_path: Path):
        if dump_path.exists():
            add_dump_to_traj(dump_path, traj_file)
            os.remove(dump_path)

class LammpsOptimizer(BaseOptimizer):
    """Wrapper for LAMMPS optimizer/annealer"""
    def __init__(self, struc):
        self.optimizer = LammpsInterface(struc)
        self.dump_path = Path("LAMMPS") / "dump.xyz"

    def optimize(self, atoms: Atoms, opt_type: str, **kwargs) -> Atoms:
        frozen_indices = kwargs.get('frozen_indices', [])
        constraint = FixAtoms(indices=frozen_indices) #prefixing the atoms
        self.optimizer.atoms = atoms
        self.optimizer.atoms.set_constraint(constraint)
        if opt_type == "anneal":
            return self.optimizer.opt_struc(
                "anneal",
                steps=kwargs['steps'],
                start_T=kwargs['start_T'],
                final_T=kwargs['final_T'],
                FF="BKS"
            )
        elif opt_type == "final":
            return self.optimizer.opt_struc(
                "final",
                steps=kwargs['steps'],
                start_T=kwargs['start_T'],
                final_T=kwargs['final_T'],
                FF="BKS"
            )


class MACEOptimizer(BaseOptimizer):
    "Wrapper for MACE optimizer/annealer"
    def __init__(self, model_path: str):
        self.optimizer = MACEInterface(model_path, device="mps")
        self.dump_path = Path("dump.xyz")

    def optimize(self, atoms: Atoms, opt_type: str, **kwargs) -> Atoms:
        if opt_type == "anneal":
            return self.optimizer.set_task(
                atoms=atoms,
                type_opt="anneal",
                n_steps_heating=kwargs.get('n_steps_heating', 1000),
                n_steps_cooling=kwargs.get('n_steps_cooling', 1000),
                start_T=kwargs['start_T'],
                final_T=kwargs['final_T']
            )
        elif opt_type == "minimize":
            _, minimized = self.optimizer.optimize(atoms, max_steps=500)
            return minimized