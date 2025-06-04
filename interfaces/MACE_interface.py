from mace.calculators import mace_mp
import os
from os import system as sys
from typing import Union
from ase.optimize import LBFGS, FIRE
from ase import Atoms
import torch
#import torch_dftd
from ase.io import read

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

if __name__ == "__main__":
    #quick test#
    device = torch.device("mps")
    torch.set_default_dtype(torch.float32)

    model_path = "../MACE-models/MACE-matpes-r2scan-omat-ft.model"
    model=torch.load(model_path, map_location="cpu")
    model=model.to(torch.float32)
    new_model_path = model_path[:-6]+"float32.model"
    torch.save(model, new_model_path)

    macemp = mace_mp(new_model_path, device="mps", default_dtype="float32")  # downlaod the model at the given url

    # Set up a crystal
    atoms = read("../growth/POSCAR_final_struc", format="vasp")
    atoms.calc = macemp

    dyn = LBFGS(atoms)
    dyn.run(fmax=0.1)
    atoms.write("mace_fix_opt_test.cif", format="cif")
