import os
from pathlib import Path
from auramaxxing.interfaces.base_interface import CalculatorInterface
import torch

match_dtype = {
    "float32": torch.float32,
    "float64": torch.float64,
}

FUNCTIONAL = {
    "omol": "pbe",
    "oc25": "rpbe",
    "oc20": "rpbe",
    "odac": "pbe",
    "omat": "pbe",
    "omc": "pbe",
}

class UMAInterface(CalculatorInterface):

    def __init__(self, uma_model_path: str, 
                 uma_atom_ref: str = None, device: str = "cuda",
                 task: str = "default", use_dispersion: bool = False,
                 dump_path: str = "dump",
                 **kwargs):
        
        try:
            from omegaconf import OmegaConf
            from fairchem.core.units.mlip_unit import load_predict_unit
            from fairchem.core import FAIRChemCalculator
        except ModuleNotFoundError:
            print("Please install fairchem-core")
        
        atom_refs = OmegaConf.load(uma_atom_ref)
        pred_unit = load_predict_unit(path=uma_model_path, device=device, atom_refs=atom_refs)
        calc = FAIRChemCalculator(predict_unit=pred_unit, task_name=task)

        if use_dispersion and task == "omol":
            print("The task of OMol does not require dispersion. wB97M-V uses VV10 corrections natively.")
            print("Fall back of PBE will be used.")

        if use_dispersion and task in ["oc25", "odac", "omc"]:
            print(f"The task {task} already uses dispersion corrections")
            print(f"Fall back of {FUNCTIONAL[task]} will be used.")

        if use_dispersion:
            try:
                from ase.calculators.mixing import SumCalculator
                from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
                from ase import units
            except ModuleNotFoundError:
                print("Please install torch_dftd")
            
            d3_calc = TorchDFTD3Calculator(
                device=device,
                damping=kwargs.pop("damping", "bj"),
                dtype=match_dtype[kwargs.pop("dtype", "float32")],
                xc=FUNCTIONAL[task],
                cutoff= 40 * units.Bohr
            )
            calc = SumCalculator([calc, d3_calc])

        self.calc = calc

        self.dump_path = Path(dump_path)
        if not self.dump_path.exists():
            os.makedirs(dump_path, exist_ok=True)