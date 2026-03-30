import os
from pathlib import Path
from interfaces.base_interface import CalculatorInterface

os.environ["PYTORCH_MPS_PREFER_FLOAT32"] = "1"

class MACEInterface(CalculatorInterface):
    """
    Interface of geometry optimization with MACE.
    """
    def __init__(self, mace_model_path:str, device: str = "mps",
                 head: str = "default", use_dispersion: bool = False,
                 dump_path: str = "dump",
                 **kwargs):
        """
        Parameters
        ----------
        mace_model_path : str
            path to mace model file.
        device : str, optional
            which device to use, by default "mps" - apple silicon GPU.
        """
        try:
            from mace.calculators import mace_mp
        except ModuleNotFoundError:
            print("Please install mace-torch")

        if not os.path.exists(mace_model_path):
            raise FileNotFoundError(f"MACE model wasn't found: {mace_model_path}")
        
        self.calc = mace_mp(
            model=mace_model_path,
            head=head,
            device=device,
            default_dtype=kwargs.pop("dtype", "float32"),
            dispersion=use_dispersion,
            damping=kwargs.pop("damping", "bj"),
            dispersion_xc=kwargs.pop("dispersion_xc", "pbe"),
        )
        self.dump_path = Path(dump_path)
        if not self.dump_path.exists():
            os.makedirs(dump_path, exist_ok=True)
