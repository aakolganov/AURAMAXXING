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

        mace_kwargs = dict(
            head=head,
            device=device,
            default_dtype=self._resolve_dtype(device, kwargs.pop("dtype", None)),
            dispersion=use_dispersion,
            damping=kwargs.pop("damping", "bj"),
            dispersion_xc=kwargs.pop("dispersion_xc", "pbe"),
        )
        self.calc = self._load_mace_mp(mace_mp, mace_model_path, mace_kwargs)
        self.dump_path = Path(dump_path)
        if not self.dump_path.exists():
            os.makedirs(dump_path, exist_ok=True)

    @staticmethod
    def _resolve_dtype(device: str, explicit_dtype):
        """Default precision: float64 on CPU/CUDA (more accurate, recommended for
        geometry optimisation), float32 on MPS (Apple Silicon has no float64). An
        explicitly passed ``dtype`` always wins.
        """
        if explicit_dtype is not None:
            return explicit_dtype
        return "float32" if device == "mps" else "float64"

    @staticmethod
    def _load_mace_mp(mace_mp, model_path: str, mace_kwargs: dict):
        """Build the MACE calculator, with a float32-safe fallback for MPS.

        The Apple-Silicon (MPS) backend cannot ``torch.load`` float64 weights, which
        is how many MACE checkpoints are stored. If the direct load hits that, we
        re-load the checkpoint on CPU, cast it to float32, and retry on the target
        device.
        """
        try:
            return mace_mp(model=model_path, **mace_kwargs)
        except TypeError as exc:
            if "float64" not in str(exc) and "MPS" not in str(exc):
                raise
            import tempfile
            import torch

            cpu_model = torch.load(model_path, map_location="cpu", weights_only=False)
            f32_path = Path(tempfile.gettempdir()) / (Path(model_path).stem + "_float32.model")
            torch.save(cpu_model.float(), f32_path)
            return mace_mp(model=str(f32_path), **mace_kwargs)
