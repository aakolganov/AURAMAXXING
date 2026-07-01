import os
from pathlib import Path
from typing import Optional

from interfaces.base_interface import CalculatorInterface

# DFT functional behind each UMA task head, used only to parameterize an optional external
# D3 dispersion term. Keys are exactly the valid UMA task names (fairchem UMATask).
FUNCTIONAL = {
    "omol": "pbe",
    "oc25": "rpbe",
    "oc20": "rpbe",
    "odac": "pbe",
    "omat": "pbe",
    "omc": "pbe",
}

# UMA tasks whose underlying training data already accounts for dispersion, so adding an
# external D3 term on top would double-count it: omol is wB97M-V (VV10 dispersion native),
# and oc25/odac/omc are dispersion-corrected. omat/oc20 (plain PBE/RPBE) are not, so a D3
# term is a legitimate add-on there.
_NATIVE_DISPERSION_TASKS = {"omol", "oc25", "odac", "omc"}


def _validate_task(task: str) -> None:
    """Reject an unknown UMA task up front with a clear error (the FUNCTIONAL keys are the
    valid task names), instead of letting it surface later as a KeyError / opaque failure."""
    if task not in FUNCTIONAL:
        raise ValueError(f"unknown UMA task {task!r}; choose one of {sorted(FUNCTIONAL)}")


def _adds_external_dispersion(task: str, use_dispersion: bool) -> bool:
    """Whether to attach an external D3 term: only when dispersion is requested AND the task's
    training data does not already include it (otherwise the D3 would double-count)."""
    return bool(use_dispersion) and task not in _NATIVE_DISPERSION_TASKS


class UMAInterface(CalculatorInterface):
    """Geometry optimization / MD with a fairchem UMA model.

    ``uma_model_path`` is either a pretrained UMA model name that fairchem resolves and
    downloads itself -- e.g. ``"uma-s-1p2"`` (the current small UMA), ``"uma-m-1p1"`` -- or a
    path to a local checkpoint file. ``task`` selects the UMA task head (one of
    ``omol/omat/odac/oc20/oc25/omc``; ``omat`` is the materials head, the right default for
    oxide slabs). ``use_dispersion`` adds a torch-dftd D3 term, but only for tasks that do not
    already include dispersion natively.
    """

    def __init__(self, uma_model_path: str, uma_atom_ref: Optional[str] = None,
                 device: str = "cuda", task: str = "omat", use_dispersion: bool = False,
                 dump_path: str = "dump", **kwargs):
        _validate_task(task)

        try:
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
            from fairchem.core.units.mlip_unit import load_predict_unit
        except ModuleNotFoundError as exc:
            raise ImportError("Please install fairchem-core to use UMAInterface") from exc

        if self._is_local_path(uma_model_path):
            # Local checkpoint: load directly; atom_refs is optional and only read if given.
            atom_refs = None
            if uma_atom_ref is not None:
                from omegaconf import OmegaConf
                atom_refs = OmegaConf.load(uma_atom_ref)
            pred_unit = load_predict_unit(path=uma_model_path, device=device, atom_refs=atom_refs)
        else:
            # Pretrained model name (e.g. "uma-s-1p2"): fairchem downloads/caches it. The
            # checkpoint already carries its reference energies, so uma_atom_ref is unused.
            pred_unit = pretrained_mlip.get_predict_unit(uma_model_path, device=device)

        calc = FAIRChemCalculator(pred_unit, task_name=task)

        add_dispersion = _adds_external_dispersion(task, use_dispersion)
        if use_dispersion and not add_dispersion:
            print(f"UMAInterface: task {task!r} already accounts for dispersion; skipping the "
                  "external D3 correction to avoid double-counting.")
        if add_dispersion:
            try:
                import torch
                from ase import units
                from ase.calculators.mixing import SumCalculator
                from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
            except ModuleNotFoundError as exc:
                raise ImportError("Please install torch-dftd to use use_dispersion=True") from exc

            d3_calc = TorchDFTD3Calculator(
                device=device,
                damping=kwargs.pop("damping", "bj"),
                dtype=self._resolve_dtype(device, kwargs.pop("dtype", None)),
                xc=FUNCTIONAL[task],
                cutoff=40 * units.Bohr,
            )
            calc = SumCalculator([calc, d3_calc])

        self.calc = calc
        self.dump_path = Path(dump_path)
        self.dump_path.mkdir(parents=True, exist_ok=True)
