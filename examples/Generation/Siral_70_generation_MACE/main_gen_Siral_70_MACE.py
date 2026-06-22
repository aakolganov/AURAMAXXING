"""
Grow Siral-70 (~2 Si : 1 Al) slabs of ~440 atoms at several surface roughnesses,
using a MACE machine-learning potential for the growth-time relaxation instead of
BKS/LAMMPS.

This is identical to main_gen_Siral_70.py except for the calculator: a MACEInterface
(point `mace_model_path` at your model) is passed to grow_structure / finalize.
"""
from pathlib import Path
import numpy as np

from base.initialize import initialize_structure_blank
from base.config import CoordinationConfig
from base.limits import make_limit_flat, make_limits_fourier, fix_limits
from growth.new_growth import grow_structure, finalize_structure
from interfaces.MACE_interface import MACEInterface
from default_constants import SIRAL_OVERCOORD

# Allow ~20% of Al to grow octahedral (CN up to 6); the rest stay tetrahedral (CN 4).
COORD_CONFIG = CoordinationConfig(overcoord_policy=SIRAL_OVERCOORD)

TARGET_RATIOS = {"Si": 4, "Al": 2, "O": 11}   # see main_gen_Siral_70.py for the stoichiometry
TARGET_ATOMS = 440
CELL = [22.0, 22.0, 40.0]
ALPHAS = np.logspace(-2, 0, 4)

MACE_MODEL_PATH = "path/to/your_mace_model.model"   # <-- set this
DEVICE = "cuda"                                     # "cuda" / "cpu" / "mps"


def generate(alpha: float, seed: int, out_dir: Path, calc) -> None:
    struct = initialize_structure_blank(cell=CELL, config=COORD_CONFIG)
    struct.set_seed(seed)

    make_limit_flat(struct, z_val=12.0, is_for="bottom")
    make_limits_fourier(struct, z_av=24.0, alpha=alpha, is_for="top")
    fix_limits(struct.limits, hard_limit="bottom")

    grow_structure(
        amorphous_struct=struct,
        target_number_atoms=TARGET_ATOMS,
        target_ratios=TARGET_RATIOS,
        calculator=calc,
        output_dir=out_dir / "growth",
    )
    finalize_structure(struct, calculator=calc)
    struct.atoms.write(str(out_dir / f"Siral_70_alpha_{alpha:.3f}.vasp"), format="vasp")


if __name__ == "__main__":
    # Build the MACE calculator once and reuse it for every roughness: the model is
    # large, so re-loading it per structure wastes time and leaks memory.
    calc = MACEInterface(
        mace_model_path=MACE_MODEL_PATH,
        device=DEVICE,
        dump_path="dump_mace",
    )
    base = Path("Siral_70_MACE")
    for i, alpha in enumerate(ALPHAS):
        generate(alpha=float(alpha), seed=i, out_dir=base / f"alpha_{alpha:.3f}", calc=calc)
