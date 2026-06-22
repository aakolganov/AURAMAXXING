"""
Grow pure amorphous silica (SiO2) slabs of ~432 atoms at several surface
roughnesses, using the BKS/LAMMPS calculator for growth-time relaxation.

The old high-level `main_generation_routine` (parallel, multi-structure) has been
replaced by the lower-level `grow_structure`. The current API builds one structure
at a time, so here we simply loop over the roughness parameter `alpha`
(smaller alpha -> rougher top surface).
"""
from pathlib import Path
import numpy as np

from base.initialize import initialize_structure_blank
from base.limits import make_limit_flat, make_limits_fourier, fix_limits
from growth.new_growth import grow_structure, finalize_structure
from interfaces.LAMMPS_Interface import LMPInterface

# SiO2 stoichiometry is 1 Si : 2 O (target_ratios are relative weights).
TARGET_RATIOS = {"Si": 1, "O": 2}
TARGET_ATOMS = 432
CELL = [22.0, 22.0, 40.0]            # in-plane x, y; z holds the slab + vacuum
ALPHAS = np.logspace(-2, 0, 4)        # roughness of the Fourier top surface


def generate(alpha: float, seed: int, out_dir: Path) -> None:
    struct = initialize_structure_blank(cell=CELL)
    struct.set_seed(seed)

    # Grow between a flat bottom and a rough (Fourier) top surface.
    make_limit_flat(struct, z_val=12.0, is_for="bottom")
    make_limits_fourier(struct, z_av=24.0, alpha=alpha, is_for="top")
    fix_limits(struct.limits, hard_limit="bottom")

    calc = LMPInterface(dump_path=str(out_dir / "dump_lmp"))
    grow_structure(
        amorphous_struct=struct,
        target_number_atoms=TARGET_ATOMS,
        target_ratios=TARGET_RATIOS,
        calculator=calc,
        output_dir=out_dir / "growth",
    )
    finalize_structure(struct, calculator=calc)
    struct.atoms.write(str(out_dir / f"SiO2_alpha_{alpha:.3f}.vasp"), format="vasp")


if __name__ == "__main__":
    base = Path("Pure_Silica")
    for i, alpha in enumerate(ALPHAS):
        generate(alpha=float(alpha), seed=i, out_dir=base / f"alpha_{alpha:.3f}")
