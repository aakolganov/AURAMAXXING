"""
Grow Siral-70 (silica-rich silica/alumina, ~2 Si : 1 Al) slabs of ~440 atoms at
several surface roughnesses, using the BKS/LAMMPS calculator.

Replaces the old parallel `main_generation_routine`; the current API grows one
structure at a time, so we loop over the roughness parameter `alpha`.
"""
from pathlib import Path
import numpy as np

from base.initialize import initialize_structure_blank
from base.limits import make_limit_flat, make_limits_fourier, fix_limits
from growth.new_growth import grow_structure, finalize_structure
from interfaces.LAMMPS_Interface import LMPInterface

# 2 Si : 1 Al, with O for a charge-neutral oxide: 2*Si(+4) + 1*Al(+3) = +11 -> 5.5 O.
# Scaled to integers: Si:4, Al:2, O:11 (target_ratios are relative weights).
TARGET_RATIOS = {"Si": 4, "Al": 2, "O": 11}
TARGET_ATOMS = 440
CELL = [22.0, 22.0, 40.0]
ALPHAS = np.logspace(-2, 0, 4)


def generate(alpha: float, seed: int, out_dir: Path) -> None:
    struct = initialize_structure_blank(cell=CELL)
    struct.set_seed(seed)

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
    struct.atoms.write(str(out_dir / f"Siral_70_alpha_{alpha:.3f}.vasp"), format="vasp")


if __name__ == "__main__":
    base = Path("Siral_70")
    for i, alpha in enumerate(ALPHAS):
        generate(alpha=float(alpha), seed=i, out_dir=base / f"alpha_{alpha:.3f}")
