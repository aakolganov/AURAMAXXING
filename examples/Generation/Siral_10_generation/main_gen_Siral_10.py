"""
Grow a Siral-10 (alumina-rich, ~1 Si : 4.5 Al) film on top of a bare
gamma-Al2O3(110) surface read from a POSCAR, using the BKS/LAMMPS calculator.

The growth limits are placed *above* the loaded surface so new atoms are added on
top of it. `target_number_atoms` counts the starting-surface atoms too, so growth
stops once the slab reaches that total. Run from this directory so the POSCAR is
found.
"""
from pathlib import Path
import numpy as np

from base.initialize import initialize_structure_file
from base.limits import make_limit_flat, make_limits_fourier, fix_limits
from growth.new_growth import grow_structure, finalize_structure
from interfaces.LAMMPS_Interface import LMPInterface

# 1 Si : 4.5 Al, with O for a charge-neutral oxide: (10*4 + 45*3)/2 = 87.5 O.
# Scaled to integers: Si:20, Al:90, O:175 (target_ratios are relative weights).
TARGET_RATIOS = {"Si": 20, "Al": 90, "O": 175}
TARGET_ATOMS = 305                 # total atoms incl. the starting surface
GROWTH_THICKNESS = 12.0            # how far above the surface to grow (Angstrom)
ALPHAS = np.logspace(-2, 0, 4)


def generate(alpha: float, seed: int, out_dir: Path) -> None:
    struct = initialize_structure_file("POSCAR_bare_gAl_110", ase_read_kwargs={})
    struct.set_seed(seed)

    # Anchor the growth region just above the existing surface.
    surface_top = float(struct.atoms.get_positions()[:, 2].max())
    make_limit_flat(struct, z_val=surface_top, is_for="bottom")
    make_limits_fourier(struct, z_av=surface_top + GROWTH_THICKNESS, alpha=alpha, is_for="top")
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
    struct.atoms.write(str(out_dir / f"Siral_10_alpha_{alpha:.3f}.vasp"), format="vasp")


if __name__ == "__main__":
    base = Path("Siral_10")
    for i, alpha in enumerate(ALPHAS):
        generate(alpha=float(alpha), seed=i, out_dir=base / f"alpha_{alpha:.3f}")
