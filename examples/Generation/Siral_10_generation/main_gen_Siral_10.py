"""
routine to generate 16 Siral-10 starting from gamma-Al2O3(110) surface
"""

from growth.parallel_growth import main_generation_routine
import numpy as np
from ase.io import read
starting_struc = read("POSCAR_bare_gAl_110")

base_output_dir = "Siral_10"
total_structures = 8
alphas = np.logspace(-2, 0, total_structures)  
timeout_seconds = 10 * 60
max_workers = 8

if __name__ == '__main__':
    main_generation_routine(
        base_output_dir=base_output_dir,
        alphas=alphas,
        starting_struc=starting_struc,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
        total_desired_atoms=135,
        n_m=3,
        target_ratio={"Si":10, "Al":45},
        traj_filename="growth_trajectory.xyz",
        final_struc_template="Siral_10_alpha_{alpha}.cif"
    )
