"""
routine to generate 16 pure silica structures of 432 atoms with different roughness in parallel
"""

from growth.parallel_growth import main_generation_routine
import numpy as np

base_output_dir = "Pure_Silica"
total_structures = 8
alphas = np.logspace(-2, 0, total_structures)  # 0.01 → 100max_workers = 4
timeout_seconds = 10 * 60
max_workers = 8

if __name__ == '__main__':
    main_generation_routine(
        base_output_dir=base_output_dir,
        alphas=alphas,
        max_workers=max_workers,
        starting_struc=None,
        timeout_seconds=timeout_seconds,
        total_desired_atoms=432,
        n_m=3,
        target_ratio={"Si":1, "Al":0},
        traj_filename="growth_trajectory.xyz",
        final_struc_template="SiO2_alpha_{alpha}.cif"
    )
