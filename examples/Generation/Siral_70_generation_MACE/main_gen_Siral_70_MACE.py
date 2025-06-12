"""
routine to generate Sirol-70 structures of with different roughness using MACE as calculator
"""

from growth.parallel_growth import main_generation_routine
import numpy as np

base_output_dir = "Siral_70"
total_structures = 4
alphas = np.logspace(-2, 0, total_structures)
timeout_seconds = 10 * 60
max_workers = 4

if __name__ == '__main__':
    main_generation_routine(
        base_output_dir=base_output_dir,
        alphas=alphas,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
        total_desired_atoms=440,
        n_m=4,
        calculator="mace",
        starting_struc=None,
        target_ratio={"Si":2, "Al":1},
        traj_filename="growth_trajectory.xyz",
        final_struc_template="Siral_70_alpha_{alpha}.cif",
    )
