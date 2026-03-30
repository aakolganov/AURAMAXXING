import os
from typing import Optional
from tqdm import tqdm
from pathlib import Path

from default_constants import sample_dist
from helpers.atom_placing import place_atom_sphere, place_atom_most_z_space, slice_structure
from interfaces.base_interface import CalculatorInterface
from base import AmorphousStruc
from helpers.atom_picker import pick_next_atom_type, choose_atom_idx_to_attach_to
from base.limits import move_limits
from helpers.files_io import write_structure_to_file

def grow_structure(
        amorphous_struct: AmorphousStruc,
        target_number_atoms: int,
        target_ratios: dict[str, float],
        calculator: Optional[CalculatorInterface] = None,
        max_placement_attempts: int = 1000,
        output_dir: Path = Path("growth")
    ):
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)

    if calculator is None:
        calculator = ... # Choose some reasonable default later

    num_placement_attempts = 0
    with tqdm(total=target_number_atoms, initial=len(amorphous_struct)) as pbar:
        while len(amorphous_struct) < target_number_atoms and num_placement_attempts < max_placement_attempts:
            num_placement_attempts += 1
            current_number_atoms = len(amorphous_struct)

            atom_to_add = pick_next_atom_type(amorphous_struct, target_ratios)
            if current_number_atoms == 0:
                place_atom_most_z_space(amorphous_struct, atom_to_add)
                pbar.update(1)
                continue
            
            idx_connect_to = choose_atom_idx_to_attach_to(amorphous_struct, atom_to_add)

            MAX_ITER, current_iter = 100, 0
            placement_success = False
            excluded_idx = []
            while current_iter <= MAX_ITER and not placement_success:
                current_iter += 1
                bond_length = sample_dist[amorphous_struct.atoms[idx_connect_to].symbol][atom_to_add]
                placement_success = place_atom_sphere(
                    amorphous_struct,
                    atom_to_add,
                    idx_connect_to,
                    bond_length,
                    )
                if not placement_success:
                    excluded_idx.append(idx_connect_to)
                    # If placement fails, pick a different anchor
                    idx_connect_to = choose_atom_idx_to_attach_to(amorphous_struct, atom_to_add, exclude_indices=excluded_idx)  # Try different connection point if placement failed
            
            if placement_success:
                pbar.update(1)
                write_structure_to_file(amorphous_struct, output_dir/f"dump_{num_placement_attempts}", write_xyz=True)

            else:
                calculator.anneal(
                    amorphous_struct.atoms, 
                    T_ini=1000,
                    T_fin=2000,
                    steps=250,
                    interval=10,
                    logfile="log.log",
                    traj_name="traj",
                    traj_fmt="xyz",
                    )

                slice_structure(amorphous_struct)
                pbar.n = len(amorphous_struct)
                pbar.refresh()
                move_limits(amorphous_struct)
    
    return len(amorphous_struct) == target_number_atoms



def finalize_structure(
    amorphous_struct: AmorphousStruc,
    calculator: Optional[CalculatorInterface] = None
    ):
    if calculator is None:
        calculator = ... # Choose some reasonable default later
    
    initial_num_atoms = len(amorphous_struct)
    # calculator.anneal(
    #     amorphous_struct.atoms, 
    #     T_ini=2000,
    #     T_fin=300,
    #     steps=5000,
    #     interval=100,
    #     traj_name="final_md",
    #     traj_fmt="xyz",
    #     )
    calculator.optimize(
        amorphous_struct.atoms,
        fmax=0.1,
        max_steps=500,
        traj_name="final_opt",
        traj_fmt="xyz",
        interval=1
    )
    # move_limits(amorphous_struct, move_limit="both")
    # slice_structure(amorphous_struct)
    
    return initial_num_atoms == len(amorphous_struct)


