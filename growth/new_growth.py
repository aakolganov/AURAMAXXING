import os
from typing import Optional
from tqdm import tqdm
from pathlib import Path
import numpy as np

from helpers.atom_placing import place_atom_sphere, place_atom_most_z_space, slice_structure, build_placement_cache
from interfaces.base_interface import CalculatorInterface
from base import AmorphousStruc
from helpers.atom_picker import pick_next_atom_type, choose_atom_idx_to_attach_to
from base.limits import move_limits
from helpers.files_io import write_structure_to_file

DEFAULT_ANNEAL_PARAMS = {"T_ini": 2000, "T_fin": 300, "steps": 250, "interval": 10}


def grow_structure(
        amorphous_struct: AmorphousStruc,
        target_number_atoms: int,
        target_ratios: dict[str, float],
        calculator: Optional[CalculatorInterface] = None,
        max_placement_attempts: int = 1000,
        per_anchor_attempts: int = 100,
        num_samples: int = 100,
        anneal_params: Optional[dict] = None,
        output_dir: Path = Path("growth"),
        workdir: Optional[Path] = None,
        write_growth_dumps: bool = False,
        write_trajectories: bool = False,
    ):
    # ``workdir`` is this slab's own directory; anneal logs/trajectories go there so
    # concurrent runs never collide. ``write_growth_dumps``/``write_trajectories`` gate
    # the debug-only per-atom snapshots and MD trajectories (heavy I/O on large runs).
    if write_growth_dumps and not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)

    if calculator is None:
        raise ValueError("a calculator (CalculatorInterface) is required")

    # Melt-and-quench schedule used to relieve steric jams (override per call).
    if anneal_params is None:
        anneal_params = DEFAULT_ANNEAL_PARAMS

    # Bond lengths are drawn from scipy distributions (sample_dist), whose .rvs()
    # uses numpy's global RNG. Seed it from the structure's seeded generator so a
    # given set_seed() makes the whole growth reproducible.
    np.random.seed(int(amorphous_struct.rng.integers(2**31 - 1)))

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
            
            # Nothing is committed until a placement succeeds, so the structure is
            # fixed across these attempts: compute the per-atom coordination numbers
            # and the per-element collision trees once and reuse them for every
            # anchor we try, instead of recomputing them on each attempt.
            all_cn = amorphous_struct.get_cn()
            placement_cache = build_placement_cache(amorphous_struct)

            idx_connect_to = choose_atom_idx_to_attach_to(amorphous_struct, atom_to_add, all_cn=all_cn)

            current_iter = 0
            placement_success = False
            excluded_idx = []
            while current_iter <= per_anchor_attempts and not placement_success:
                current_iter += 1
                # No valid opposite-class anchor left (all excluded / none exist): stop trying
                # anchors and fall through to the melt-quench that relieves the jam.
                if idx_connect_to < 0:
                    break
                bond_length = amorphous_struct.sample_dist[amorphous_struct.atoms[idx_connect_to].symbol][atom_to_add]
                placement_success = place_atom_sphere(
                    amorphous_struct,
                    atom_to_add,
                    idx_connect_to,
                    bond_length,
                    num_samples=num_samples,
                    cache=placement_cache,
                    )
                if not placement_success:
                    excluded_idx.append(idx_connect_to)
                    # If placement fails, pick a different anchor
                    idx_connect_to = choose_atom_idx_to_attach_to(amorphous_struct, atom_to_add, exclude_indices=excluded_idx, all_cn=all_cn)  # Try different connection point if placement failed
            
            if placement_success:
                pbar.update(1)
                if write_growth_dumps:
                    write_structure_to_file(amorphous_struct, output_dir/f"dump_{num_placement_attempts}", write_xyz=True)

            else:
                # Melt-and-quench to relieve the steric jam: start hot, cool to ~RT.
                calculator.anneal(
                    amorphous_struct.atoms,
                    logfile="log.log",
                    traj_name="traj" if write_trajectories else None,
                    traj_fmt="xyz",
                    workdir=workdir,
                    **anneal_params,
                    )
                # The anneal MD moved every atom in place; invalidate the cached graph so
                # the slice below (when it removes nothing) and the next iteration's
                # get_cn() anchor selection both see the post-anneal geometry.
                amorphous_struct.invalidate_graph()

                slice_structure(amorphous_struct)
                pbar.n = len(amorphous_struct)
                pbar.refresh()
                move_limits(amorphous_struct)
    
    return len(amorphous_struct) == target_number_atoms



def finalize_structure(
    amorphous_struct: AmorphousStruc,
    calculator: Optional[CalculatorInterface] = None,
    fmax: float = 0.1,
    max_steps: int = 500,
    traj_interval: int = 1,
    workdir: Optional[Path] = None,
    write_trajectories: bool = False,
    ):
    if calculator is None:
        raise ValueError("a calculator (CalculatorInterface) is required")

    initial_num_atoms = len(amorphous_struct)
    calculator.optimize(
        amorphous_struct.atoms,
        fmax=fmax,
        max_steps=max_steps,
        traj_name="final_opt" if write_trajectories else None,
        traj_fmt="xyz",
        interval=traj_interval,
        workdir=workdir,
    )
    # The optimizer relaxed atoms.positions in place, which the cached coordination graph
    # cannot detect; mark it stale so the next get_cn()/get_graph() (e.g. the saturation
    # stage that runs right after finalize) rebuilds from the relaxed geometry.
    amorphous_struct.invalidate_graph()
    # Drop any leftover MD velocities (set by a growth-time anneal) so the finalized
    # structure is static and is written without a spurious velocity block.
    amorphous_struct.atoms.arrays.pop("momenta", None)

    return initial_num_atoms == len(amorphous_struct)


