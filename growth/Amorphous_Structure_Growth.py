import os
import time
from ase import Atoms
from tqdm import tqdm
from base.AmorphousStrucASE import AmorphousStrucASE
from interfaces.Optimizer_Interface import LammpsOptimizer, MACEOptimizer
from ase.constraints import FixAtoms
from helpers.atom_picker import pick_next_atom
from helpers.files_io import save_traj, add_dump_to_traj
from typing import Optional, Dict, Callable, Literal
from ase.io import read
from pathlib import Path

TEST_TARGET_RATIO = {
    "Si": 2,
    "Al": 1,
}


def generate_amorphous_structure(total_desired_atoms: int,
                                 alpha:float,
                                 n_m:int,
                                 starting_struc:Optional[Atoms]=None,
                                 TARGET_RATIO:Dict[str, int]=TEST_TARGET_RATIO,
                                 traj_file:Optional[str]="growth_trajectory.xyz",
                                 final_struc_file:Optional[str]="final_struc.cif",
                                 progress_cb: Optional[Callable[[int], None]] = None,
                                 demo_mode: Optional[bool] = False,
                                 calculator: Literal["lammps", "mace"] = "lammps",
                                 mace_model_path: Optional[str] = None
                                 ) -> AmorphousStrucASE:
    """
    Grow a mixed Si/Al amorphous oxide slab by sequentially placing atoms until the desired number of atoms is reached.

    At each step, the code:
          1. Decides which atom to add (Si, O, or Al) based on charge neutrality and a target ratio.
          2. Chooses an existing atom to bond the new atom to (via `set_i`).
          3. Attempts random placement around that anchor, respecting slab limits, which are set by a periodic Fourier function.
          4. If placement fails repeatedly, calls LAMMPS to anneal/optimize the structure, then retries.
          5. Slice atoms after Lammps optimization to ensure that the slab is within the limits

    Parameters
    ____________
    :param int total_desired_atoms:
            Total number of atoms (Si + O + Al) to grow in the final crystal.
    :param float alpha:  Attenuation factor for the Fourier‐based surface (controls roughness).
    :param int n_m: Maximum Fourier mde along each axis (controls surface frequency content).
    :param Optional[Atoms] starting_struc: starting structure, if empty - starts from scratch
    :param Dict[str, int], default=TEST_TARGET_RATIO TARGET_RATIO: Desired stoichiometric ratio (e.g., {"Si": 7, "Al": 1}) for selecting cations.
    :param Optional[str], default="growth_trajectory.xyz" traj_file:  Path to the XYZ file to record the growth trajectory.
    :param Optional[str], default="final_struc.cif" final_struc_file: Path to the CIF file to record the final structure.
    :param Optional[Callable[[int], None]] = None, default=False progress_cb: Whether to display a progress bar during growth.
    :param Optional[bool] = False, default=False demo_mode: Whether to run in demo mode (for testing purposes)
    :param Literal["lammps", "mace"] calculator: which calculator to use for annealing
    :param Optional[str] = None, default=None mace_model_path: path to the MACE (foundational) model file
    :return AmorphousStrucASE: the final grown structure (ASE wrapper) containing exactly total_desired_atoms atoms.
    """

    # 1. Remove any existing trajectory or old LAMMPS output
    if os.path.exists(traj_file):
        os.remove(traj_file)
    if os.path.exists("LAMMPS/final_struc.data"):
        os.remove("LAMMPS/final_struc.data")

    # 2. Initialize the AmorphousStrucASE object
    if starting_struc is None:
        # Start from an empty structure, set cell and periodicity
        amorph_struc = AmorphousStrucASE(atoms=None)
        amorph_struc.atoms.set_cell([21.5, 21.5, 50.0]) #default cell parameters; 3x3 beta-cristobalite unit cell
        amorph_struc.atoms.set_pbc([True, True, True])
        # Define rough surface limits using a Fourier function
        amorph_struc.set_limits(alpha=alpha, n_m=n_m) #, const_V=True, H=3)
        # indicates that we are starting from scratch
        is_existing_structure = False
        frozen_indices = [] #no atoms are frozen
    else:
        # Copy an existing structure to grow on top of it
        base = starting_struc.copy()
        amorph_struc = AmorphousStrucASE(atoms=base)
        amorph_struc.freeze_bottom_half() #fix the bottom layers for further manipulations
        frozen_indices = amorph_struc.frozen_indices
        #indicates that we are starting from the initial structure-more gentle anneal will be applied
        is_existing_structure = True #at the moment we're growing the structure on top of some other oxides,
        amorph_struc.set_limits(alpha=alpha, n_m = n_m)
        #later we implement growing an AROUND certain pattern

    # Counters and limits
    number_write = 0
    TOTAL_DESIRED_ATOM, new_total_atoms = total_desired_atoms, 0
    max_placement_attempts = 500  # bail out if too many failed placement attempts
    placement_attempts = 0

    # Initialazing the calculator:
    if calculator == "mace":
        if not mace_model_path:
            raise ValueError("MACE model path required for MACE calculator")
        optimizer = MACEOptimizer(mace_model_path)
    else:
        optimizer = LammpsOptimizer(AmorphousStrucASE(atoms=amorph_struc.atoms.copy()))

    # 3.5 (Optional). Progress bar for demo mode:
    if demo_mode:
        pbar = tqdm(total=TOTAL_DESIRED_ATOM, desc=f"Growing atoms for α={alpha:.3f}", leave=True)

    # 4. Main growth loop: add one atom at a time
    while new_total_atoms < TOTAL_DESIRED_ATOM and placement_attempts < max_placement_attempts:
        placement_attempts += 1
        current_number_atoms = len(amorph_struc.atoms)
        # 4a. If structure is empty, place a single Si atom at the center of a box
        if current_number_atoms == 0:
            atom_to_add = "Si"
            made_placement = amorph_struc.place_atom_random(atom_to_add)

            if made_placement:
                number_write += 1
                save_traj(amorph_struc, "growth_trajectory.xyz", number_write)
                new_total_atoms = len(amorph_struc.atoms)
                placement_attempts = 0  # Reset attempts counter after successful placement
                if progress_cb:
                    progress_cb(1)
                if demo_mode:
                    pbar.update(1)
                    #view(amorph_struc.atoms)
                    #time.sleep(0.5)
            continue

        # 4b. Decide which atom to add next:
        #     - If no Al in target, just alternate O/Si to enforce SiO2 stoichiometry.
        #     - Otherwise use pick_next_atom(...) to maintain both charge neutrality and TARGET_RATIO.

        if TARGET_RATIO["Al"] == 0:
            n_Si = amorph_struc.count_atoms("Si")
            n_O = amorph_struc.count_atoms("O")

        # enforce O < 2*Si → add O until you hit 2:1, else add Si
            atom_to_add = "O" if (2 * n_Si > n_O) else "Si"
        else:
            # Mixed Si/Al oxide: choose based on oxidation states + desired ratio
            atom_to_add = pick_next_atom(amorph_struc, TARGET_RATIO)

        # 4c. Pick an existing atom to which we'll bond the new atom
        idx_connect_to = amorph_struc.set_i(atom_to_add)


        # 4d. Try random placement up to MAX_ITER times
        MAX_ITER, current_iter = 100, 0
        made_placement = False
        while current_iter <= MAX_ITER and not made_placement:
            current_iter += 1
            made_placement = amorph_struc.place_atom_random(
                atom_to_add,
                idx_anchor=idx_connect_to,
                limits=amorph_struc.limits,
                max_iter=50   # per‐attempt limit for random sampling
            )

            if not made_placement:
                # If placement fails, pick a different anchor
                idx_connect_to = amorph_struc.set_i(atom_to_add)  # Try different connection point if placement failed

        # 4e. If placement succeeded, update counters, save to trajectory, and reset attempts
        if made_placement:
            new_total_atoms = len(amorph_struc.atoms)
            number_write += 1
            save_traj(amorph_struc, "growth_trajectory.xyz", number_write)
            placement_attempts = 0  # Reset attempts counter after successful placement
            if progress_cb:
                progress_cb(1)
            if demo_mode:
                pbar.update(1)
                #time.sleep(0.5)
                #view(amorph_struc.atoms)

        # 4f. If repeated placement attempts all failed, run a LAMMPS anneal/optimize cycle
        else:
            #pbar.write("⚠️  Attempting structure optimization…")
            # Choose anneal parameters depending on whether we have the existing structrue
            new_atoms = optimizer.optimize(
                atoms=amorph_struc.atoms.copy(),
                opt_type="anneal",
                frozen_indices=frozen_indices,
                steps=1000,
                start_T=1000 if is_existing_structure else 4000,
                final_T=298,
                n_steps_heating=1000,
                n_steps_cooling=1000
            )

            # Обновление структуры
            amorph_struc.atoms = new_atoms
            optimizer.process_dump(traj_file, optimizer.dump_path)

            # If still growing (not final), apply a “slice” to remove atoms above the surface
            prev_count = len(amorph_struc.atoms)
            amorph_struc.slice(raise_by=1.0)
            # Once we delete atoms, recompute how many remain
            new_count = len(amorph_struc.atoms)
            delta_atoms = prev_count - new_count
            if delta_atoms > 0 and progress_cb:
                progress_cb(-delta_atoms)
            if demo_mode:  # Sync tqdm bar with the new actual atom count
                #view(amorph_struc.atoms)
                pbar.n=new_total_atoms
                pbar.refresh()
                #time.sleep(0.5)


            # Save an intermediate snapshot after optimization and resume growth
            number_write += 1
            save_traj(amorph_struc, "growth_trajectory.xyz", number_write)
            if progress_cb:
                progress_cb(1)
            if demo_mode:
                #view(amorph_struc.atoms)
                time.sleep(0.5)
                pbar.update(1)
                pbar.write("  ✅  Optimization done, resuming growth")
    if demo_mode:
        pbar.close()

# 5. Once we've reached exactly TOTAL_DESIRED_ATOM, perform a final optimization
    if new_total_atoms == TOTAL_DESIRED_ATOM:
        if demo_mode:
            pbar.write("  ✅  Perfoming final optimization...")
        if is_existing_structure:
            new_atoms = optimizer.optimize(
                atoms=amorph_struc.atoms.copy(),
                opt_type="anneal",
                frozen_indices=frozen_indices,
                steps=500,
                start_T=298,
                final_T=2000
            )
            if calculator == "mace":
                new_atoms = optimizer.optimize(new_atoms, opt_type="minimize")
        else:
            new_atoms = optimizer.optimize(
                atoms=amorph_struc.atoms.copy(),
                opt_type="anneal",
                frozen_indices=frozen_indices,
                steps=750,
                start_T=2000,
                final_T=298
            )

        amorph_struc.atoms = new_atoms
        optimizer.process_dump(traj_file, optimizer.dump_path)

    # 6. Write out the very final structure (XYZ + CIF+VASP(if w/ base structure))
    number_write += 1
    save_traj(amorph_struc, "growth_trajectory.xyz", number_write)
    amorph_struc.atoms.write("growth_trajectory.xyz", format="xyz", append=True, comment="final structure")
    amorph_struc.atoms.write(final_struc_file, format="cif", append=False)
    amorph_struc.atoms.write(final_struc_file[:-3]+"xyz", format="xyz", append=False)
    if is_existing_structure: #writing POSCAR file to easily track fixed atoms
        amorph_struc.atoms.write("POSCAR_"+final_struc_file[:-4], format="vasp", append=False)

    if demo_mode:
        pbar.close()
        tqdm.write("  ✅  Final structure written to final_struc.cif")
        #view(amorph_struc.atoms)
    #check coordination
    #for n, atom in enumerate (amorph_struc.atoms):
    #    print (amorph_struc.get_cn(n))
    return amorph_struc


if __name__ == "__main__":
    # Default ratio for testing growth on top

    init_struc_test = read("../examples/Generation/Siral_10_generation/POSCAR_bare_gAl_110")

    generate_amorphous_structure(total_desired_atoms=320,alpha= 0.05, n_m= 3,
                                 #starting_struc= init_struc_test,
                                 demo_mode=True, calculator="lammps"
                                 )
                                 #mace_model_path="../examples/Saturation/Siral_10_saturation/2024-01-07-mace-128-L2_epoch-199.model")
