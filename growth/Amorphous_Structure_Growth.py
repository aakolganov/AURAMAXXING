import os
import time
from ase import Atoms
from tqdm import tqdm
from base.AmorphousStrucASE import AmorphousStrucASE
from interfaces.LAMMPS_Interface import LammpsInterface
from ase.constraints import FixAtoms
from helpers.atom_picker import pick_next_atom
from helpers.files_io import save_traj, add_LAMMPS_dump_to_traj
from typing import Optional, Dict, Callable
from ase.io import read
from pathlib import Path

# Default ratio for testing growth on top

TEST_TARGET_RATIO = {
    "Si": 10,
    "Al": 45,
}



#init_struc_test = read("../POSCAR_bare_gAl_110_2x2x2")

def generate_amorphous_structure(total_desired_atoms: int,
                                 alpha:float,
                                 n_m:int,
                                 starting_struc:Optional[Atoms]=None,
                                 TARGET_RATIO:Dict[str, int]=TEST_TARGET_RATIO,
                                 traj_file:Optional[str]="growth_trajectory.xyz",
                                 final_struc_file:Optional[str]="final_struc.cif",
                                 progress_cb: Optional[Callable[[int], None]] = None,
                                 demo_mode: Optional[bool] = False
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
    :return AmorphousStrucASE: the final grown structure (ASE wrapper) containing exactly total_desired_atoms atoms.
    """

    # 1. Remove any existing trajectory or old LAMMPS output
    if os.path.exists(traj_file):
        os.remove(traj_file)
    if os.path.exists("../Silica_Alumina/LAMMPS/final_struc.data"):
        os.remove("../Silica_Alumina/LAMMPS/final_struc.data")

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

    # 3. Progress bar for demo mode:
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
            if is_existing_structure:
                max_steps, start_T, final_T = 1000, 298, 800
            else:
                max_steps, start_T, final_T = 1000, 298, 4000

            FF = "BKS" #BKS force field by default
            struc_optimizer = LammpsInterface(amorph_struc)
            if frozen_indices:
                constraint = FixAtoms(indices=frozen_indices)
                struc_optimizer.atoms.set_constraint(constraint)
            else:
                # even if there are none - we still need to provide empty object to LAMMPS interface
                struc_optimizer.atoms.set_constraint(FixAtoms(indices=[]))
            # Run the LAMMPS anneal, update the ASE Atoms from the output
            amorph_struc.atoms = struc_optimizer.opt_struc(
                "anneal",
                steps=max_steps,
                start_T=start_T,
                final_T=final_T,
                FF=FF
            )

            dump_xyz_path = Path("LAMMPS") / "dump.xyz"

            add_LAMMPS_dump_to_traj(dump_xyz_path, traj_file)

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
        struc_optimizer = LammpsInterface(amorph_struc)
        if frozen_indices:
            constraint = FixAtoms(indices=frozen_indices)
            struc_optimizer.atoms.set_constraint(constraint)
        else:
            # even if there are none - we still need to provide empty object to LAMMPS interface
            struc_optimizer.atoms.set_constraint(FixAtoms(indices=[]))

        if is_existing_structure:
            # If we started from an existing structure, do a quick anneal + final minimize
            _ = struc_optimizer.opt_struc("anneal", steps=500, start_T=298, final_T=2000, FF="BKS")
            add_LAMMPS_dump_to_traj(dump_xyz_path, traj_file)
            constraint = FixAtoms(indices=frozen_indices)
            struc_optimizer.atoms.set_constraint(constraint)
            new_atoms = struc_optimizer.opt_struc(
                "final",
                steps=1000,
                start_T=298,
                final_T=298,
                FF="BKS"
            )
            add_LAMMPS_dump_to_traj(dump_xyz_path, traj_file)
            amorph_struc.update_atoms(new_atoms)
            #if demo_mode:
                #view(amorph_struc.atoms)
        else:
            # If this was a “fresh” growth, do two-stage annealing
            _ = struc_optimizer.opt_struc("anneal", steps=250, start_T=298, final_T=2000, FF="BKS")
            add_LAMMPS_dump_to_traj(dump_xyz_path, traj_file)
            if frozen_indices:
                constraint = FixAtoms(indices=frozen_indices)
                struc_optimizer.atoms.set_constraint(constraint)
            else:
                # even if there are none - we still need to provide empty object to LAMMPS interface
                struc_optimizer.atoms.set_constraint(FixAtoms(indices=[]))
            amorph_struc.atoms = struc_optimizer.opt_struc(
                "anneal",
                steps=750,
                start_T=2000,
                final_T=298,
                FF="BKS"
            )
            amorph_struc.update_atoms(amorph_struc.atoms)
            add_LAMMPS_dump_to_traj(dump_xyz_path, traj_file)
            #if demo_mode:
                #view(amorph_struc.atoms)


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


#if __name__ == "__main__":
 #   generate_amorphous_structure(total_desired_atoms=270,alpha= 100, n_m= 4, starting_struc= init_struc_test, demo_mode=True)
