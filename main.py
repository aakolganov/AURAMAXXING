from MakeCrystal import generate_crystal
from crystal_class import CrystalStruc
from LAMMPS_interface import Lammps_interface

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Set
from glob import glob


def read_xyz(file_name):
    atoms = np.empty(0)
    coordinates = np.array([]).reshape((0, 3))
    with open(file_name, "r") as f:
        lines = f.readlines()

    for line in lines[2:]:
        split_line = line.strip().split()
        assert len(split_line) == 4, ("Not a .xyz file, try again")
        coords = np.array([float(split_line[1]), float(split_line[2]), float(split_line[3])])

        atoms = np.append(atoms, str(split_line[0]))
        coordinates = np.vstack([coordinates, coords])
    return atoms, coordinates

def find_idx_too_under_atoms(crystal_struc: CrystalStruc, atoms_remove: Dict[str, int]) -> List[int]:
    """
    Finds all of the indicies of the atoms which are too under-coordinated for us to want

    --- input ---
      crystal_struc: the structure in the form of the CrystalStruc class
      atoms_remove: a dictionary with the atom types that we want to remove as the key and the cooridnation numbers
      we want to remove as the values

    --- output ---
      A list of the indicies of the atoms which we would want to delete
    """
    to_remove = []
    for idx, atom in enumerate(crystal_struc.atoms):
        if atom in atoms_remove:
            cn = crystal_struc.give_cn(idx)
            if cn in atoms_remove[atom]:
                to_remove.append(idx)
    return to_remove


def determine_silanol_types(crystal_struc: CrystalStruc, injected_atoms: List[int] = None) -> Tuple[set, set, set]:
    """
    Determines the silanol types of the strucutre before anything is being protonated

    --- input ---
      crystal_struc: the structure in the form of the CrystalStruc class
      injected_atoms (optional): atoms to add to the list to check against

    --- output ---
      Three sets of integers signifying the the indicies of the geminal, vicinal, and isolated silanol groups
    """
    # make the initial list of all under-coordinated atoms
    saturatable_atoms = {}
    for atom_type in crystal_struc.atom_types:
        saturatable_atoms[atom_type] = []
    
    CN_dict = crystal_struc.give_all_cn(CN_IdxAtom=True)
    for atom_type in CN_dict.keys():
        for cn, idx_atoms in CN_dict[atom_type].items():
            if cn < crystal_struc.max_cn[atom_type]:
                for _ in range(crystal_struc.max_cn[atom_type] - cn):
                    saturatable_atoms[atom_type].append(idx_atoms)
    
    for atom_type, list_idx_atoms in saturatable_atoms.items():
        saturatable_atoms[atom_type] = np.array([idx for sublist in list_idx_atoms for idx in sublist], dtype=int)
    
    Si_of_O = {}
    for idx_O in saturatable_atoms["O"]:
        neighbors = crystal_struc.give_nl(idx_O)
        Si_of_O[idx_O] = [i for i in neighbors]

    all_Si = np.append(np.array(saturatable_atoms["Si"]), np.array([val for sublist in Si_of_O.values() for val in sublist]))
    if injected_atoms is not None:
        all_Si = np.append(all_Si, injected_atoms)

    # find number of geminal silanol
    idx_Si_geminal = set()
    unique_Si, counts_Si = np.unique(all_Si, return_counts=True)

    for idx_Si, count_Si in zip(unique_Si, counts_Si):
        assert count_Si <= 2, f"got three or more OH on that Si. {idx_Si}"
        if count_Si == 2:
            idx_Si_geminal.add(idx_Si)

    # find the number of vicinal
    idx_Si_vicinal = set()
    
    # make a dictionary of the O atoms which are bound to silanol Si
    idx_bound_O = {}
    for idx_Si in unique_Si:
        assert crystal_struc.atoms[idx_Si] == "Si", f"not an Si atom. {idx_Si}"
        for idx_neighbor_O in crystal_struc.give_nl(idx_Si):
            if idx_neighbor_O not in idx_bound_O:
                idx_bound_O[idx_neighbor_O] = []
            idx_bound_O[idx_neighbor_O].append(idx_Si)
    
    # If one of those O atoms is bound to two Si which will be silanol then we have a vicinal
    for list_idx_Si in idx_bound_O.values():
        if len(list_idx_Si) > 1:
            [idx_Si_vicinal.add(idx_Si) for idx_Si in list_idx_Si]
    
    # check for which atoms are in both lists and remove the duplicate from geminal
    idx_Si_geminal.difference_update(idx_Si_vicinal)

    # The remaining must be isolated
    idx_Si_isolated = set()
    [idx_Si_isolated.add(i) for i in all_Si if i not in idx_Si_geminal and i not in idx_Si_vicinal]

    return idx_Si_geminal, idx_Si_vicinal, idx_Si_isolated

def add_O_on_3Si(crystal_struc: CrystalStruc):
    for idx, atom in enumerate(crystal_struc.atoms):
        if atom == "Si" and crystal_struc.give_cn(idx) <= 3:
            crystal_struc.place_new_atom(idx, "O", 1.62)
    crystal_struc.wrap_coordinates()
    crystal_struc.write_xyz_file("added_O_on_Si")

charges = {"Si": 4, "O": -2, "H": 1}
def calculate_formal_charge(crystal_struc: CrystalStruc) -> int:
    """
    calculated the formal charge of the structure if the 1O are prtonated. Current not fully applicable to all 
    models. Charges need to be be externally defined; will probably make it so that this is a global variable

    --- input ---
       crystal_struc: the structure in the form of the CrystalStruc class

    --- output ---
      The formal charge of the model after protonating the 1O -> is essentially a indication of how many 3Si need to be
      induced
    """
    formal_charge = 0
    for idx, atom in enumerate(crystal_struc.atoms):
        formal_charge += charges[atom]

        # If the O atom is going to get protonated add 1 for the proton that will be there later
        if atom == "O" and crystal_struc.give_cn(idx) == 1:
            formal_charge += 1

    return formal_charge

# change implementation
def not_bound_to_over_coord(crystal_struc: CrystalStruc, idx_atom: int) -> bool:
    """
    Checks if the given atom is bound to other over-cooridnated atoms

    --- input ---
      crystal_struc: the structure in the form of the CrystalStruc class
      idx_atom: the index of the atom to check

    --- output ---
      boolean as to wether it is not bound to another over-coordinated atom
    """
    for neighbor in crystal_struc.give_nl(idx_atom):
        if crystal_struc.give_cn(neighbor) > crystal_struc.max_cn[crystal_struc.atoms[idx_atom]]:
            return False
    return True

def determine_number_shift(crystal_struc: CrystalStruc, allowed_atoms: List[str]) -> Tuple[int, List[int]]:
    idx_possible_atoms = [idx for idx, atom in enumerate(crystal_struc.atoms) if atom in allowed_atoms and crystal_struc.give_cn(idx) > crystal_struc.max_cn[atom]]
    number_additions_needed = calculate_formal_charge(crystal_struc)
    assert len(idx_possible_atoms) >= number_additions_needed, "There are not enough atoms available for shifting"
    return number_additions_needed, idx_possible_atoms

def check_possible_shifts(crystal_struc: CrystalStruc, number_shifts_needed: int, idx_possible_atoms: List[int]):
    n = number_shifts_needed
    initial_geminal, initial_vicinal, initial_isolated = determine_silanol_types(crystal_struc)
    num_ini_gem, num_ini_vic, num_ini_iso = len(initial_geminal), len(initial_vicinal), len(initial_isolated)

    combinations_for_move: List[Tuple[int, int]] = [] # List[Tuple[idx_O, idx_Si]]
    for idx_O in idx_possible_atoms:
        for idx_neighbor in crystal_struc.give_nl(idx_O):
            if crystal_struc.give_cn(idx_neighbor) <= crystal_struc.max_cn[crystal_struc.atoms[idx_neighbor]]:
                combinations_for_move.append((idx_O, idx_neighbor))
    
    results_of_shift = {}
    for idx_atoms_tuple in combinations_for_move:
        idx_O, idx_Si = idx_atoms_tuple
        assert crystal_struc.atoms[idx_O] == "O", f"index is not an O atom, {idx_O=}"
        assert crystal_struc.atoms[idx_Si] == "Si", f"index is not an Si atom, {idx_Si=}"

        new_gem, new_vic, new_iso = determine_silanol_types(crystal_struc, [idx_Si])
        results_of_shift[idx_atoms_tuple] = [len(new_gem) - num_ini_gem, len(new_vic) - num_ini_vic, len(new_iso) - num_ini_iso]

    df = pd.DataFrame.from_dict(results_of_shift, orient='index', columns=['new_gem_diff', 'new_vic_diff', 'new_iso_diff'])
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'idx_atoms_tuple'}, inplace=True)
    df.to_csv('predicted_results_of_shift.csv', index=False)

    actual_results = {}
    for idx_tuple in results_of_shift.keys():
        new_file_name = "O_{}__Si_{}_shift".format(idx_tuple[0], idx_tuple[1])
        temp_crystal = crystal_struc.copy()
        temp_crystal.move_atom(idx_tuple[1], idx_tuple[0], 2.25)
        temp_crystal.place_new_atom(idx_tuple[1], "O", 1.62)
        struc_optimizer = Lammps_interface(temp_crystal)
        new_atoms, new_coords, new_dims = struc_optimizer.opt_struc(type_opt="minimize")

        temp_crystal.set_xyz_info(new_atoms, new_coords)
        temp_crystal.write_xyz_file(new_file_name)
        
        pe = struc_optimizer.properies["pe"]
        geminal, vicinal, isolated = determine_silanol_types(temp_crystal)
        actual_results[idx_tuple] = np.array([pe, len(geminal), len(vicinal), len(isolated)])

    df = pd.DataFrame.from_dict(actual_results, orient='index', columns=['pe', 'new_gem_diff', 'new_vic_diff', 'new_iso_diff'])
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'idx_atoms_tuple'}, inplace=True)
    df.to_csv('actual_results_of_shift.csv', index=False)
### implement later ####################################
def vicinal_solution(crystal_struc: CrystalStruc, vicinal_idx: set):
    ...

def fix_silanol(atoms, xyz_coords, pre_opt=False):
    if pre_opt:
        opt_tool = Lammps_interface()
        opt_tool.add_structure(atoms, xyz_coords)
        opt_tool.add_dims(xlo = 0, xhi = 21,
                        ylo = 0, yhi = 21,
                        zlo = 0, zhi = 50)
        atoms, xyz_coords, dims = opt_tool.opt_struc("anneal", steps=1500, start_T=298, final_T=298, FF="ReaxFF")
    
    silica_struc = CrystalStruc(atoms, xyz_coords)
    silica_struc.set_cl([21, 21, 50])
    silica_struc.set_cut_offs({"Si": {"Si": 2.0, "O": 2.0, "H":1.0}, "O":{"Si": 2.0, "O": 1.8, "H": 1.2}, "H":{"Si": 1.0, "O": 1.2, "H": 0.8}})

    geminal, vicinal, isolated = determine_silanol_types(silica_struc)

    if len(vicinal) == 0:
        return atoms, xyz_coords
    else:
        atoms, xyz_coords = ...
###########################################################

DIMS = {
        "XLO": 0, 
        "XHI": 21.0,
        "YLO": 0,
        "YHI": 21.0,
        "ZLO": 0,
        "ZHI": 50.0
    }
wanted_CN = {"Si": 4, "O": 2}


def main():
    # crystal_struc = generate_crystal(315, 0.03, 4)
    os.chdir("fixing_structure")
    atoms, xyz = read_xyz("added_O_on_Si_copy_1.xyz")
    crystal_struc = CrystalStruc(atoms=atoms, xyz_coordinates=xyz)
    crystal_struc.set_cut_offs({
        'Si': {'Si': 2.2, 'O': 2.0, 'H': 1.0},
        'O':  {'Si': 2.0, 'O': 1.8, 'H': 1.3},
        'H':  {'Si': 1.0, 'O': 1.3, 'H': 1.0}
    })
    crystal_struc.add_dims(xlo = DIMS["XLO"], xhi = DIMS["XHI"],
                           ylo = DIMS["YLO"], yhi = DIMS["YHI"],
                           zlo = DIMS["ZLO"], zhi = DIMS["ZHI"])
    crystal_struc.set_max_CN({"Si": 4, "O": 2})

    geminal, vicinal, isolated = determine_silanol_types(crystal_struc)
    print(f"{geminal=}, {vicinal=}, {isolated=}")
    num_shift, idx_possible_atoms = determine_number_shift(crystal_struc, ["O"])
    check_possible_shifts(crystal_struc, num_shift, idx_possible_atoms)
    # print(calculate_formal_charge(crystal_struc))
    # add_O_on_3Si(crystal_struc)

def test():
    os.chdir("fixing_structure")
    file_name = glob("*__*.xyz")
    output = {}
    for file in file_name:
        atoms, xyz = read_xyz(file)
        crystal_struc = CrystalStruc(atoms=atoms, xyz_coordinates=xyz)
        crystal_struc.set_cut_offs({
            'Si': {'Si': 2.2, 'O': 2.0, 'H': 1.0},
            'O':  {'Si': 2.0, 'O': 1.8, 'H': 1.3},
            'H':  {'Si': 1.0, 'O': 1.3, 'H': 1.0}
        })
        crystal_struc.add_dims(xlo = DIMS["XLO"], xhi = DIMS["XHI"],
                            ylo = DIMS["YLO"], yhi = DIMS["YHI"],
                            zlo = DIMS["ZLO"], zhi = DIMS["ZHI"])
        crystal_struc.set_max_CN({"Si": 4, "O": 2})

        geminal, vicinal, isolated = determine_silanol_types(crystal_struc)
        output[file] = [len(geminal), len(vicinal), len(isolated)]

    df = pd.DataFrame.from_dict(output, orient='index', columns=['new_gem_diff', 'new_vic_diff', 'new_iso_diff'])
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'file_name'}, inplace=True)
    df.to_csv('actualy_results_of_shift.csv', index=False)


if __name__ == "__main__":
    main()
    # test()