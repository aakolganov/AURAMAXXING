from ase.io import read, write
from ase import Atoms
from pathlib import Path
from base import AmorphousStruc

def write_structure_to_file(amorphous_struct: AmorphousStruc, file_name: Path, write_xyz: bool=False, append: bool=True) -> None:
    amorphous_struct.sort_atoms()
    file_path = str(file_name)
    amorphous_struct.atoms.write(file_path + ".vasp", format="vasp", append=False)
    if write_xyz:
        amorphous_struct.atoms.write(file_path +".xyz", format="xyz", append=append)


def add_dump_to_traj(dump_path: str = "LAMMPS/dump.xyz", traj_file: str = "growth_trajectory.xyz"):
    """
    Add the content of LAMMPS dump to the trajectory file.
    """
    frames = read(dump_path, index=":")
    for frame in frames:
        write(traj_file, frame, format="xyz", append=True)


def highlight_coordination(amorphous_struct, output_file: str) -> None:
    """
    Save structure with modified atomic numbers to highlight coordination defects.
    """
    amorphous_struct.sort_atoms()
    atoms_copy: Atoms = amorphous_struct.atoms.copy()
    numbers = atoms_copy.get_atomic_numbers()

    for i, atom in enumerate(atoms_copy):
        symbol = atom.symbol
        if symbol in amorphous_struct.max_cn:
            target_cn = amorphous_struct.max_cn[symbol]
            current_cn = amorphous_struct.get_cn(i)
            if current_cn < target_cn:
                numbers[i] -= 1
            elif current_cn > target_cn:
                numbers[i] += 1

    atoms_copy.set_atomic_numbers(numbers)
    atoms_copy.write(output_file, format="xyz")
    
