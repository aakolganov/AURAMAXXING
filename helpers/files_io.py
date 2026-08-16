from ase import Atoms
from pathlib import Path
from base import AmorphousStruc

def write_structure_to_file(amorphous_struct: AmorphousStruc, file_name: Path, write_xyz: bool=False, append: bool=True) -> None:
    # Write a species-sorted COPY (VASP needs species grouping). Sorting the live structure
    # here would reorder atom indices mid-run, so a dumps-on run would consume a different
    # RNG stream than a dumps-off run -- dumps must be pure observers of the trajectory.
    atoms = amorphous_struct.atoms.copy()
    atoms = atoms[atoms.numbers.argsort()]
    file_path = str(file_name)
    atoms.write(file_path + ".vasp", format="vasp", append=False)
    if write_xyz:
        atoms.write(file_path +".xyz", format="xyz", append=append)


def highlight_coordination(amorphous_struct, output_file: str) -> None:
    """
    Save structure with modified atomic numbers to highlight coordination defects.
    """
    amorphous_struct.sort_atoms()
    atoms_copy: Atoms = amorphous_struct.atoms.copy()
    numbers = atoms_copy.get_atomic_numbers()
    # Per-atom targets so a tagged Al is compared against its own max CN.
    target_cns = amorphous_struct.max_cn_array()

    for i, atom in enumerate(atoms_copy):
        if atom.symbol in amorphous_struct.max_cn:
            target_cn = target_cns[i]
            current_cn = amorphous_struct.get_cn(i)
            if current_cn < target_cn:
                numbers[i] -= 1
            elif current_cn > target_cn:
                numbers[i] += 1

    atoms_copy.set_atomic_numbers(numbers)
    atoms_copy.write(output_file, format="xyz")
    
