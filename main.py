import numpy as np
import numba as nb # type: ignore

from typing import Dict, List, Callable
from scipy.stats import burr12, norm

wanted_CN = {
    "Si": 4,
    "O": 2
}

class distribution_intervals(dict):
    def __init__(self, *args, **kwargs):
        super(distribution_intervals, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        random_distribution = super(distribution_intervals, self).__getitem__(key)
        if hasattr(random_distribution, "interval"):
            return random_distribution.interval(0.95)
        return random_distribution

P_r_from_radial: Dict[str, distribution_intervals[str, Callable]] = {
    "Si": distribution_intervals({
        "Si": burr12(c=20.50918422948114, d=3.282331385061921, loc=1.8153399428512698, scale=1.3978541397862818),
        "O": norm(loc=1.613346742833259, scale=0.014426287232598306)
        }),
    
    "O": distribution_intervals({
        "Si": norm(loc=1.613346742833259, scale=0.014426287232598306),
        "O": burr12(c=68.50536301711077, d=0.4299182937422296, loc=-0.2260984913136991, scale=2.788587313802839)
    })
}

def write_xyz(file_name, atoms, xyz_coords):
    # assert len(atoms) == len(xyz)
    with open("{}.xyz".format(file_name), "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for atom, xyz in zip(atoms, xyz_coords):
            f.write(f"{atom}")
            for i in xyz:
                f.write(f" {i}")
            f.write("\n")
@nb.njit
def mic(xyz_atom1, xyz_atoms2, cl, num_coords: int):
    output_array = np.empty((num_coords, 3))
    for i in range(num_coords):
        dr = xyz_atoms2[i] - xyz_atom1
        for j in range(3):
            if dr[j] > 0.5*cl[j]:
                dr[j] -= cl[j]
            elif dr[j] < -0.5 * cl[j]:
                dr[j] += cl[j]
            else:
                pass
        output_array[i] = dr

    return output_array

def get_cn(idx_atom, current_xyz_coords, cl):
    mic_coords = mic(current_xyz_coords[idx_atom], current_xyz_coords, cl, len(current_xyz_coords))
    dists = np.linalg.norm(mic_coords, axis = 1)
    return np.sum(((dists != 0) & (dists <= 2.0)))

def place_atom(atom_type, cl, current_atoms, current_xyz):
    if len(current_atoms) == 0:
        current_atoms.append(atom_type)
        current_xyz = np.vstack((current_xyz, np.array(cl[0]/2, cl[1]/2, np.min(cl[2]/2, 7.5))))

    return current_atoms, current_xyz


def main():
    atoms = np.empty(0)
    xyz_coords = np.empty((0,3))
    cl = [21.5, 21.5, 80]
    
    MAX_SI = 144
    MAX_O = 2*MAX_SI
    desired_atoms = MAX_SI + MAX_O
    total_atoms = 0

    atom_types = ["O", "Si"]
    i, j = 1, 1
    while total_atoms != desired_atoms:
        if i == j:
            atom_type = np.random.choice(atom_types)
            atoms, xyz_coords = place_atom(atom_type, )
        else:
            ...
        total_atoms += 1


    write_xyz("test", atoms, xyz_coords)

if __name__ == "__main__":
    main()