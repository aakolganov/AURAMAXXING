import numpy as np
# import numba as nb
import time

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
    
d_min_max: Dict[str, distribution_intervals[str, Callable]] = {
    "Si": distribution_intervals({
        "Si": [2.9182204094224216, 3.2639004221169623],
        "O": [1.5850717394267364, 1.6416217462397817]
        }),
    
    "O": distribution_intervals({
        "Si": [1.5850717394267364, 1.6416217462397817],
        "O": [2.450711959298154, 2.9345719964268846]
    })
}

class random_sample(dict):
    def __init__(self, *args, **kwargs):
        super(random_sample, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        random_distribution = super(random_sample, self).__getitem__(key)
        if hasattr(random_distribution, "rvs"):
            return random_distribution.rvs()
        return random_distribution

sample_dist: Dict[str, random_sample[str, Callable]] = {
    "Si": random_sample({
        "Si": burr12(c=20.50918422948114, d=3.282331385061921, loc=1.8153399428512698, scale=1.3978541397862818),
        "O": norm(loc=1.613346742833259, scale=0.014426287232598306)
        }),
    
    "O": random_sample({
        "Si": norm(loc=1.613346742833259, scale=0.014426287232598306),
        "O": burr12(c=68.50536301711077, d=0.4299182937422296, loc=-0.2260984913136991, scale=2.788587313802839)
    })
}  

def write_xyz(file_name, atoms, xyz_coords):
    assert len(atoms) == len(xyz_coords)
    with open("{}.xyz".format(file_name), "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for atom, xyz in zip(atoms, xyz_coords):
            f.write(f"{atom}")
            for i in xyz:
                f.write(f" {i}")
            f.write("\n")
# @nb.njit
def mic(xyz_atom1, xyz_atoms2, cl, num_coords: int):
    output_array = np.empty((num_coords, 3))
    for i in range(num_coords):
        if num_coords == 1:
            dr = xyz_atoms2.flatten() - xyz_atom1
        else:
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

def place_atom(atom_type, cl, i, current_atoms, current_xyz):
    made_placement = False
    start = time.time()
    if len(current_atoms) == 0:
        made_placement = True
        current_atoms = np.append(current_atoms, atom_type)
        current_xyz = np.vstack((current_xyz, np.array([cl[0]/2, cl[1]/2, np.min([cl[2]/2, 7.5])])))
        return current_atoms, current_xyz, made_placement

    else:
        MAX_ITER, current_iter = 1000, 0
        while current_iter <= MAX_ITER and not made_placement:
            # print(current_iter)
            current_iter += 1
            random_direction = np.random.randn(3)
            dist = sample_dist[current_atoms[i]][atom_type]
            unit_vector = random_direction/np.linalg.norm(random_direction)

            new_coords = current_xyz[i] + unit_vector * dist
            if len(current_atoms) > 2:
                for k in range(len(current_atoms)):
                    if k != i:
                        if np.linalg.norm(mic(new_coords, current_xyz[k], cl, 1)) >= d_min_max[current_atoms[k]][atom_type][0]:
                            made_placement = True
                        else:
                            # If there is over lap it may be due to the atom now being bound to another atom
                            # other_atom_type = "Si" if atom_type == "O" else "O"
                            # dists = np.linalg.norm(mic(new_coords, current_xyz[current_atoms == other_atom_type], cl, len(current_xyz[current_atoms == other_atom_type])), axis=1)
                            # bonds = np.where((dists >= d_min_max["Si"]["O"][0]) & (dists <= d_min_max["Si"]["O"][1]))[0]
                            
                            # if len(bonds) > 1:
                            #     for bound_atom, bound_xyz in zip(current_atoms[bonds], current_xyz[bonds]):
                            #         if np.linalg.norm(mic(new_coords, bound_xyz, cl, 1)) >= d_min_max[bound_atom][atom_type][0]:
                            #             print(bound_atom, atom_type)
                            #             made_placement = True
                            #         else:
                            #             made_placement = False
                            #             break
                            # else:
                            made_placement = False
                            break
                            # # If this sum is greater than 0 then that means that there is at least one more atom near by
                            # if sum(bonds) > 1:
                            #     # So we check the atoms which it is not bound to to see if there is a problem with placements against other atoms
                            #     bound_xyz_coords = current_xyz[~bonds]
                            #     bound_atoms = current_atoms[~bonds]
                            #     for bounds_atom, bound_xyz in zip(bound_atoms, bound_xyz_coords):
                            #         if np.linalg.norm(mic(new_coords, bound_xyz, cl, 1)) >= sample_dist[bounds_atom][atom_type]:
                            #             # If the distance is greater than the minimum distance that we have set: YIPPEE
                            #             made_placement = True
                            #         else:
                            #             # If we find that there is another atom too close by we can stop the check and continue
                            #             made_placement = False
                            #             break
                            # else:
                            #     # For if there are no other atoms near by enough for it to be bound
                            
            else:
                made_placement = True      
        
        if made_placement:
            current_atoms = np.append(current_atoms, atom_type)
            current_xyz = np.vstack((current_xyz, new_coords))
            # print(time.time()-start)
            return current_atoms, current_xyz, made_placement
        else:
            # print(time.time()-start)
            return current_atoms, current_xyz, made_placement

def main():
    atoms = np.empty(0)
    xyz_coords = np.empty((0,3))
    cl = [1E6, 1E6, 1E6]
    
    atom_types = ["Si", "O"]
    MAX_SI = 50
    MAX_O = 2*MAX_SI
    desired_atoms = MAX_SI + MAX_O
    total_atoms = 0

    i, j = 0, 0
    while total_atoms != desired_atoms:
        if total_atoms == 0:
            atom_type = "Si"
            atoms, xyz_coords, _ = place_atom(atom_type, cl, i, atoms, xyz_coords)
            total_atoms += 1
            write_xyz(f"number_{total_atoms}", atoms, xyz_coords)
            j += 1

        elif i == j:
            prob_Si = (MAX_SI - len(atoms[atoms == "Si"]))/desired_atoms
            atom_type = np.random.choice(atom_types, p=(prob_Si, 1-prob_Si))
            atoms, xyz_coords, _ = place_atom(atom_type, cl, i, atoms, xyz_coords)
            total_atoms += 1
            write_xyz(f"number_{total_atoms}", atoms, xyz_coords)
            print("hit")
            j += 1
        
        else:
            if get_cn(i, xyz_coords, cl) == wanted_CN[atoms[i]]:
                i += 1
            
            else:
                atom_type = "Si" if atoms[i] == "O" else "O"
                atoms, xyz_coords, made_placement = place_atom(atom_type, cl, np.min([i, j]), atoms, xyz_coords)
                
                if made_placement:
                    j += 1
                    total_atoms += 1
                    print(f"placed {total_atoms}")
                    write_xyz(f"number_{total_atoms}", atoms, xyz_coords)

                else:
                    to_remove = np.random.choice([1, 2], p=[0.9, 0.1])
                    for _ in range(to_remove):
                        j -= 1
                        total_atoms -= 1
                        atoms = np.delete(atoms, -1)
                        xyz_coords = np.delete(xyz_coords, -1, axis=0)
                    # print(len(atoms))
                    


if __name__ == "__main__":
    main()