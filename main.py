import numpy as np
import numba as nb
import time
from LAMMPS_interface import Lammps_interface

from typing import Dict, List, Callable
from scipy.stats import burr12, norm, uniform

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
        "Si": [2.9182204094224216, 3.0639004221169623],
        "O": [1.5850717394267364, 1.6416217462397817]
        }),
    
    "O": distribution_intervals({
        "Si": [1.5850717394267364, 1.6416217462397817],
        "O": [2.350711959298154, 2.45]
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

# sample_dist: Dict[str, random_sample[str, Callable]] = {
#     "Si": random_sample({
#         "Si": burr12(c=20.50918422948114, d=3.282331385061921, loc=1.8153399428512698, scale=1.3978541397862818),
#         "O": norm(loc=1.613346742833259, scale=0.014426287232598306)
#         }),
    
#     "O": random_sample({
#         "Si": norm(loc=1.613346742833259, scale=0.014426287232598306),
#         "O": burr12(c=68.50536301711077, d=0.4299182937422296, loc=-0.2260984913136991, scale=2.788587313802839)
#     })
# }  

sample_dist: Dict[str, random_sample[str, Callable]] = {
    "Si": random_sample({
        "Si": burr12(c=20.50918422948114, d=3.282331385061921, loc=1.8153399428512698, scale=1.3978541397862818),
        "O": uniform(loc=1.6, scale=0.32)
        }),
    
    "O": random_sample({
        "Si": uniform(loc=1.6, scale=0.32),
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
    mic_coords = mic(current_xyz_coords[idx_atom], current_xyz_coords, nb.typed.List(cl), len(current_xyz_coords))
    dists = np.linalg.norm(mic_coords, axis = 1)
    return np.sum(((dists != 0) & (dists <= 2.0)))

def get_all_cn(all_xyz_coords, cl):
    num_coords = len(all_xyz_coords)
    dists = np.empty((num_coords, num_coords))
    for i, xyz in enumerate(all_xyz_coords):
        mic_coords = mic(xyz, all_xyz_coords, cl, num_coords)
        dist = np.linalg.norm(mic_coords, axis=1)
        dists[:, i] = dist
   
    bond_mat = np.where((dists <= 2.0) & (dists > 0.01), 1, 0)
    all_cn = np.sum(bond_mat, axis=1)
    return all_cn

def beyond_d_max(new_atom, new_coords, current_atoms, current_xyz, cl, idx_atom_check_agaisnt) -> bool:
    dist_from_atom = np.linalg.norm(mic(new_coords, current_xyz[idx_atom_check_agaisnt], nb.typed.List(cl), 1))
    d_max = d_min_max[current_atoms[idx_atom_check_agaisnt]][new_atom][1]
    # d_max = sample_dist[current_atoms[idx_atom_check_agaisnt]][new_atom]
    return dist_from_atom >= d_max

def beyond_d_min(new_atom, new_coords, current_atoms, current_xyz, cl, idx_atom_check_agaisnt) -> bool:
    dist_from_atom = np.linalg.norm(mic(new_coords, current_xyz[idx_atom_check_agaisnt], nb.typed.List(cl), 1))
    d_min = d_min_max[current_atoms[idx_atom_check_agaisnt]][new_atom][0]
    # d_min = sample_dist[current_atoms[idx_atom_check_agaisnt]][new_atom]
    return dist_from_atom >= d_min

def wrap_struc(current_xyz, cl):
    return current_xyz % cl

def place_atom(atom_type, cl, current_atoms, current_xyz, idx_connect_to = None):
    assert len(current_atoms) == len(current_xyz)
    made_placement = False

    if idx_connect_to is None: # implies that there are no other atoms here
        assert len(current_atoms) == 0
        new_coords = np.array([cl[0]/2, cl[1]/2, cl[2]/2])
        made_placement = True
    
    else:
        MAX_ITER, current_iter = 250, 0
        while current_iter <= MAX_ITER and not made_placement:
            current_iter += 1
            
            random_direction = np.random.randn(3) # need something more intelligent for making a direction
            dist = sample_dist[current_atoms[idx_connect_to]][atom_type]
            unit_vector = random_direction/np.linalg.norm(random_direction)
            new_coords = current_xyz[idx_connect_to] + unit_vector * dist

            if len(current_atoms) >= 2:
                for k in range(len(current_atoms)):
                    if k != idx_connect_to:
                        # Far enough away from those atoms for there to be no problem
                        if beyond_d_max(atom_type, new_coords, current_atoms, current_xyz, cl, k):
                            made_placement = True

                        elif current_atoms[k] == atom_type:
                            # Too close and the same type of atom so we are assuming that that cannot be bound
                            # completely reject placement
                            made_placement = False
                            break
                        
                        elif beyond_d_min(atom_type, new_coords, current_atoms, current_xyz, cl, k):
                            # not too close
                            made_placement = True
                        
                        else:
                            # Too close and completely reject placement
                            made_placement = False
                            break

            else:
                made_placement = True
            
    if made_placement:
        current_atoms = np.append(current_atoms, atom_type)
        current_xyz = np.vstack((current_xyz, new_coords))
        return current_atoms, current_xyz, made_placement
    else:
        return current_atoms, current_xyz, made_placement

def set_i(current_atoms, current_xyz, cl) -> int:
    current_number_Si = len(np.where(current_atoms == "Si")[0])
    current_number_O = len(np.where(current_atoms == "O")[0])
    # print(current_number_Si, current_number_O)
    i = []
    if len(current_atoms) == 0:
        i = [0]
    elif 2*current_number_Si > current_number_O:
        for n, atom in enumerate(current_atoms):
            if atom == "Si":
                cn = get_cn(n, current_xyz, cl)
                if cn != wanted_CN[current_atoms[n]]:
                    i.append(n)
                    break
    else:
        for n, atom in enumerate(current_atoms):
            if atom == "O":
                cn = get_cn(n, current_xyz, cl)
                if cn != wanted_CN[current_atoms[n]]:
                    i.append(n)
                    break
    
    return np.random.choice(np.array(i, dtype=int))

def main():
    atoms = np.empty(0)
    xyz_coords = np.empty((0,3))
    dims = {
        "xlo": 0, 
        "xhi": 21.5,
        "ylo": 0,
        "yhi": 21.5,
        "zlo": 0,
        "zhi": 50.0
    }
    cl = [dims["xhi"] - dims["xlo"], 
          dims["yhi"] - dims["ylo"],
          dims["zhi"] - dims["zlo"]]
    # cl = nb.typed.List(cl)
    stuck = 0
    TOTAL_DESIRED_ATOM, new_total_atoms = 432, 0
    while new_total_atoms <= TOTAL_DESIRED_ATOM:
        i = set_i(atoms, xyz_coords, cl)
        if i is None:
            idx_remove = np.random.randint(0, len(atoms))
            atoms = np.delete(atoms, idx_remove)
            xyz_coords = np.delete(xyz_coords, idx_remove, axis=0)
            pass

        if len(atoms) == 0:
            atom_to_add = "Si"
        else:
            atom_to_add = "O" if atoms[i] == "Si" else "Si"
        
        current_number_atoms = len(atoms)
        if current_number_atoms == 0:
            atoms, xyz_coords, made_placement = place_atom(atom_to_add, cl, atoms, xyz_coords)

        else:
            MAX_ITER, current_iter = 50, 0
            made_placement = False
            while current_iter <= MAX_ITER and not made_placement:
                atoms, xyz_coords, made_placement = place_atom(atom_to_add, cl, atoms, xyz_coords, i)
                current_iter += 1
                i = set_i(atoms, xyz_coords, cl)

        new_total_atoms = len(atoms)
        if made_placement:
            if new_total_atoms % 144 == 0:
                opt_tool = Lammps_interface()
                opt_tool.add_structure(atoms, xyz_coords)
                opt_tool.add_dims(xlo = dims["xlo"], xhi = dims["xhi"],
                                  ylo = dims["ylo"], yhi = dims["yhi"],
                                  zlo = dims["zlo"], zhi = dims["zhi"])
                # atoms, xyz_coords, _ = opt_tool.opt_struc()
                atoms, xyz_coords, dims = opt_tool.opt_struc("anneal", steps=250)
                cl = [dims["xhi"] - dims["xlo"], 
                      dims["yhi"] - dims["ylo"],
                      dims["zhi"] - dims["zlo"]]
            write_xyz(f"number_{new_total_atoms}", atoms, xyz_coords)
            print(f"{new_total_atoms}")
        
        else:
            stuck += 1
            opt_tool = Lammps_interface()
            opt_tool.add_structure(atoms, xyz_coords)
            opt_tool.add_dims(xlo = dims["xlo"], xhi = dims["xhi"],
                                ylo = dims["ylo"], yhi = dims["yhi"],
                                zlo = dims["zlo"], zhi = dims["zhi"])
            if stuck == 15:
                atoms, xyz_coords, dims = opt_tool.opt_struc("anneal", steps=15)
                stuck = 0
            else:
                atoms, xyz_coords, dims = opt_tool.opt_struc()
            cl = [dims["xhi"] - dims["xlo"], 
                    dims["yhi"] - dims["ylo"],
                    dims["zhi"] - dims["zlo"]]
            # atoms, xyz_coords, _ = opt_tool.opt_struc()
            print(f"stuck opt {stuck}")

if __name__ == "__main__":
    main()