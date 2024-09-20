import numpy as np
import numba as nb
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from typing import Dict, List, Callable
from scipy.stats import burr12, norm, uniform

from LAMMPS_interface import Lammps_interface

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

d_min_max: Dict[str, Dict[str, List]] = {
    "Si": {
        "Si": [2.6, 2.8],
        "O": [1.5850717394267364, 1.92]
        },
    
    "O": {
        "Si": [1.5850717394267364, 1.92],
        "O": [2.05, 2.2]
    }
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
@nb.njit
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

def make_Fourier_function(Lx, Ly, steps, alpha, n_max, m_max):
    x = np.linspace(0, Lx, steps)
    y = np.linspace(0, Ly, steps)
    mesh_x, mesh_y = np.meshgrid(x, y)
    Fourier_Series = np.zeros((steps, steps))

    for n in range(1, n_max + 1):
        for m in range(1, m_max + 1):
            # determine the standard deviation for the normal distribution and sample it
            std = 1 / np.sqrt(alpha * (m ** 2 + n ** 2))
            # add the given mode to the Fourier series
            b = np.random.normal(loc=0, scale=std, size=1)
            Fourier_Series = Fourier_Series + b * np.sin((m * np.pi * mesh_x) / Lx) * np.sin((n * np.pi * mesh_y) / Ly)
    return Fourier_Series

def get_cn(idx_atom, current_xyz_coords, cl):
    mic_coords = mic(current_xyz_coords[idx_atom], current_xyz_coords, nb.typed.List(cl), len(current_xyz_coords))
    dists = np.linalg.norm(mic_coords, axis=1)
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
    return dist_from_atom >= d_max

def beyond_d_min(new_atom, new_coords, current_atoms, current_xyz, cl, idx_atom_check_agaisnt) -> bool:
    dist_from_atom = np.linalg.norm(mic(new_coords, current_xyz[idx_atom_check_agaisnt], nb.typed.List(cl), 1))
    d_min = d_min_max[current_atoms[idx_atom_check_agaisnt]][new_atom][0]
    return dist_from_atom >= d_min

def wrap_struc(current_xyz, cl):
    return current_xyz % cl

def find_nearest(value, array):
    value = np.array(value)
    return np.abs(array-value).argmin()

def choose_vector(current_atoms, current_xyz_coords, atom_type, idx_connect_to, cl, limits: List = None):
    # if len(current_atoms) < 2:
    dist = sample_dist[current_atoms[idx_connect_to]][atom_type]
    random_direction = np.random.randn(3)
    unit_vector = random_direction/np.linalg.norm(random_direction)
    new_coords = current_xyz_coords[idx_connect_to] + unit_vector * dist
    # else:
    #     new_coords = absolute_place_atom(current_atoms, current_xyz_coords, atom_type, idx_connect_to, cl)

    if limits is None:
       pass
    else:
        choosing_vector = True
        MAX_ITER, current_iter = 250, 0 
        max_fac = 0
        max_fac_coords = None
        while choosing_vector and current_iter < MAX_ITER:
            current_iter += 1
            x, y, z = new_coords
            idx_x, idx_y = find_nearest(x, limits[0]), find_nearest(y, limits[1])
            idx_z_min = np.where(current_xyz_coords[:, 2] == np.min(current_xyz_coords[:, 2]))[0][0]
            # lower_bound = current_xyz_coords[idx_z_min, 2] - limits[2][idx_x, idx_y] 
            # upper_bound = current_xyz_coords[idx_z_min, 2] + limits[3][idx_x, idx_y] 
            lower_bound = np.mean(current_xyz_coords[:, 2]) - limits[2][idx_x, idx_y] 
            upper_bound = np.mean(current_xyz_coords[:, 2]) + limits[3][idx_x, idx_y]
            
            # upper_bound = current_xyz_coords[np.where(current_xyz_coords[:, 2] == np.min(current_xyz_coords[:, 2]))[0], 2] + limits[3][idx_x, idx_y] 

            if z >= lower_bound and z <= upper_bound: 
                choosing_vector = False
            else:
                if z <= lower_bound:
                    # factor = np.exp(-(z-lower_bound)**2)
                    factor = 0
                else:
                    # factor = np.exp(-(np.abs(upper_bound-z)))
                    factor = 0
                if factor > max_fac:
                    max_fac = factor
                    max_fac_coords = new_coords

                rng = np.random.rand(1)[0]
                factor_weight = factor/(1+factor)
                if factor_weight <= rng:
                    dist = sample_dist[current_atoms[idx_connect_to]][atom_type]
                    random_direction = np.random.randn(3)
                    unit_vector = random_direction/np.linalg.norm(random_direction)
                    new_coords = current_xyz_coords[idx_connect_to] + unit_vector * dist
                else:
                    choosing_vector = False

    if current_iter == MAX_ITER:
        # return max_fac_coords
        return None
    else:
        return new_coords

def absolute_place_atom(current_atoms, current_xyz_coords, atom_type, idx_connect_to, cl):
    all_idx = np.arange(len(current_atoms))
    mic_xyz = mic(current_xyz_coords[idx_connect_to], current_xyz_coords, cl, len(current_xyz_coords))
    mic_xyz_abs = np.abs(mic_xyz)
    all_d_min = np.array([d_min_max[atom_type][other_atom][0] for other_atom in current_atoms])
    all_d_max = np.array([d_min_max[atom_type][other_atom][1] for other_atom in current_atoms])

    mask_chosen_atom = all_idx != idx_connect_to
    mask_x = mic_xyz_abs[:, 0] <= 5.0
    mask_y = mic_xyz_abs[:, 1] <= 5.0
    mask_z = mic_xyz_abs[:, 2] <= 5.0

    close_xyz = mic_xyz[np.logical_and.reduce([mask_chosen_atom, mask_x, mask_y, mask_z])] + current_xyz_coords[idx_connect_to]
    close_atoms = current_atoms[np.logical_and.reduce([mask_chosen_atom, mask_x, mask_y, mask_z])]
    all_d_min = [np.logical_and.reduce([mask_chosen_atom, mask_x, mask_y, mask_z])]
    all_d_max = [np.logical_and.reduce([mask_chosen_atom, mask_x, mask_y, mask_z])]
    
    range_theta = np.linspace(0, 2 * np.pi, 250)
    range_phi = np.linspace(0, 2 * np.pi, 250)
    mesh_theta, mesh_phi = np.meshgrid(range_theta, range_phi)

    r = sample_dist[current_atoms[idx_connect_to]][atom_type]
    x = r * np.sin(mesh_phi) * np.cos(mesh_theta)
    y = r * np.sin(mesh_phi) * np.sin(mesh_theta)
    z = r * np.cos(mesh_phi)


    new_points = current_xyz_coords[idx_connect_to][:, np.newaxis, np.newaxis] + np.array([x,y,z])
    dists = np.linalg.norm(close_xyz[:, :, np.newaxis, np.newaxis] - new_points, axis=1)
    dist_points = np.min(dists, axis=0)

    mask_d_min = np.where(dist_points >= all_d_min, True, False)
    mask_d_max = np.where(dist_points >= all_d_max, True, False)
    same_atom_type = np.where(close_atoms == atom_type, True, False)

    past_d_min_not_d_max = new_points.transpose(1, 2, 0)[mask_d_min & ~mask_d_max & ~same_atom_type]
    if len(past_d_min_not_d_max) != 0:
        if past_d_min_not_d_max.shape[0] == 1:
            return past_d_min_not_d_max.flatten()
        else:
            return past_d_min_not_d_max[np.random.choice(past_d_min_not_d_max.shape[0])].flatten()
    
    past_d_max = new_points.transpose(1, 2, 0)[mask_d_max]
    if len(past_d_max) != 0:
        if past_d_max.shape[0] == 1:
            return past_d_max.flatten()
        else:
            return past_d_max[np.random.choice(past_d_max.shape[0])].flatten()
    
    return None


def place_atom(atom_type, cl, current_atoms, current_xyz, idx_connect_to = None, limits=None):
    if idx_connect_to is None: # implies that there are no other atoms here
        new_coords = np.array([cl[0]/2, cl[1]/2, cl[2]/2])
        made_placement = True
    
    else:
        made_placement = False
        MAX_ITER, current_iter = 25, 0
        while current_iter <= MAX_ITER and not made_placement:
            current_iter += 1

            new_coords = choose_vector(current_atoms, current_xyz, atom_type, idx_connect_to, cl, limits=limits)
            if new_coords is None:
                continue
            
            if len(current_atoms) >= 2:
                if len(current_atoms) <= 999:
                    check_idx = np.arange(len(current_atoms))[np.abs(new_coords[2] - current_xyz[:, 2]) <= 2.8]
                    made_placement = worker_job(
                        check_idx[::-1],
                        atom_type = atom_type,
                        new_coords = new_coords,
                        current_atoms = current_atoms,
                        current_xyz = current_xyz,
                        cl = cl
                        )
                else:
                    made_placement = concurrent_worker_process_k(
                        np.arange(len(current_atoms))[::-1],
                        atom_type = atom_type,
                        new_coords = new_coords,
                        current_atoms = current_atoms,
                        current_xyz = current_xyz,
                        cl = list(cl)
                        )
            else:
                made_placement = True
            
    if made_placement:
        current_atoms = np.append(current_atoms, atom_type)
        current_xyz = np.vstack((current_xyz, new_coords))
        return current_atoms, current_xyz, made_placement
    else:
        return current_atoms, current_xyz, made_placement

def set_i(current_atoms, current_xyz, cl) -> int:
    possible_choices = {}
    # i = []
    if len(current_atoms) != 0:
        current_number_Si = len(np.where(current_atoms == "Si")[0])
        current_number_O = len(np.where(current_atoms == "O")[0])

        if 2*current_number_Si > current_number_O:
            for n, atom in enumerate(current_atoms):
                if atom == "Si":
                    cn = get_cn(n, current_xyz, cl)
                    if cn < wanted_CN[atom]:
                        if cn not in possible_choices:
                            possible_choices[cn] = []
                        possible_choices[cn].append(n)
                        # i.append(n)
        else:
            for n, atom in enumerate(current_atoms):
                if atom == "O":
                    cn = get_cn(n, current_xyz, cl)
                    if cn < wanted_CN[atom]:
                        if cn not in possible_choices:
                            possible_choices[cn] = []
                        possible_choices[cn].append(n)
                        # i.append(n)
        keys = np.array([key for key in possible_choices.keys()])
        chosen_key = np.random.choice(keys, p=((keys+1)/np.sum(keys+1)))
        i = possible_choices[chosen_key]
        
        if np.all(current_xyz[i, 2] - np.mean(current_xyz[i, 2]) == 0):
            weighting = np.empty(len(current_atoms)).fill(1/len(current_atoms))
        else:
            exp_factor = np.exp(-(np.abs(current_xyz[i, 2] - np.mean(current_xyz[i, 2]))**2))
            weighting = (exp_factor)/sum(exp_factor)
        return np.random.choice(np.array(i, dtype=int), p=weighting)

    else:
        return 0

def chunk_data(data, num_chunks):
    data = list(data)
    total_items = len(data)
    base_chunk_size = total_items // num_chunks
    remainder = total_items % num_chunks
    
    start = 0
    for i in range(num_chunks):
        # Calculate the size of the current chunk
        end = start + base_chunk_size + (1 if i < remainder else 0)
        yield data[start:end]
        start = end

def chunk_data_round_robin(data, num_chunks):
    # Initialize the chunks as empty lists
    chunks = [[] for _ in range(num_chunks)]

    # Distribute data elements into chunks in a round-robin fashion
    for i, item in enumerate(data):
        chunks[i % num_chunks].append(item)
    
    return chunks

def worker_job(idx_chunk, atom_type, new_coords, current_atoms, current_xyz, cl):
    made_placement = 0
    for k in idx_chunk: ### maybe splits this work becuase I have a feeling that this is that is taking a while
        past_d_min = beyond_d_min(atom_type, new_coords, current_atoms, current_xyz, cl, k)
        past_d_max = beyond_d_max(atom_type, new_coords, current_atoms, current_xyz, cl, k)
        
        if past_d_min and not past_d_max:    
            if current_atoms[k] != atom_type:
                made_placement = True
            else:
                made_placement = False
                break
        elif past_d_max:
            made_placement = True
        else:
            # Too close and completely reject placement
            made_placement = False
            break
    
    return made_placement

def concurrent_worker_process_k(idx_all, **kwargs):
    """ kwargs must be: atom_type, new_coords, current_atoms, current_xyz, cl"""
    atom_type = kwargs["atom_type"] 
    new_coords = kwargs["new_coords"]
    current_atoms = kwargs["current_atoms"] 
    current_xyz = kwargs["current_xyz"]
    cl = kwargs["cl"]

    expected_number_workers = 4 
    actual_number_workers = np.min([expected_number_workers, len(idx_all)])
    chunked_idx = chunk_data_round_robin(idx_all, actual_number_workers)
    with ProcessPoolExecutor(max_workers=actual_number_workers) as executor:
        futures = {executor.submit(
            worker_job, chunk, atom_type, new_coords, current_atoms, current_xyz, cl): 
            (chunk, atom_type, new_coords, current_atoms, current_xyz, cl) for chunk in chunked_idx}
        
        all_results = []
        for future in as_completed(futures):
            job_results = future.result()
            if job_results is False:
                for remaining_job in futures:
                    if not remaining_job.done():
                        remaining_job.cancel()
                return False
            try:
                all_results.append(future.result())
            except Exception as e:
                print(f"Error processing {job_results}: {e}", flush=True)
    
    return True

def slice(current_atoms, current_xyz_coords, limits):
    z_min = np.min(current_xyz_coords[:, 2])
    z_max = np.max(current_xyz_coords[:, 2])
    write_xyz("before_slice", current_atoms, current_xyz_coords)
    for idx, xyz in enumerate(current_xyz_coords):
        z = xyz[2]
        idx_x = find_nearest(xyz[0], limits[0])
        idx_y = find_nearest(xyz[1], limits[1])
        z_bound = limits[3][idx_x, idx_y] + z_min + 2.5
        if z > z_bound:
            current_xyz_coords[idx, 2] -= (z_max - z_min)
    write_xyz("after_slice", current_atoms, current_xyz_coords)
    return current_atoms, current_xyz_coords

def main():
    atoms = np.empty(0)
    xyz_coords = np.empty((0,3))
    dims = {
        "xlo": 0, 
        "xhi": 21.0,
        "ylo": 0,
        "yhi": 21.0,
        "zlo": 0,
        "zhi": 50.0
    }
    cl = [dims["xhi"] - dims["xlo"], 
          dims["yhi"] - dims["ylo"],
          dims["zhi"] - dims["zlo"]]
    cl = nb.typed.List(cl)
     
    x_vals, y_vals = np.linspace(dims["xlo"], dims["xhi"], 500), np.linspace(dims["ylo"], dims["yhi"], 500)
    lower_limit = np.full((500, 500), 0.0) 
    # upper_limit = np.load("test_fourier_mesh.npy")
    upper_limit = make_Fourier_function(cl[0], cl[1], 500, 0.03, 12, 12)
    np.save("upper_limit_all.npy", upper_limit, allow_pickle=True)
    upper_limit[upper_limit < 0] = 0
    upper_limit += 1.5
    lower_limit = upper_limit.copy()
    np.save("upper_limit_altered.npy", upper_limit, allow_pickle=True)
    limits = [x_vals, y_vals, lower_limit, upper_limit]

    number_write = 0
    TOTAL_DESIRED_ATOM, new_total_atoms = 105*3, 0
    hit_final = False
    while new_total_atoms < TOTAL_DESIRED_ATOM:
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
            MAX_ITER, current_iter = 100, 0
            made_placement = False
            while current_iter <= MAX_ITER and not made_placement:
                atoms, xyz_coords, made_placement = place_atom(atom_to_add, cl, atoms, xyz_coords, i, limits=limits)
                current_iter += 1
                i = set_i(atoms, xyz_coords, cl)

            if len(atoms) % 999 == 0:
                opt_tool = Lammps_interface()
                opt_tool.add_structure(atoms, xyz_coords)
                opt_tool.add_dims(xlo = dims["xlo"], xhi = dims["xhi"],
                                  ylo = dims["ylo"], yhi = dims["yhi"],
                                  zlo = dims["zlo"], zhi = dims["zhi"])
                atoms, xyz_coords, dims = opt_tool.opt_struc(removal="rings", max_num_rings=3)
        
        new_total_atoms = len(atoms)
        if made_placement:
            xyz_coords = wrap_struc(xyz_coords, cl)
            print(f"{new_total_atoms}", flush=True)
            
            number_write += 1
            write_xyz(f"strucutre_{number_write}", atoms, xyz_coords)
        
        else:
            opt_tool = Lammps_interface()
            opt_tool.add_structure(atoms, xyz_coords)
            opt_tool.add_dims(xlo = dims["xlo"], xhi = dims["xhi"],
                              ylo = dims["ylo"], yhi = dims["yhi"],
                              zlo = dims["zlo"], zhi = dims["zhi"])
            
            print(f"stuck anneal")
            if hit_final:
                max_steps = 500
                start_T = 298
                final_T = 500
            else:
                max_steps = 1000
                start_T = 298
                final_T = 2000
                
            atoms, xyz_coords, dims = opt_tool.opt_struc("anneal", steps=max_steps, start_T=start_T, final_T=final_T,
                                                         removal="rings", max_num_rings=2)
            limits[3] += 0.5
            number_write += 1
            write_xyz(f"strucutre_{number_write}", atoms, xyz_coords)
        
        new_total_atoms = len(atoms)
        if new_total_atoms == TOTAL_DESIRED_ATOM:
            opt_tool = Lammps_interface()
            opt_tool.add_structure(atoms, xyz_coords)
            opt_tool.add_dims(xlo = dims["xlo"], xhi = dims["xhi"],
                              ylo = dims["ylo"], yhi = dims["yhi"],
                              zlo = dims["zlo"], zhi = dims["zhi"])
            
            if hit_final:
                atoms, xyz_coords, dims = opt_tool.opt_struc("final", steps=2500, start_T=298, final_T=298, 
                                                             removal="everything", max_num_rings=1, max_remove_over=6)
            else:
                atoms, xyz_coords, dims = opt_tool.opt_struc("anneal", steps=2500, start_T=298, final_T=298)
                # atoms, xyz_coords = slice(atoms, xyz_coords, limits)

                atoms, xyz_coords, dims = opt_tool.opt_struc("anneal_with_min", steps=2500, start_T=298, final_T=1000, 
                                                             removal="rings", max_num_rings=0)
            
            hit_final = True
            new_total_atoms = len(atoms)
    
    number_write += 1
    write_xyz(f"strucutre_{number_write}", atoms, xyz_coords)
    write_xyz(f"final_strucutre", atoms, xyz_coords)

if __name__ == "__main__":
    main()