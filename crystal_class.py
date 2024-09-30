from xyzClass import XyzStruc
from typing import Dict, List, Callable
from scipy.stats import burr12, uniform

import numpy as np
import numba as nb
import copy

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

d_min_max: Dict[str, Dict[str, List]] = {
    "Si": {"Si": [2.6, 3.0],
          "O": [1.5850717394267364, 1.92]},
    
    "O": {"Si": [1.5850717394267364, 1.92],
          "O": [2.05, 2.4]}
}

class CrystalStruc(XyzStruc):
    def __init__(self, atoms, xyz_coordinates):
        super().__init__(atoms, xyz_coordinates)

        self.dims: Dict[str, float|None] = {
            "xlo": None, 
            "xhi": None, 
            "ylo": None, 
            "yhi": None, 
            "zlo": None, 
            "zhi": None
        }

        self.limits = None
        self.max_cn = None
    
    def add_dims(self, xlo: float = None, xhi: float = None, ylo: float = None, yhi: float = None, zlo: float = None, zhi: float = None):
        args = locals()
        for key, val in args.items():
                if key != "self" and val is not None:
                    self.dims[key] = float(val)

        self.set_cl([self.dims["xhi"] - self.dims["xlo"],
                     self.dims["yhi"] - self.dims["ylo"],
                     self.dims["zhi"] - self.dims["zlo"]])
    
    def set_limits(self, alpha: float, n_m: int) -> None:
        """
        Sets the upper and lower limits of between the structure is allowed to grow. 
        Current behavious is to make a stochastic fourier series specified through alpha and n_m for upper limit
        and setting the lower limit to 0.

        --- input ---
          alpha - empirical roughenss parameter of the Fourier function
          n_m - the number of Fouier mode to use when making the function

        --- effect ---
          sets the class variable "limits" with a list containing, in this order:
            1D array of x-values used to made the function
            1D array of y-values used to made the function
            2D array of lower limit
            2D array of the upper limit
        """
        x_vals = np.linspace(self.dims["xlo"], self.dims["xhi"], 500)
        y_vals = np.linspace(self.dims["xlo"], self.dims["xhi"], 500), np.linspace(self.dims["ylo"], self.dims["yhi"], 500)
        upper_limit = make_Fourier_function(self.cell_lengths[0], self.cell_lengths[1], 500, alpha, n_m, n_m)
        np.save("upper_limit_all.npy", upper_limit, allow_pickle=True)

        positive_portion = upper_limit.copy() + 1.5
        positive_portion[positive_portion < 0] = 0
        negative_portion = upper_limit.copy() - 1.5
        negative_portion[negative_portion > 0] = 0
        negative_portion = np.abs(negative_portion)
        
        roughness_positive = np.mean((positive_portion - np.mean(positive_portion))**2)
        roughness_negative = np.mean((negative_portion - np.mean(negative_portion))**2)
        if roughness_positive > roughness_negative:
            upper_limit = positive_portion
        else:
            upper_limit = negative_portion
        np.save("upper_limit_altered.npy", upper_limit, allow_pickle=True)

        lower_limit = np.full((500, 500), 0.0) 
        self.limits = [x_vals, y_vals, lower_limit, upper_limit]
    
    def set_max_CN(self, wanted_CN: Dict[str, int]):
        self.max_cn = wanted_CN
    
    def copy(self) -> 'CrystalStruc':
        return copy.deepcopy(self)

    def set_i(self, weight_z=False) -> int:
        """
        Sets the index of atom which will attempt to have an atom added to it.
        Does this through checking the current coordination of each atom and adding it to a dict of coordination numbers.
        Chooses key and is weighted such that higher-coordination atoms are more likely to be chosen.

        --- input ---
          weight_z: boolean - allows for option of having the choosing of atom weighted by z-coodinate. Closer to mean of z means more likely

        --- output ---
          index: integer - the index of the atom which will have the new atom added to it
        """
        possible_choices = {}
        if len(self.atoms) != 0:
            current_number_Si = len(np.where(self.atoms == "Si")[0])
            current_number_O = len(np.where(self.atoms == "O")[0])

            if 2*current_number_Si > current_number_O:
                atom_to_connect_new_to = "Si"
            else:
                atom_to_connect_new_to = "O"
            
            for n, atom in enumerate(self.atoms):
                if atom == atom_to_connect_new_to:
                    cn = self.give_cn(n)
                    if cn < self.max_cn[atom]:
                        if cn not in possible_choices:
                            possible_choices[cn] = []
                        possible_choices[cn].append(n)
            keys = np.array([key for key in possible_choices.keys()])
            if len(keys) == 0:
                return np.random.choice(np.arange(len(self.atoms)))
            chosen_key = np.random.choice(keys, p=((2**(keys+1))/np.sum(2**(keys+1))))
            i = possible_choices[chosen_key]
            
            if weight_z:
                if np.all(self.xyz[i, 2] - np.mean(self.xyz[i, 2]) == 0):
                    weighting = np.empty(len(self.xyz)).fill(1/len(self.xyz))
                else:
                    exp_factor = np.exp(-(np.abs(self.xyz[i, 2] - np.mean(self.xyz[i, 2]))**2))
                    weighting = (exp_factor)/sum(exp_factor)
                return np.random.choice(np.array(i, dtype=int), p=weighting)
            else:
                return np.random.choice(np.array(i, dtype=int))

        else:
            return 0
        
    def place_atom_random(self, atom_type: str, idx_connect_to: int = None, limits=None) -> bool:
        """
        Places a new atom on to the current structure.

        --- input ---
          atom_type - The type of atom which will be added to the structure. string of the chemical symbol.
          idx_connect_to: int - (optional) the specific index to which the new atom will be added to. Default behavious is to place in the center of the box
          limits: List - (optional) will limit the placement to defined bounds in z-direction. list contains, in this order:   
            
            \tarray of x-values used to make limits
            \tarray of x-values used to make limits
            \tlower limit
            \tupper limit
        
        --- output ---
          made_placement - boolean value for if the atom could be placed to the specified atom
        """
        if idx_connect_to is None: # implies that there are no other atoms here
            new_coords = np.array([self.cell_lengths[0]/2, self.cell_lengths[1]/2, self.cell_lengths[2]/2])
            made_placement = True
        
        else:
            made_placement = False
            MAX_ITER, current_iter = 25, 0
            while current_iter <= MAX_ITER and not made_placement:
                current_iter += 1

                new_coords = choose_vector(self.atoms, self.xyz, atom_type, idx_connect_to, self.cell_lengths, limits=limits)
                if new_coords is None:
                    continue
                
                if len(self.atoms) >= 2:
                    check_idx = np.arange(len(self.atoms))[np.abs(new_coords[2] - self.xyz[:, 2]) <= 2.8]
                    made_placement = self.check_placement(
                        check_idx[::-1],
                        atom_type = atom_type,
                        new_coords = new_coords)
                    
                else:
                    made_placement = True
                
        if made_placement:
            self.atoms = np.append(self.atoms, atom_type)
            self.xyz = np.vstack((self.xyz, new_coords))
            return made_placement
        else:
            return made_placement
    
    def check_placement(self, idx_chunk: List[int], atom_type: str, new_coords: np.ndarray) -> bool:
        """
        Checks whether the chosen placement for an atom is considered valid.

        --- input ---
        idx_chunk - the list of indicies that must be checked for if that atom can be placed where it will be placed
        atom_type - the atom that wants to be placed
        new_coords - the xyz coordinates of where the atom will be placed

        --- output ---
        made_placement - boolean value signifying if the placement is considered valid
        """
        made_placement = 0
        for k in idx_chunk: ### maybe splits this work becuase I have a feeling that this is that is taking a while
            past_d_min = self.beyond_d_min(atom_type, new_coords, k)
            past_d_max = self.beyond_d_max(atom_type, new_coords, k)
            
            if past_d_min and not past_d_max:    
                if self.atoms[k] != atom_type:
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
    
    def beyond_d_max(self, new_atom_type: str, new_coords: np.ndarray, idx_atom_check_agaisnt: int) -> bool:
        """
        checks if the new atom to be placed is at least d_max units of length away from a specific atom

        --- input ---
          new_atom_type: type of new atom
          new_coords: xyz coordinates of the new atom
          idx_atom_check_agaisnt: index of the atom that will be checked against
        
        --- output ---
          boolean value indicating wether it is far away enough from specific atom
        """
        dist_from_atom = np.linalg.norm(
            _compiled_mic(new_coords, self.xyz[idx_atom_check_agaisnt], nb.typed.List(self.cell_lengths), 1))
        d_max = d_min_max[self.atoms[idx_atom_check_agaisnt]][new_atom_type][1]
        return dist_from_atom >= d_max

    def beyond_d_min(self, new_atom_type: str, new_coords: np.ndarray, idx_atom_check_agaisnt: int) -> bool:
        """
        checks if the new atom to be placed is at least d_min units of length away from a specific atom

        --- input ---
          new_atom_type: type of new atom
          new_coords: xyz coordinates of the new atom
          idx_atom_check_agaisnt: index of the atom that will be checked against
        
        --- output ---
          boolean value indicating wether it is far away enough from specific atom
        """
        dist_from_atom = np.linalg.norm(
            _compiled_mic(new_coords, self.xyz[idx_atom_check_agaisnt], nb.typed.List(self.cell_lengths), 1))
        d_min = d_min_max[self.atoms[idx_atom_check_agaisnt]][new_atom_type][0]
        return dist_from_atom >= d_min
    
    def slice(self, raise_by: float = 0) -> None:
        """
        Slices the structure based on the upper limit for which it is allowed to grow.

        --- input ---
          raise_by - value with which to raise the upper limit by. Default behavious is to not raise the upper limit

        --- effect ---
          Removes all atoms above the value of z equal to lowest atom + upper limit + 1 at given x and y value.
          Rasises the upper
        """
        self.write_xyz_file("before_slice")
        to_delete = []
        for idx, xyz in enumerate(self.xyz):
            z = xyz[2]
            idx_x = find_nearest(xyz[0], self.limits[0])
            idx_y = find_nearest(xyz[1], self.limits[1])
            z_bound = self.limits[3][idx_x, idx_y] + np.min(self.xyz[:, 2]) + 1
            if z > z_bound:
                to_delete.append(idx)
        
        self.atoms = np.delete(self.atoms, to_delete)
        self.xyz = np.delete(self.xyz, to_delete, axis=0)
        self.write_xyz_file("after_slice")
        
        self.limits[3] += raise_by

    # taken from saturation
    def _make_point(self, idx_ref_atom: int, bond_length: float, tolerance: float) -> np.ndarray:
        """
        Makes the point for the function "place_new_atom". Does this through brute-forcing all of the points around
        the already exsiting atom to which the new atom will be added. Will always return a point

        --- input ---
          idx_ref_atom: the atom to which the new atom will be bound to
          bond_length: the distance between the old and the new atom
          tolerance: the distance the new atom should be from the other atoms in the structure

        --- output ---
          numpy array of dimensions (1,3): the new xyz point of the atoms 
        """
        idx_ref_atom = idx_ref_atom % len(self.atoms)
        chosen_xyz = self.xyz[idx_ref_atom]
        all_idx = np.arange(len(self.atoms))
        mic_xyz = self.mic(idx_ref_atom, all_idx)
        mic_xyz_abs = np.abs(mic_xyz)
        
        mask_chosen_atom = all_idx != idx_ref_atom
        mask_x = mic_xyz_abs[:, 0] <= 5.0
        mask_y = mic_xyz_abs[:, 1] <= 5.0
        mask_z = mic_xyz_abs[:, 2] <= 5.0
        
        #Add the values of chosen_xyz to move the mic points back to their "origional"/"intended" positions.
        close_xyz = mic_xyz[np.logical_and.reduce([mask_chosen_atom, mask_x, mask_y, mask_z])] + chosen_xyz

        range_theta = np.linspace(0, 2 * np.pi, 250)
        range_phi = np.linspace(0, 2 * np.pi, 250)
        mesh_theta, mesh_phi = np.meshgrid(range_theta, range_phi, indexing='xy')
        
        r = bond_length
        x = r * np.sin(mesh_phi) * np.cos(mesh_theta)
        y = r * np.sin(mesh_phi) * np.sin(mesh_theta)
        z = r * np.cos(mesh_phi)
        chosen_xyz = chosen_xyz[:, np.newaxis, np.newaxis]
        new_points = np.array((chosen_xyz + np.array([x, y, z])))

        dists = np.linalg.norm(close_xyz[:, :, np.newaxis, np.newaxis] - new_points, axis=1)
        dist_points = np.min(dists, axis=0)
        mask = np.where(dist_points >= tolerance, True, False)
        possible_points = new_points.transpose(1, 2, 0)[mask]
        
        # For the edge case that there are two points which are have the same distance from the bulk.
        if len(possible_points.flatten()) == 3:
            # only one possible point here
            new_point = possible_points
        else:
            # have no tolerable points so choose the one farthestaway from everything
            mask = np.where(dist_points == np.max(dist_points), True, False)

            # make sure that there is only one maximum
            if np.sum(mask) > 1:
                new_point = new_points.transpose(1, 2, 0)[mask]
                new_point = new_point[0]
            else:
                new_point = new_points.transpose(1, 2, 0)[mask]
            #print(f"point taken lower than tolerance by {tolerance - np.max(dist_points)} angstrom")
        
        return new_point.flatten()
    
    # taken from saturation
    def place_new_atom(self, idx_ref_atom: int, new_atom_type: str, bond_length: float, tolerance=1.5) -> None:
        """
        Places specified atom type on reference atom with a specific bond length

        --- input ---
          idx_ref_atom: atom to which the new one will be bound
          new_atom_type: the new atom which will be added to the structure
          bond_length: desired distance between old and new atom
          tolerance: the distance from the other atoms which the new one is allowed to be
        
        --- effect --- 
          adds new atom to the atoms list and the xyz array in the class
        """
        assert idx_ref_atom <= len(self.atoms), f"Trying to access an atom that does not exist. {idx_ref_atom=}, {len(self.atoms)=}"
        new_point = self._make_point(idx_ref_atom, bond_length, tolerance)
        self.add_new_atom(new_atom_type, new_point)

    # taken from saturation
    def move_atom(self, idx_move: int, idx_ref: int, new_dist: float):
        """
        Move a specific atom away from another by a specified distance

        --- input ---
          idx_move: the index of the atom that will be moved
          idx_ref: the index of the atom that will be moved from
          new_dist: the new distance between the atoms

        --- effect ---
          Will move the specific atom in the xyz coordinated of the class instance
        """
        atom_move = self.atoms[idx_move]
        xyz_ref = self.xyz[idx_ref]

        direction = self.mic(idx_move, idx_ref)
        unit_direction = direction / np.linalg.norm(direction)
        new_point = xyz_ref - new_dist * unit_direction
        self.replace_atom(idx_move, atom_move, new_point) 


# methods that cannot be part of the class instance or ones that have simply not been put in yet

# cannot be part of class
@nb.njit
def _compiled_mic(xyz_atom1, xyz_atoms2, cl, num_coords: int):
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

# Not implemented yet -> going to generalie function which calls this
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

# just lazy to refactor
def find_nearest(value, array):
    value = np.array(value)
    return np.abs(array-value).argmin()

# just too lazy to refactor
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
            lower_bound = current_xyz_coords[idx_z_min, 2] - limits[2][idx_x, idx_y] 
            upper_bound = current_xyz_coords[idx_z_min, 2] + limits[3][idx_x, idx_y] 
            # lower_bound = np.mean(current_xyz_coords[:, 2]) - limits[2][idx_x, idx_y] 
            # upper_bound = np.mean(current_xyz_coords[:, 2]) + limits[3][idx_x, idx_y]
            
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