import numpy as np
import numba as nb
from typing import Tuple, List, Dict
import scipy
import os
# from skimage import measure

class XyzStruc:
    def __init__(self, atoms, coordinates):
        atoms = np.array(atoms, dtype=str)
        coordiates = np.array(coordinates, dtype=float)

        assert coordiates.shape == (len(atoms), 3), "Coordiantes are not cartesian"
        assert len(atoms) == len(coordiates), f"Different number atoms and coordinates.\n{len(atoms)=}\n{len(coordiates)=}"

        self.atoms = atoms
        self.xyz = coordiates
        self.atom_types = np.unique(self.atoms)
        
        self.cell_lengths = None
        self.cut_offs = None

        self.idx_surface_atoms_top = None
        self.idx_surface_atoms_bot = None
        self.total_surface_area = None

    def write_xyz_file(self, new_file_name: str):
        if os.path.exists(new_file_name + ".xyz"):
            print("File exists")
            n = 1
            copy_file_name = new_file_name + f"_copy_{n}.xyz"
            while os.path.exists(copy_file_name):
                n += 1
                copy_file_name = new_file_name + f"_copy_{n}.xyz"
            new_file_name = new_file_name + f"_copy_{n}"
            print(f"New file name is {new_file_name}")

        num_atoms = len(self.atoms)
        with open(new_file_name + ".xyz", "w") as f:
            f.write(f"{num_atoms}\n")
            f.write("\n")

            for atom, coord in zip(self.atoms, self.xyz):
                f.write(f"{atom}\t{coord[0]}\t{coord[1]}\t{coord[2]}\n")
    
    #Setters
    def set_cl(self, lengths: List[float]):
        """
        Setting the new cell lengths. Must always do all three at once.
        lenghts -- convention will be x, y, z.
        """
        assert len(lengths) == 3, f"Missing cell lengths. {len(lengths)=}"
        if self.cell_lengths is None:
            self.cell_lengths = np.array(lengths, dtype=np.float64)
        else:
            #print("Replacing old cell")
            self.cell_lengths = np.array(lengths, dtype=np.float64)
    
    def set_xyz_info(self, new_atoms: str, new_coordinates):
        # assert new_atoms.dtype == np.str_, f"new atoms is not of type str, {new_atoms.dtype=}"
        # assert new_coordinates.shape == (3,), "new coordinates are not cartesian"
        # assert new_coordinates.dtype == np.float64, "new coordinates are not floats"
        #print("over writing all atoms and coordinates")

        self.atoms = new_atoms
        self.xyz = new_coordinates
    
    def set_cut_offs(self, cut_offs: dict[dict]):
        """
        cutoff must be given as (for example):
                cutoffs = {
                        "Si": {"Si": 2.6, "O": 2.0},
                        "O" : {"Si": 2.0, "O": 1.7}
                        }
        """
        for key in cut_offs.keys():
            assert isinstance(key, str), f"The key {key} is not a string"
        for inner_dict in cut_offs.values():
            for key, val in inner_dict.items():
                assert isinstance(key, str), f"The key {key} in inner dictionary is not a string"
                assert isinstance(val, float) | isinstance(val, int), f"The value {val} in inner dictionary is not a number"
        
        if self.cut_offs is None:
            self.cut_offs = cut_offs
        else:
            print("Replacing old cut-offs")
            self.cut_offs = cut_offs

    def add_new_co(self, type_atom1: str, type_atom2: str, new_cut_off: float | int):
        """
        Adding new cut offs. We add the cut-off to both instances in dictionary becasue the should be "symmetric"
        """
        assert isinstance(type_atom1, str), f"type_atom1 is not str, {type(type_atom1)}"
        assert isinstance(type_atom2, str), f"type_atom2 is not str, {type(type_atom2)}"
        assert isinstance(new_cut_off, int)|isinstance(new_cut_off, float), f"type_atom2 is not int or float, {type(new_cut_off)}"
        if self.cut_offs is None:
            self.cut_offs = {}

        if type_atom1 not in self.cut_offs:
            print(f"Adding new atom of type {type_atom1}")
            self.cut_offs[type_atom1] = {}
        if type_atom2 not in self.cut_offs:
            print(f"Adding new atom of type {type_atom2}")
            self.cut_offs[type_atom2] = {}

        if type_atom2 not in self.cut_offs[type_atom1]:
            print(f"adding new cutoff for atom of type {type_atom2} in {type_atom1} of value {new_cut_off}")
            self.cut_offs[type_atom1][type_atom2] = new_cut_off
        else:
            print(f"overwriting old cutoff for atom of type {type_atom2} in {type_atom1} with value {new_cut_off}")
            self.cut_offs[type_atom1][type_atom2] = new_cut_off

        if type_atom1 not in self.cut_offs[type_atom2]:
            print(f"adding new cutoff for atom of type {type_atom1} in {type_atom2} of value {new_cut_off}")
            self.cut_offs[type_atom2][type_atom1] = new_cut_off
        else:
            print(f"overwriting old cutoff for atom of type {type_atom1} in {type_atom2} with value {new_cut_off}")
            self.cut_offs[type_atom2][type_atom1] = new_cut_off

    # getters
    def get_co(self, type_atom1: str, type_atom2: str):
        """Get the cut-off between two atoms"""
        assert type_atom1 in self.cut_offs, f"Trying to access the cut-offs for an atom that is not defined, {type_atom1=}"
        assert type_atom2 in self.cut_offs[type_atom1], f"Trying to access the cut-off for an atom does not have a cut-off with the first, {type_atom2=}"
        return self.cut_offs[type_atom1][type_atom2]
    
    # adding/removing/replacing atoms
    def add_new_atom(self, new_atom, new_coordiantes):
        # assert isinstance(new_atom, str), "new atom not a string"
        # assert isinstance(new_coordiantes, np.ndarray), "new coordinates not an array"
        assert new_coordiantes.dtype == np.float64, "new_coordinates not a float"
        assert len(new_coordiantes.flatten()) == 3, f"New coordinates not cartesian, {new_coordiantes.flatten()}"

        self.atoms = np.append(self.atoms, new_atom)
        self.xyz = np.vstack([self.xyz, new_coordiantes])

    def replace_atom(self, idx: int, new_atom: str, new_coordinates):
        # assert isinstance(new_atom, str), "new atom not a string"
        # assert isinstance(new_coordinates, np.ndarray), "new coordinates not an array"
        assert new_coordinates.dtype == np.float64, "new_coordinates not a float"
        assert len(new_coordinates) == 3, f"New coordinates not cartesian; {new_coordinates=}"

        self.atoms[idx] = new_atom
        self.xyz[idx] = new_coordinates

    def del_atom(self, idx_atom: int):
        assert idx_atom <= len(self.atoms), f"Trying to delete an atom that does not exist, {idx_atom}"
        self.atoms = np.delete(self.atoms, idx_atom)
        self.xyz = np.delete(self.xyz, idx_atom, axis=0)

    def wrap_coordinates(self):
        if self.cell_lengths is None:
            raise AssertionError("No cell lengths specified")
        self.xyz = self.xyz % self.cell_lengths

    def update_atom_types(self):
        self.atom_types = np.unique(self.atoms)

    # calculating values
    def mic(self, idx_atom1: int, idx_atoms2: List[int] | int):
        """calculate the coordiantes between two atoms according to the minimum image convention. Only support cuboidic unit cells."""
        assert np.all(idx_atom1 <= len(self.atoms)), f"Trying to access an atom which does not exist, {idx_atom1}"
        if isinstance(idx_atoms2, list) | isinstance(idx_atoms2, np.ndarray):
            for i in idx_atoms2:
                assert i <= len(self.atoms), "Trying to access an atom which does not exist"
        else:
            assert idx_atoms2 <= len(self.atoms), "Trying to access an atom which does not exist"
        
        try:
            num_atoms = len(idx_atoms2)
        except TypeError:
            num_atoms = 1
        
        xyz_atom1 = self.xyz[idx_atom1]
        xyz_atoms2 = self.xyz[idx_atoms2]
        xyz_atoms2 = xyz_atoms2.reshape(num_atoms, 3)
        output = _compiled_mic(xyz_atom1, xyz_atoms2, self.cell_lengths, num_atoms)
        
        # here becaue of a skill issue. Otherwise it itterates over x, y, z individually, not all together.
        match num_atoms:
            case 1:
                return output[0]
            case _:
                return output
    
    def give_nl(self, idx_target_atom: int):
        """Gives the indicies of all of the neighbors of an atoms."""
        # assert isinstance(idx_target_atom, int) | isinstance(idx_target_atom, np.int32), f"atom index not given as an int. {type(idx_target_atom)}"
        assert np.all(idx_target_atom <= len(self.atoms)), f"Trying to access an atom which does not exist, {idx_target_atom}"
        target_atom_type = self.atoms[idx_target_atom]

        prep_array: Tuple[List[float, float, float], List[float]] = ([], [])
        for idx, atom in enumerate(self.atoms):
            prep_array[0].append(self.mic(idx_target_atom, idx))
            prep_array[1].append(self.get_co(target_atom_type, atom))
            
        dists = np.linalg.norm(np.array(prep_array[0]), axis=1)
        neighbors = [idx for idx in np.where(np.logical_and(dists <= np.array(prep_array[1]), dists > 0.01))[0]]
        return np.array(neighbors, dtype=np.int32)
    
    def give_cn(self, idx_target_atom: int) -> int:
        # assert isinstance(idx_target_atom, int), f"idx_target_atom is not type int {type(idx_target_atom)}"
        assert idx_target_atom <= len(self.atoms), f"Trying to access an atom which does not exist, {idx_target_atom}"
        return len(self.give_nl(idx_target_atom))
    
    def give_bl(self, idx_atom1: int, idx_atom2: int) -> float:
        assert idx_atom1 <= len(self.atoms), f"Trying to access an atom which does not exist, {idx_atom1}"
        assert idx_atom2 <= len(self.atoms), f"Trying to access an atom which does not exist, {idx_atom2}"
        return np.linalg.norm(self.mic(idx_atom1, idx_atom2))

    def give_angle(self, idx_atom_A, idx_atom_B, idx_atom_C) -> float:
        xyz_BA = self.mic(idx_atom_A, idx_atom_B)
        xzy_BC = self.mic(idx_atom_C, idx_atom_B)
        return np.rad2deg(np.arccos(np.dot(xyz_BA, xzy_BC) / (np.linalg.norm(xyz_BA) * np.linalg.norm(xzy_BC))))

    def give_all_cn(self, IdxAtom_CN: bool = False, CN_IdxAtom: bool = False) -> Dict[str, Dict[int, int]] | Dict[str, Dict[int, List[int]]]:
        """
        Give the coordination of all the atoms in the strucutre. Inputs change the dictionary that is output

        IdxAtom_CN -- If True makes the output {atom_type: {idx: coordination}}
        CH_IdxAtom -- If True makes the output {atom_type: {coordination: [idx]}
        """
        assert IdxAtom_CN != CN_IdxAtom, "Haven't chosen an option for the output dictionary"
        xyz = self.xyz.copy()
        num_coords = len(xyz)
        dists = np.empty((num_coords, num_coords))
        co = np.empty((num_coords, num_coords))

        for i, xyz1 in enumerate(xyz):
            #Get the mic distances of each point
            mic_coords = _compiled_mic(xyz1, xyz, self.cell_lengths, num_coords)
            dist = np.linalg.norm(mic_coords, axis=1)
            dists[:, i] = dist

            #get the cutoff of everything
            atom_i = self.atoms[i]
            for j, _ in enumerate(xyz):
                if j < i:
                    co[j, i] = co[i, j]
                else:
                    atom_j = self.atoms[j]
                    co[j, i] = self.get_co(atom_i, atom_j)
            
        bond_mat = np.where((dists <= co) & (dists > 0.01), 1, 0)
        cn = np.sum(bond_mat, axis=1)

        output = {}
        
        if IdxAtom_CN:
            for atom_type in self.atom_types:
                output[atom_type] = {} # design choice. Gaurentees looking at desired atom. Will throw error if idx and atom type do not match
            for idx, (atom, coordination) in enumerate(zip(self.atoms, cn)):
                output[atom][int(idx)] = int(coordination)
            return output
       
        else:
            for atom_type in self.atom_types:
                output[atom_type] = {}
            for idx, (atom, coordination) in enumerate(zip(self.atoms, cn)):
                if coordination not in output[atom]:
                    output[atom][int(coordination)] = []
                output[atom][int(coordination)].append(int(idx))
            return output
    
    def probe_surface(self,  probe_width, probe_interval, from_top=True):
        idx_surface_atoms = _compiled_probe(atoms=self.atoms, xyz=self.xyz, cell_lengths=self.cell_lengths, 
                                            probe_width=probe_width, probe_interval=probe_interval,
                                            from_top = from_top)
        
        if from_top:
            self.idx_surface_atoms_top = np.array(list(set(idx_surface_atoms)), dtype=int)
        else:
            self.idx_surface_atoms_bot = np.array(list(set(idx_surface_atoms)), dtype=int)

    def get_surface_area(self, dims, radii:dict):
        # radius = np.empty(0)
        # for atom in self.atoms:
        #     radius = np.append(radius, radii[atom])
        
        # xyz = self.xyz
        # xyz[:, 2] += 5
        # scalar_field = _compiled_make_mesh(xyz, radius, dims, self.cell_lengths)
        # verts, faces, normals, values = measure.marching_cubes(scalar_field, 0, 
        #                                 spacing=(self.cell_lengths[0]/dims[0], self.cell_lengths[1]/dims[1], self.cell_lengths[2]/dims[2]))
        
        # area = 0.0
        # for face in faces:
        #     v1, v2, v3 = verts[face[0]], verts[face[1]], verts[face[2]]
        #     triangle_area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        #     area += triangle_area

        # self.total_surface_area = area
        print("not implemented")

    def probe_surface_rays(self, max_iter, from_top=True):
        if from_top:
            z = np.max(self.xyz[:,2]) + 20
            print("Starting probing top", flush=True)
        else:
            print("Starting probing bottom", flush=True)
            z = np.min(self.xyz[:,2]) - 20
        probe_points = np.array([[0, 0, z],
                                 [self.cell_lengths[0]*0.25, self.cell_lengths[1]*0.25, z],
                                 [self.cell_lengths[0]*0.50, self.cell_lengths[1]*0.50, z],
                                 [self.cell_lengths[0]*0.75, self.cell_lengths[1]*0.75, z]]).reshape(-1, 3)

        cut_off_dict = {"Si": 2.1+1.4, "O":1.52+1.4, "H":1.1+1.4}
        cut_offs = np.empty(0)
        for atom in self.atoms:
            cut_offs = np.append(cut_offs, cut_off_dict[atom])

        all_hit_atoms = np.empty(0)
        for current_iter in range(max_iter):
            hit_atoms = []
            next_iter_probe = np.empty((0,3))
            for probe_point in probe_points:
                for (atom, xyz) in zip(self.atoms, self.xyz):
                    for t in np.linspace(0,1,150):
                        point_on_line = probe_point + t*(xyz-probe_point)
                        mic = _compiled_mic(point_on_line, self.xyz, self.cell_lengths, len(self.xyz))
                        dists = np.linalg.norm(mic, axis=1)
                        hits = np.where(dists <= cut_offs)[0]
                        if len(hits) == 0:
                            continue
                        else:
                            for i in hits:
                                move_t = np.random.uniform(0.001,0.1)
                                next_iter_probe = np.vstack((next_iter_probe, probe_point + (t-move_t)*(xyz-probe_point)))
                                hit_atoms.append(i)
                                all_hit_atoms = np.append(all_hit_atoms, i)
                            break
            all_hit_atoms = np.unique(all_hit_atoms).astype(int)
            idx_probe_points = np.random.choice(len(next_iter_probe), len(all_hit_atoms), replace=False)
            mask_probe_points = np.zeros(len(hit_atoms), dtype=bool)
            mask_probe_points[idx_probe_points] = True
            probe_points = next_iter_probe[mask_probe_points]

            # print(len(all_hit_atoms), flush=True)
            # if from_top:
            #     name = f"{current_iter+1}_iter_top.xyz"
            # else:
            #     name = f"{current_iter+1}_iter_bot.xyz"

            # with open(name, "w") as f:
            #     f.write(f"{len(self.atoms)}\n")
            #     f.write("\n")
            #     for idx, (atom, xyz) in enumerate(zip(self.atoms, self.xyz)):
            #         if idx in all_hit_atoms:
            #             if atom == "Si":
            #                 f.write("P ")
            #             if atom == "O":
            #                 f.write("F ")
            #             if atom == "H":
            #                 f.write("He ")
            #         else:
            #             f.write(f"{atom} ")
            #         for i in xyz:
            #             f.write(f"{i} ")
            #         f.write("\n")
            # print("iter done")
        if from_top:
            self.idx_surface_atoms_top = all_hit_atoms
        else:
            self.idx_surface_atoms_bot = all_hit_atoms

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

@nb.njit
def _compiled_probe(atoms, xyz, cell_lengths, probe_width, probe_interval, from_top):
    num_probes_x = int(np.ceil(cell_lengths[0] / probe_interval) + 1)
    num_probes_y = int(np.ceil(cell_lengths[1] / probe_interval) + 1)
    x_vals = np.linspace(0, cell_lengths[0], num_probes_x)
    y_vals = np.linspace(0, cell_lengths[1], num_probes_y)
    
    all_idx = np.arange(0, len(atoms))
    z_av = np.mean(xyz[:, 2])

    surface_atoms = []
    for x in x_vals:
        for y in y_vals:
            probe_point = np.array([x, y, z_av])
            xyz_mic = _compiled_mic(probe_point, xyz, cell_lengths, len(atoms))
            abs_xyz_mic = np.abs(xyz_mic)

            mask_x = abs_xyz_mic[:, 0] <= probe_width 
            mask_y = abs_xyz_mic[:, 1] <= probe_width
            idx_xyz = all_idx[mask_x & mask_y]
            filtered_xyz = xyz_mic[mask_x & mask_y]
            
            idx_Si = [idx for idx in idx_xyz if atoms[idx] == "Si"]
            xyz_Si = [xyz_mic[idx] for idx in idx_Si]

            idx_O = [idx for idx in idx_xyz if atoms[idx] == "O"]
            xyz_O = [xyz_mic[idx] for idx in idx_O]

            if filtered_xyz.shape[0] != 0:
                top = np.max(filtered_xyz[:, 2]) + probe_width
                bottom = np.min(filtered_xyz[:, 2]) - probe_width
                num_points_travel = int(np.ceil((top - bottom)/0.05) + 1)
                
                if from_top:
                    probe_travel = np.linspace(top, bottom, num_points_travel)
                else:
                    probe_travel = np.linspace(bottom, top, num_points_travel)
                probe_xyz = np.column_stack((np.full_like(probe_travel, 0), np.full_like(probe_travel, 0), probe_travel))

                hit = False
                for probe_pos in probe_xyz:
                    if hit:
                        break
                    for i, Si in zip(idx_Si, xyz_Si):
                        dist = np.sqrt((probe_pos[0] - Si[0])**2 + (probe_pos[1] - Si[1])**2 + (probe_pos[2] - Si[2])**2)
                        if dist <= 2.44:
                            surface_atoms.append(i)
                            hit = True
                            break
                    
                    if hit:
                        break
                    for i, O in zip(idx_O, xyz_O):
                        dist = np.sqrt((probe_pos[0] - O[0])**2 + (probe_pos[1] - O[1])**2 + (probe_pos[2] - O[2])**2)
                        if dist <= 2.16:
                            surface_atoms.append(i)
                            hit = True
                            break

    return surface_atoms

@nb.njit
def _compiled_make_mesh(xyz, radii, dims, cell_lengths):
    x_arr = np.linspace(0, cell_lengths[0], dims[0])
    y_arr = np.linspace(0,  cell_lengths[1], dims[1])
    z_arr = np.linspace(0,  cell_lengths[2], dims[2])
    scalar_field = np.empty((dims[0], dims[1], dims[2]), dtype=np.float32)
    
    for ix, x in enumerate(x_arr):
        for iy, y in enumerate(y_arr):
            for iz, z in enumerate(z_arr):
                mic = _compiled_mic(np.array([x,y,z]), xyz, cell_lengths, len(xyz))
                vals = ((mic[:, 0])**2 + (mic[:, 1])**2 + (mic[:, 2])**2 - radii**2)/(radii**2)
                scalar_field[ix, iy, iz] = np.min(vals)
    return scalar_field