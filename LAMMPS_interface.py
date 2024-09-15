import numpy as np
import numba as nb
import os
import subprocess
from typing import Dict
import itertools

class Lammps_interface:
    def __init__(self):
        self.atoms = None
        self.xyz_coords = None
        self.dims: Dict[str, float|None] = {
            "xlo": None, 
            "xhi": None, 
            "ylo": None, 
            "yhi": None, 
            "zlo": None, 
            "zhi": None
        }

        self.atom_masses: Dict[str, float] = {
            "Si": 28.085,
            "O": 15.9999
        }
        self.atom_charges: Dict[str, float] = {
            "Si": 2.4,
            "O": -1.2
        }
    
    def write_xyz(self, file_name):
        assert len(self.atoms) == len(self.xyz_coords)
        with open("{}.xyz".format(file_name), "w") as f:
            f.write(f"{len(self.atoms)}\n\n")
            for atom, xyz in zip(self.atoms, self.xyz_coords):
                f.write(f"{atom}")
                for i in xyz:
                    f.write(f" {i}")
                f.write("\n")

    def opt_struc(self, type_opt = "minimize", steps = None, removal = None, max_num_rings=None, start_T=None, final_T=None):
        """type opt wither 'minimize' or 'anneal' or 'final'. If 'anneal' or 'final' need to specify the number of steps
        removals: None removes nothing, rings removes 2MR, over-coord removes only over-coordianted atoms, 
        everything removes over-coordianted and rings 
        """
        try:
            os.mkdir("LAMMPS")
        except:
            ...
        os.chdir("LAMMPS")
        
        self._write_data_file()

        match type_opt:
            case 'minimize':
                self._write_in_file_minimize()
            case 'anneal':
                assert steps is not None, "Need to specify number of steps"
                self._write_in_file_anneal(steps, start_T, final_T)
            case 'final':
                assert steps is not None, "Need to specify number of steps"
                self._write_in_file_anneal(steps, start_T, final_T)
            case _:
                raise ValueError
        
        self._run()
        self._read_final_struc_data()
        self.write_xyz("final_struc_convert")

        match removal:
            case "rings":
                self.remove_2MR(max_num_rings)
                self.write_xyz("yeet_2MR")
            case "over-coord":
                self.remove_over_coord({"Si": 4, "O": 2}, 2.0)
                self.write_xyz("yeet_over")
            case "everything":
                self.remove_over_coord({"Si": 4, "O": 2}, 2.0)
                self.write_xyz("yeet_over")
                self.remove_2MR(max_num_rings)
                self.write_xyz("yeet_2MR")
            case None:
                pass
            case _:
                raise ValueError

        os.chdir("..")
        return self.atoms, self.xyz_coords, self.dims
    
    def add_structure(self, atoms, xyz_coords):
        self.atoms = np.array(atoms, dtype=str)
        self.xyz_coords = np.array(xyz_coords, dtype=float)
        
    def add_dims(self, xlo: float = None, xhi: float = None, ylo: float = None, yhi: float = None, zlo: float = None, zhi: float = None):
       args = locals()
       for key, val in args.items():
            if key != "self" and val is not None:
               self.dims[key] = float(val)

    def add_atom_mass(self, atom_type: str, mass: int|float):
        assert mass >= 0, "mass cannot be negative"
        self.atom_masses[atom_type] = mass
    
    def add_atom_charge(self, atom_type, charge: int|float):
        self.atom_charges[atom_type] = charge

    def _write_data_file(self):
        atom_numbering: Dict[str, int] = {"Si": 1, "O":2}
        assert len(self.xyz_coords) == len(self.atoms), "different number of cooridnates and atoms"
        for atom in self.atoms:
            assert atom in self.atom_masses, f"atom {atom} not know in masses"
            assert atom in self.atom_charges, f"atom {atom} not know in charges"
        self.cl = [self.dims["xhi"] - self.dims["xlo"],
                   self.dims["yhi"] - self.dims["ylo"],
                   self.dims["zhi"] - self.dims["zlo"]]
        with open("structure.data", "w") as f:
            f.write("Generated through interface; ")
            for atom_type, atom_number in atom_numbering.items():
                f.write(f"{atom_type} is {atom_number}  ")
            f.write("\n\n")

            f.write("{:d} atoms\n{:d} atom types\n\n".format(len(self.atoms), len(atom_numbering.keys())))

            for key, val in self.dims.items():
                assert val is not None, f"{key} is not specified"
            f.write("{} {} xlo xhi\n{} {} ylo yhi\n{} {} zlo zhi\n\n".format(
                self.dims["xlo"],
                self.dims["xhi"],
                self.dims["ylo"],
                self.dims["yhi"],
                self.dims["zlo"],
                self.dims["zhi"]
                ))
            
            f.write("Masses\n\n")
            for atom_type, atom_number in atom_numbering.items():
                f.write(f"{atom_number} {self.atom_masses[atom_type]}\n")
            f.write("\n")

            f.write("Atoms # charge\n\n")
            for i, (atom, xyz) in enumerate(zip(self.atoms, self.xyz_coords)):
                f.write(f"{i+1} {atom_numbering[atom]} {self.atom_charges[atom]} {xyz[0]} {xyz[1]} {xyz[2]} 0 0 0\n")
            f.write("\n")

            f.write("Velocities\n\n")
            for i, _ in enumerate(self.atoms):
                f.write(f"{i+1} 0 0 0\n")

    def _write_in_file_minimize(self):
        with open("instruction.in", "w") as f:
            f.write(f"""units           real
atom_style      charge
boundary        p p p

read_data       structure.data

pair_style      hybrid/overlay buck/coul/long 5.5 8.0 lj/cut 1.2
kspace_style    ewald 1.0e-4

pair_coeff      1   1   buck/coul/long  0.0 0.2 0.0 #SI-SI
pair_coeff      2   2   buck/coul/long  32026.68173 0.362318841 4035.698637 #SI-SI
pair_coeff      1   2   buck/coul/long  415187.07650 0.205204815 3079.540161 #SI-O
pair_coeff      1   1   lj/cut  0.0 0.0 #Si-Si
pair_coeff      2   2   lj/cut  59.95595939 1.6 1.6 #O-O 
pair_coeff      1   2   lj/cut  46.11996875 1.2 1.2 #Si-O

set             type 1 charge 2.4  # Si charge
set             type 2 charge -1.2 # O charge

neighbor        2.0 bin
neigh_modify    every 2 delay 0 check yes
group mobile    type 1 2
timestep        0.5
run_style       verlet

thermo_modify   lost warn
thermo_style    custom step temp press time vol density etotal lx ly lz
thermo          10

dump            xyz  mobile xyz 1 dump.xyz
dump_modify     xyz  element Si O

minimize        0  5.0e-1  1000  1000000

write_data      final_struc.data""")
            
    def _write_in_file_anneal(self, steps: int, start_T, final_T):
        with open("instruction.in", "w") as f:
            f.write(f"""units           real
atom_style      charge
boundary        p p p

read_data       structure.data

pair_style      hybrid/overlay buck/coul/long 5.5 8.0 lj/cut 1.2
kspace_style    ewald 1.0e-4

pair_coeff      1   1   buck/coul/long  0.0 0.2 0.0 #SI-SI
pair_coeff      2   2   buck/coul/long  32026.68173 0.362318841 4035.698637 #SI-SI
pair_coeff      1   2   buck/coul/long  415187.07650 0.205204815 3079.540161 #SI-O
pair_coeff      1   1   lj/cut  0.0 0.0 #Si-Si
pair_coeff      2   2   lj/cut  59.95595939 1.6 1.6 #O-O 
pair_coeff      1   2   lj/cut  46.11996875 1.2 1.2 #Si-O

set             type 1 charge 2.4  # Si charge
set             type 2 charge -1.2 # O charge

neighbor        2.0 bin
neigh_modify    every 2 delay 0 check yes
group mobile    type 1 2
timestep        0.5
run_style       verlet

thermo_modify   lost warn
thermo_style    custom step temp press time vol density etotal lx ly lz
thermo          10

dump            xyz  mobile xyz {steps//50} dump.xyz
dump_modify     xyz  element Si O

minimize        1.0e-2  1.0e-3  1000  10000
velocity        mobile create {start_T} {np.random.randint(10000)} dist gaussian
fix             1 mobile nve
#fix             3 mobile press/berendsen z 1.0 1.0 100 modulus 360000
fix             4 mobile temp/berendsen {start_T} {start_T} 100
run             {4*int(steps)}
                    
fix             4 mobile temp/berendsen {start_T} {final_T} 100
run             {int(steps)} 
 
write_data      final_struc.data""")
    
    
    @staticmethod        
    def _run():
        command = ['lmp', '-in', "instruction.in"]
        with open("lmp.out", "w") as outfile:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for stdout_line in iter(process.stdout.readline, ''):
                outfile.write(stdout_line)
                outfile.flush()  # Ensure output is written immediately

            for stderr_line in iter(process.stderr.readline, ''):
                outfile.write(stderr_line)
                outfile.flush()  # Ensure error output is written immediately

            # Wait for the process to complete
            process.stdout.close()
            process.stderr.close()
            process.wait()
    
    def _read_final_struc_data(self):
        atoms = np.empty(0)
        xyz_coords = np.array([]).reshape((0, 3))

        with open("final_struc.data") as f:
            lines = f.readlines()

        std_dict = {"1": "Si", "2": "O"}
        reading_file = False
        for line in lines:
            if "xlo xhi" in line:
                line = line.strip().split()
                self.dims["xlo"] = float(line[0])
                self.dims["xhi"] = float(line[1])
            if "ylo yhi" in line:
                line = line.strip().split()
                self.dims["ylo"] = float(line[0])
                self.dims["yhi"] = float(line[1])
            if "zlo zhi" in line:
                line = line.strip().split()
                self.dims["zlo"] = float(line[0])
                self.dims["zhi"] = float(line[1])
            
            if "Velocities" in line:
                reading_file = False
                
            if reading_file:
                line = line.strip().split()
                if len(line) != 0:
                    atoms = np.append(atoms, std_dict[line[1]])
                    xyz_coords = np.vstack((xyz_coords, np.array([line[3], line[4], line[5]], dtype=float)))

            if "Atoms # charge" in line:
                reading_file = True
            
        self.atoms = atoms
        self.xyz_coords = xyz_coords

    def determine_fragments(self):
        num_coords = len(self.xyz_coords)
        
        dist_matrix = np.empty((num_coords, num_coords))
        for i, xyz in enumerate(self.xyz_coords):
            mic_coords = mic(xyz, self.xyz_coords, nb.typed.List(self.cl), num_coords)
            dist = np.linalg.norm(mic_coords, axis=1)
            dist_matrix[:, i] = dist
    
        # Build the adjacency matrix (connected if distance <= max_distance)
        adjacency_matrix = dist_matrix <= 2.0
        
        # Number of points
        num_points = len(self.xyz_coords)
        
        # Visited array to track which points have been visited
        visited = [False] * num_points
        components = []
        
        # DFS function to traverse connected points
        def dfs(node, component):
            visited[node] = True
            component.append(node)
            for neighbor, connected in enumerate(adjacency_matrix[node]):
                if connected and not visited[neighbor]:
                    dfs(neighbor, component)
        
        # Find all connected components using DFS
        for i in range(num_points):
            if not visited[i]:
                component = []
                dfs(i, component)
                components.append(component)
        
        return components


    def find_2MR(self, tolerance):
        paths = _find_2MR(self.atoms, self.xyz_coords, self.cl, tolerance)
        return paths
    
    def remove_2MR(self, max_allowed):
        to_delete = []

        idx_2MRs = self.find_2MR(2.0)
        number_2MR = len(idx_2MRs)
        print(f"{number_2MR=}")
        if number_2MR > max_allowed:
            combinations = list(itertools.combinations(idx_2MRs, number_2MR-max_allowed))
            chosen_combination = combinations[0]
            for idx_2MR in chosen_combination:
                possible_delete = []
                for idx in idx_2MR:
                    if self.atoms[idx] == "O":
                        possible_delete.append(idx)
                to_delete.append(np.random.choice(possible_delete))
        
            to_delete = np.unique(to_delete)
            self.atoms = np.delete(self.atoms, np.array(to_delete, dtype=int))
            self.xyz_coords = np.delete(self.xyz_coords, np.array(to_delete, dtype=int), axis=0)
        else:
            pass

    def remove_over_coord(self, wanted_CN, tolerance):
        num_coords = len(self.xyz_coords)
        dist_matrix = np.empty((num_coords, num_coords))
        for i, xyz in enumerate(self.xyz_coords):
            mic_coords = mic(xyz, self.xyz_coords, nb.typed.List(self.cl), num_coords)
            dist = np.linalg.norm(mic_coords, axis=1)
            dist_matrix[:, i] = dist
        adjacency_matrix = np.where((dist_matrix <= tolerance) & (dist_matrix > 0.01), 1, 0)

        to_delete = []
        CN = np.sum(adjacency_matrix, axis=1)
        for i, val in enumerate(CN):
            if val > wanted_CN[self.atoms[i]]:
                to_delete.append(i)
        
        to_delete = np.unique(to_delete)
        self.atoms = np.delete(self.atoms, np.array(to_delete, dtype=int))
        self.xyz_coords = np.delete(self.xyz_coords, np.array(to_delete, dtype=int), axis=0)

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

def _find_paths(neighbors, current_node, depth, current_path, all_paths):
    if depth == 0:
        all_paths.append(current_path)
        return

    for neighbor in neighbors.get(current_node, []):
        _find_paths(neighbors, neighbor, depth - 1, current_path + [neighbor], all_paths)

def _get_all_paths(neighbors, start_node, max_depth):
    all_paths = []
    for neighbor in neighbors.get(start_node, []):
        _find_paths(neighbors, neighbor, max_depth - 1, [neighbor], all_paths)
    return all_paths


def _find_n_MR(all_Si_idx, neighbors: dict, num_Si_in_ring: int):
    depth = 2 * num_Si_in_ring
    all_paths = []
    for start_node in neighbors.keys():
        if start_node in all_Si_idx:
            # absolute jank but whatever
            all_paths_for_starting_node = _get_all_paths(neighbors, start_node, depth)

            # removing all of the paths which visited the same atom twice
            filter_1 = []
            for idx, inner_list in enumerate(all_paths_for_starting_node):
                if len(inner_list) == len(set(inner_list)):
                    filter_1.append(idx)
            all_paths_for_starting_node = [inner_list for idx, inner_list in enumerate(all_paths_for_starting_node)
                                           if idx in filter_1]

            # removing all of the paths which do not end at the target atom.
            filter_2 = []
            for idx, inner_list in enumerate(all_paths_for_starting_node):
                if inner_list[-1] == start_node:
                    filter_2.append(idx)
            all_paths_for_starting_node = [inner_list for idx, inner_list in enumerate(all_paths_for_starting_node)
                                           if idx in filter_2]
            for inner_list in all_paths_for_starting_node:
                if start_node in inner_list:
                    all_paths.append(inner_list)

    all_paths_sorted = [sorted(inner_list) for inner_list in all_paths]
    all_unique_paths = list(map(list, set(map(tuple, all_paths_sorted))))
    return all_unique_paths


def _find_2MR(atoms, xyz_coords, cl, tolerance: float):
    Si_idx = np.where(atoms == "Si")[0]
    
    num_coords = len(xyz_coords)
    dist_matrix = np.empty((num_coords, num_coords))
    for i, xyz in enumerate(xyz_coords):
        mic_coords = mic(xyz, xyz_coords, nb.typed.List(cl), num_coords)
        dist = np.linalg.norm(mic_coords, axis=1)
        dist_matrix[:, i] = dist

    # Build the adjacency matrix (connected if distance <= max_distance)
    adjacency_matrix = dist_matrix <= tolerance

    neighbors = {}
    for idx, row in enumerate(adjacency_matrix):
        temp = np.where(row == 1)[0]
        neighbors[idx] = temp

    paths = _find_n_MR(Si_idx, neighbors, 2)
    return paths