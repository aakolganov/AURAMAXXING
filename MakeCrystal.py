from crystal_class import CrystalStruc
from LAMMPS_interface import Lammps_interface

import numpy as np


def initialize_crystal_class():
    ...

def generate_crystal(total_desired_atoms, alpha, n_m):
    DIMS = {
        "XLO": 0, 
        "XHI": 21.0,
        "YLO": 0,
        "YHI": 21.0,
        "ZLO": 0,
        "ZHI": 50.0
    }

    crystal_struc = CrystalStruc(atoms=np.empty(0), xyz_coordinates=np.empty((0, 3)))
    crystal_struc.set_cut_offs({
        'Si': {'Si': 2.2, 'O': 2.0, 'H': 1.0},
        'O':  {'Si': 2.0, 'O': 1.8, 'H': 1.3},
        'H':  {'Si': 1.0, 'O': 1.3, 'H': 1.0}
    })
    crystal_struc.add_dims(xlo = DIMS["XLO"], xhi = DIMS["XHI"],
                           ylo = DIMS["YLO"], yhi = DIMS["YHI"],
                           zlo = DIMS["ZLO"], zhi = DIMS["ZHI"])
    
    crystal_struc.set_limits(alpha=alpha, n_m=n_m)
    crystal_struc.set_max_CN({"Si": 4, "O": 2})

    number_write = 0
    TOTAL_DESIRED_ATOM, new_total_atoms = total_desired_atoms, 0
    hit_final = False

    while new_total_atoms < TOTAL_DESIRED_ATOM:
        idx_connect_to = crystal_struc.set_i()

        current_number_atoms = len(crystal_struc.atoms)
        if current_number_atoms == 0:
            atom_to_add = "Si"
            made_placement = crystal_struc.place_atom_random(atom_to_add)

            number_write += 1
            crystal_struc.write_xyz_file(f"structure_{number_write}")
            continue
        else:
            atom_to_add = "O" if crystal_struc.atoms[idx_connect_to] == "Si" else "Si"
            
            MAX_ITER, current_iter = 100, 0
            made_placement = False
            while current_iter <= MAX_ITER and not made_placement:
                current_iter += 1
                made_placement = crystal_struc.place_atom_random(atom_to_add, idx_connect_to=idx_connect_to, limits=crystal_struc.limits)
                idx_connect_to = crystal_struc.set_i()

            new_total_atoms = len(crystal_struc.atoms)
            if made_placement:
                print(f"{new_total_atoms}", flush=True)
                crystal_struc.wrap_coordinates()

                number_write += 1
                crystal_struc.write_xyz_file(f"structure_{number_write}")

            else:
                print("giving structure a kick")
                if hit_final:
                    max_steps, start_T, final_T = 250, 500, 1000
                    FF = "BKS"
                    removal = "rings"
                else:
                    max_steps, start_T, final_T = 500, 298, 4000
                    FF = "BKS"
                    removal = "rings"
                
                struc_optimizer = Lammps_interface(crystal_struc)
                _ = struc_optimizer.opt_struc("anneal", steps=1000, start_T=298, final_T=298, FF="BKS")
                new_atoms, new_coords, new_dims = struc_optimizer.opt_struc("anneal", 
                                                                            steps=max_steps, start_T=start_T, final_T=final_T,
                                                                            removal=removal, 
                                                                            max_num_rings=5, max_remove_over=100,
                                                                            FF=FF)
                crystal_struc.set_xyz_info(new_atoms, new_coords)
                crystal_struc.dims = new_dims

                if not hit_final:
                    crystal_struc.slice(raise_by=1.0)
                number_write += 1
                crystal_struc.write_xyz_file(f"structure_{number_write}")

        current_number_atoms = len(crystal_struc.atoms)
        if new_total_atoms == TOTAL_DESIRED_ATOM:
            
            print("final opt")
            struc_optimizer = Lammps_interface(crystal_struc)
            
            if hit_final:
                _ = struc_optimizer.opt_struc("anneal", steps=250, start_T=500, final_T=2000, FF="BKS")
                new_atoms, new_coords, new_dims = struc_optimizer.opt_struc("final", 
                                               steps=1000, start_T=298, final_T=298, 
                                               removal="rings", 
                                               max_num_rings=3, max_remove_over=8,
                                               FF="BKS")
                # new_atoms, new_coords, new_dims = struc_optimizer.opt_struc("anneal", steps=150, start_T=1500, final_T=1500, FF="BKS")
            else:
                _ = struc_optimizer.opt_struc("anneal", steps=250, start_T=298, final_T=298, FF="BKS")
                new_atoms, new_coords, new_dims = struc_optimizer.opt_struc("anneal", 
                                                                            steps=750, start_T=298, final_T=1000, 
                                                                            removal="rings", 
                                                                            max_num_rings=3, max_remove_over=4,
                                                                            FF="BKS")
            
            hit_final = True

            crystal_struc.set_xyz_info(new_atoms, new_coords)
            crystal_struc.dims = new_dims
            new_total_atoms = len(crystal_struc.atoms)
    
    number_write += 1
    crystal_struc.write_xyz_file(f"structure_{number_write}")
    crystal_struc.write_xyz_file(f"final_structure")

    return crystal_struc

if __name__ == "__main__":
    generate_crystal(315, 0.03, 4)