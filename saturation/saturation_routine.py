from saturation.SaturationLogic import SaturationChildClass, Generation
from glob import glob
from ase.io import read
from interfaces.MACE_interface import MACEInterface
import os
os.environ["OMP_NUM_THREADS"] = "16"

def saturate_structure(filename: str,
                       mace_model_path: str,
                       device: str = "mps"):
    print("started", flush=True)

    init_struc = read(filename)

   #preoptimizing structure
    mace = MACEInterface(
        mace_model_path=mace_model_path,
        device=device
    )

    energy0, atoms = mace.optimize(init_struc, max_steps=150)
    atoms.write("preopt.cif", format="cif")
    atoms.set_calculator(None)


    saturation = SaturationChildClass(atoms)

    max_num_children = 8
    number_new_parents = 1
    next_gen = [saturation]

    continue_sat = False
    for parent in next_gen:
        _, under_cn = parent.check_cn()
        for list_under_cn in under_cn.values():
            if len(list_under_cn) > 0:
                continue_sat = True
    while continue_sat:
        new_gen = Generation(
            max_children=max_num_children,
            mace_model_path=mace_model_path,
            device=device
        )
        for parent in next_gen:
            new_gen.add_parent(parent)

        print("saturating", flush=True)
        new_gen.make_gen_sat(max_steps=40)
        new_gen.dump_children()
        next_gen = new_gen.new_parents(number_new_parents)

        continue_sat = False  # set the conintue flag to False for the moment
        # go through all children and if there is one that is fully saturated stop
        for parent in next_gen:
            _, under_cn = parent.check_cn()
            for list_under_cn in under_cn.values():
                if len(list_under_cn) > 0:
                    continue_sat = True  # Set it to True if there are any remaining under-saturated atoms

    for parent in next_gen:
        parent.atoms.write("saturated.cif", format="cif")

    print("done", flush=True)


if __name__ == "__main__":
    saturate_structure(filename="../growth/POSCAR_final_struc",
                       mace_model_path="/MACE-models/2024-01-07-mace-128-L2_epoch-199.model",
                       device="mps")
