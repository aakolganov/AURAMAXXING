import os
from ase.io import read
from saturation.SaturationLogic import SaturationChildClass, Generation
from interfaces.MACE_interface import MACEInterface


def generate_initial_population(
    file_path: str,
    mace_model_path: str,
    device: str = "mps",
    population_size: int = 8,
    opt_steps: int = 100,
):
    """
    Generate initial population of saturated structures for further "proper" GC sampling"
    :param file_path: filename of the initial structure
    :param mace_model_path: path to the MACE model
    :param device: device to calculate on
    :param population_size: self-explanatory
    :param opt_steps: maximum of the optimization steps
    :return:
    """
    # 1)  Calculating initial structures
    init_atoms = read(file_path, format="vasp")
    mace = MACEInterface(mace_model_path=mace_model_path, device=device)
    _, preopt = mace.optimize(init_atoms, max_steps=opt_steps)
    preopt.set_calculator(None)
    # saving the initial file
    preopt.write("preoptimized.cif", format="cif")

    # 2) Constructing the first gen and saturate:

    parent = SaturationChildClass(preopt)
    gen = Generation(
        max_children=population_size,
        mace_model_path=mace_model_path,
        device=device
    )
    gen.add_parent(parent)

    # 3) Optiimizing the first gen:
    gen.make_gen_sat(max_steps=opt_steps)
