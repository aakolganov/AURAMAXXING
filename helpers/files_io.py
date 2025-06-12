from ase.io import read, write

def save_traj(struc, traj_path, step):
    """
    Append the current atomic positions to an XYZ trajectory file.

    Parameters
    ----------
    struc : AmorphousStrucASE
        The current  structure (ASE wrapper) whose atoms are to be written.
    traj_path : str
        Path to the trajectory file (XYZ format). If it does not exist, it will be created.
    step : int
        An integer counter to record in the comment field of the XYZ frame, e.g. "step_{step}".
    """
    struc.atoms.write(traj_path, format="xyz", append=True, comment=f"step_{step}")


def add_dump_to_traj(dump_path: str = "LAMMPS/dump.xyz", traj_file: str = "growth_trajectory.xyz"):
    """
    Add the content of LAMMPS dump to the trajectory file.
    """
    frames = read(dump_path, index=":")
    for frame in frames:
        write(traj_file, frame, format="xyz", append=True)
