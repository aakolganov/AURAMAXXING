import os
import subprocess
from typing import Dict, Optional
from ase.io import read
from base.AmorphousStrucASE import AmorphousStrucASE
from pathlib import Path
from contextlib import contextmanager
import textwrap
from ase.data import atomic_numbers
from ase.constraints import FixAtoms
from ase.io.lammpsdata import write_lammps_data

from default_constants import default_masses, default_charges, ev_to_kcal
@contextmanager
def cd(path: Path):
    prev = Path.cwd()
    path.mkdir(exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class LammpsInterface:
    """
    Interface for interacting with LAMMPS for molecular dynamics and structure optimization.

    :ivar num_final: Keeps track of the number of 'final' optimization calls.
    :type num_final: int
    :ivar atoms: Copy of the atomic structure from the input `AmorphousStrucASE`.
    :type atoms: Atoms
    :ivar rng: Random number generator object inherited from the input structure.
    :type rng: Random
    :ivar atom_masses: Dictionary mapping atomic symbols to their masses, customizable by input or defaults.
    :type atom_masses: Dict[str, float]
    :ivar atom_charges: Dictionary mapping atomic symbols to their charges, customizable by input or defaults.
    :type atom_charges: Dict[str, float]
    :ivar properties: Dictionary storing calculated properties (e.g., 'pe' for potential energy).
    :type properties: Dict[str, float]
    :ivar lammps_cmd: Path to the LAMMPS executable. Defaults to a preconfigured path if not provided.
    :type lammps_cmd: str
    :ivar _bks_params: Dictionary containing BKS force field parameters for different atomic interactions.
    :type _bks_params: Dict[Tuple[str, str], Tuple[float, float, float]]
    :ivar _bks_lj: Dictionary containing LJ potential parameters for specific interactions.
    :type _bks_lj: Dict[Tuple[str, str], Tuple[float, float, float]]
    :ivar _bks_charges: Default dictionary of atomic charges for the BKS force field.
    :type _bks_charges: Dict[str, float]
    """
    num_final = 1
    def __init__(self, struc: AmorphousStrucASE,
                 masses: Optional[Dict[str, float]] = None,
                 charges: Optional[Dict[str, float]] = None,
                 lammps_cmd: Optional[str] = None):
        self.atoms = struc.atoms.copy()
        self.rng = struc.rng
        self.atom_masses = masses or default_masses.copy()
        self.atom_charges = charges or default_charges.copy()
        self.properties: Dict[str, float] = {}
        self.lammps_cmd = lammps_cmd or "/Users/akolganov/.local/bin/lmp"
        if self.lammps_cmd is None:
            raise FileNotFoundError("LAMMPS executable 'lmp' not found on PATH.")

    _bks_params = {
        # buck/coul/long: (A_ij, eV; b_ij, A^-1, c_ij, eV*A^6)
        ("Si", "Si"): (    834.40,        1/0.29,      0.0),
        ("O",  "O" ): (1388.7730, 2.76000,175.0000),
        ("Si", "O" ): (45296.72 ,1/0.161,46.1395),
        ("Si", "Al"): (     646.67,        1/0.120,      0.0),
        ("O",  "Al"): (     28287.00 ,        1/0.172,      34.7600),
        ("Al", "Al"): (    351.94,        1/0.360 ,      0.0),
    }
    # lj/cut: (epsilon, sigma, sigma)
    _bks_lj = {
        ("Si", "Si"): ( 0.0,       0.0, 0.0),
        ("O",  "O" ): (2.6, 1.6, 1.6),
        ("Si", "O" ): (2.0, 1.2, 1.2),
        ("Si", "Al"): (0.0, 0.0, 0.0),
        ("O",  "Al"): (2.3,    1.4, 1.4),
        ("Al", "Al"): ( 0.0,       0.0, 0.0),
    }
    # charges
    _bks_charges = {
        "Si":  2.4,
        "O" : -1.2,
        "Al":  1.8,
    }

    @property
    def xyz_coords(self):
        return self.atoms.get_positions()

    @property
    def _element_list(self) -> str:
        # collect unique elements in the order of increasing atomic numbers
        syms = sorted(set(self.atoms.get_chemical_symbols()),
                      key=lambda s: atomic_numbers[s])
        return " ".join(syms)

    def _execute_and_save(self, label:str):
        # wrapper to run lammps and save the structure
        self._run()
        self._read_final_struc_data()
        self.atoms.write(f"{label}.xyz", format="xyz")


    def _get_frozen_ids(self) -> Optional[str]:
        """
        Gets a space-separated string of 1-based indices for frozen atoms from ASE constraints.

        Processes FixAtoms constraints from the atoms object and converts the zero-based indices
        to one-based indices suitable for LAMMPS input format.

        Returns
        -------
        Optional[str]
            A space-separated string of 1-based atom indices that are frozen (e.g. "1 5 12"),
            or None if no atoms are frozen.
        """
        frozen_zero_based = []
        for constr in self.atoms.constraints:
            if isinstance(constr, FixAtoms):
                frozen_zero_based.extend(constr.get_indices())

        if not frozen_zero_based:
            return None
        #sorting to 1-based list
        ids = sorted({idx + 1 for idx in frozen_zero_based})
        return " ".join(str(i) for i in ids)
        # forming string "1 5 12 etc"

    def _add_freeze_groups(self, block:str) -> str:
        """
        Modifies the LAMMPS input block to handle frozen atoms based on ASE constraints.

        Replaces static group declarations with dynamic ones based on the frozen atom indices.
        If no atoms are frozen, removes the frozen/mobile group declarations.

        Parameters
        ----------
        block : str
            LAMMPS input block containing group declarations for frozen/mobile atoms

        Returns
        -------
        str
            Modified LAMMPS input block with updated or removed group declarations
        """

        frozen_ids = self._get_frozen_ids()
        if frozen_ids is None:
            # 1) deleting old strings
            blk = block.replace("group           frozen id 1", "")
            blk = blk.replace("group           mobile id > 1", "")
            header1 = "group           mobile id > 1\n"
            header2 = "group           frozen id 1\n"
            return header1 +header2 + blk
        # changing the strings to dynamical ones

        new_block = block

        new_block = new_block.replace(
            "group           frozen id 1",
            f"group           frozen id {frozen_ids}"
        )
        # Instead of "id > 1" we'll make group mobile = all \ frozen
        new_block = new_block.replace(
            "group           mobile id > 1",
            "group           mobile subtract all frozen"
        )
        return new_block


    def opt_struc(self, type_opt = "minimize", steps = None, removal = None, max_num_rings=None, start_T=None, final_T=None, max_remove_over=None, FF: str = "BKS"):
        """
        Optimizes the structure of the current model based on the specified optimization type.
        The function supports minimizing the structure, performing annealing, annealing with
        minimization, and final structure optimizations with various conditions applied,
        including temperature, force-field, and number of steps for annealing. Intermediate
        data and results are saved during the optimization processes.

        :param type_opt: Type of optimization to perform. Available options are 'minimize',
                         'anneal', 'anneal_with_min', and 'final'.
        :param steps: Number of steps for the annealing process, required for 'anneal',
                      'anneal_with_min', and 'final' optimization types.
        :param removal: Optionally specify parameters for removal operations (if applicable).
        :param max_num_rings: Maximum number of rings allowed in the structure (if applicable).
        :param start_T: Starting temperature for the annealing process, required for 'anneal',
                        'anneal_with_min', and 'final' optimization types.
        :param final_T: Final temperature for the annealing process, required for 'anneal',
                        'anneal_with_min', and 'final' optimization types.
        :param max_remove_over: Maximum number of atoms to be removed over certain processes
                                (if applicable).
        :param FF: Force-field to be used for optimization (default is 'BKS').

        :return: Updated atomic structure after optimization.
        :rtype: Any
        """

        with cd(Path("LAMMPS")):
            self._write_data_file()

            match type_opt:
                case 'minimize':
                    self._write_in_file_minimize(FF=FF)
                    self._execute_and_save(label="opted_struc")
                    self.properties["pe"] = float(self.get_pot())
                case 'anneal':
                    assert steps is not None, "Need to specify number of steps"
                    self._write_in_file_anneal(steps, start_T, final_T, FF=FF)
                    self._execute_and_save(label="opted_struc")
                case 'anneal_with_min':
                    assert steps is not None, "Need to specify number of steps"
                    self._write_in_file_anneal_with_min(steps, start_T, final_T, FF=FF)
                    self._execute_and_save(label="opted_struc")
                case 'final':
                    self.num_final += 1
                    assert steps is not None, "Need to specify number of steps"
                    self._write_in_file_anneal(steps, start_T, final_T, FF=FF)
                    self._execute_and_save(f"final_struc_convert_{self.num_final}")
                case _:
                    raise ValueError

            return self.atoms

    def get_pot(self) -> float:
        """
        Read potential energy from 'final_pe.txt', return as float.
        """
        path = Path("final_pe.txt")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        line = path.open().readline().strip()
        if not line:
            raise ValueError(f"{path} is empty")
        try:
            return float(line.split()[0])
        except ValueError as e:
            raise ValueError(f"Cannot parse PE from '{line}'") from e

    def _write_data_file(self, filename: str = "structure.data"):
        """
        Writes data file 'filename'.
        Atomic types goes in increase of atomic numbers
        Using atom_style=charge, units=real.
        """
        # 1) Checking if every atom has charge
        charges = [self.atom_charges[atom.symbol] for atom in self.atoms]
        self.atoms.set_initial_charges(charges)

        # 2) Ordering the atoms
        symbols = self.atoms.get_chemical_symbols()
        unique_syms = sorted(set(symbols), key=lambda s: atomic_numbers[s])

        # 3) Write up
        write_lammps_data(
            filename,
            self.atoms,
            specorder=unique_syms,
            atom_style="charge",
            masses=True,  #  Masses
            velocities=False,  # Velocities
            units="real",  # like in the original input
            bonds=False  # Bonds is the social construct anyway
        )

        with open("structure.data", "a") as f:
            f.write("\nVelocities\n\n")
            for i in range(1, len(self.atoms) + 1):
                f.write(f"{i}  0.0 0.0 0.0\n")


    def _ff_block(self, FF: str) -> str:
        """
        Generates a formatted string representing force field parameters for a specified force
        field type. Currently method supports the "BKS" and "ReaxFF" force field types.

        :param FF: The force field type as a string ("BKS" or "ReaxFF").
        :type FF: str
        :raises ValueError: If the provided force field type is not recognized.
        :return: A formatted string containing force field parameters for the specified type.
        :rtype: str
        """

        if FF == "BKS":
            # 1) the same types as in data_file
            syms = sorted(set(self.atoms.get_chemical_symbols()),
                          key=lambda s: atomic_numbers[s])
            type_index = {s: i + 1 for i, s in enumerate(syms)}

            lines = [
                "pair_style      hybrid/overlay buck/coul/long 5.5 8.0 lj/cut 1.2",
                "kspace_style    ewald 1.0e-4",
                ""
            ]
            # 2) buck/coul/long — convert A and C из eV→kcal/mol and B to Angst
            for (a, b), (A_eV, rho_inv, C_eV) in self._bks_params.items():
                if a in type_index and b in type_index:
                    i, j = type_index[a], type_index[b]
                    A = A_eV * ev_to_kcal
                    rho = 1/rho_inv
                    C = C_eV * ev_to_kcal
                    lines.append(f"pair_coeff {i:>2d} {j:>2d} buck/coul/long {A:.6g} {rho:.6g} {C:.6g} # {a}-{b}")
            lines.append("")

            # 3) lj/cut — ε from eV→kcal/mol, σ to Å
            for (a, b), (eps_eV, sigma, cutoff) in self._bks_lj.items():
                if a in type_index and b in type_index:
                    i, j = type_index[a], type_index[b]
                    eps = eps_eV * ev_to_kcal
                    lines.append(f"pair_coeff {i:>2d} {j:>2d} lj/cut {eps:.6g} {sigma:.6g} {cutoff:.6g} # {a}-{b}")
            lines.append("")

            # 4) charges
            for a, q in self._bks_charges.items():
                if a in type_index:
                    t = type_index[a]
                    lines.append(f"set type {t} charge {q:.6g}   # {a}")
            lines.append("")

            return "\n".join(lines)
        elif FF == "ReaxFF":
            return textwrap.dedent("""\
                pair_style reaxff NULL safezone 3.0 mincap 150
                pair_coeff * * ../ffield_Yeon Si O
                pair_coeff * * ../ffield_Yeon O H
                pair_coeff * * ../ffield_Yeon Al O
                fix myqeq all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 4000

            """)
        else:
            raise ValueError(f"Unknown FF {FF!r}")

    def _minimize_block(self) -> str:
        """
        Constructs and minimizes a block of configuration instructions in a simulation script.

        :return: The assembled simulation script block with frozen groups added.
        :rtype: str
        """
        elems = self._element_list
        base = textwrap.dedent(f"""\
                    neighbor        2.0 bin
                    neigh_modify    every 2 delay 0 check yes
                    group           frozen id 1
                    group           mobile id > 1
                    timestep        0.5
                    run_style       verlet
    
                    thermo_modify   lost warn
                    thermo_style    custom step temp press time vol density etotal lx ly lz
                    thermo          10
    
                    variable pe equal pe
                    dump    xyz  all xyz 1 dump.xyz
                    dump_modify xyz element {elems}
    
                    minimize        0 5.0e-1 1000 1000000
    
                    compute myPE all pe
                    print   ${{pe}}"  file final_pe.txt
    
                    write_data final_struc.data
                """)
        return self._add_freeze_groups(base)

    def _anneal_block(self, steps: int, start_T: float, final_T: float) -> str:
        """
        Constructs a LAMMPS input block for annealing

        :param steps: Number of simulation steps for the annealing process.
        :type steps: int
        :param start_T: Initial temperature of the simulation in Kelvin.
        :type start_T: float
        :param final_T: Final target temperature of the simulation in Kelvin.
        :type final_T: float
        :return: A formatted string containing the LAMMPS input script for the annealing
            phase, including temperature control, output settings, and group definitions.
        :rtype: str
        """
        dump_freq = max(1, steps // 15)
        seed = self.rng.integers(1000000)
        elems = self._element_list
        base =  textwrap.dedent(f"""\
            neighbor        2.0 bin
            neigh_modify    every 2 delay 0 check yes
            group           frozen id 1
            group           mobile id > 1
            timestep        0.5
            run_style       verlet

            thermo_modify   lost warn
            thermo_style    custom step temp press time vol density etotal lx ly lz
            thermo          10

            dump    xyz  all xyz {dump_freq} dump.xyz
            dump_modify xyz element {elems}

            velocity        mobile create {start_T:.1f} {seed:d} dist gaussian
            fix 1 frozen move linear 0 0 0

            fix 2 mobile nve
            fix 4 mobile temp/berendsen {start_T:.1f} {start_T:.1f} 100
            run   {4*steps:d}

            fix 4 mobile temp/berendsen {start_T:.1f} {final_T:.1f} 50
            run   {steps:d}

            write_data final_struc.data
        """)
        return self._add_freeze_groups(base)

    def _anneal_with_min_block(self, steps: int, start_T: float, final_T: float) -> str:
        """
        Constructs a LAMMPS input block for annealing with minimization.

        :param steps: Number of steps for the annealing process.
        :type steps: int
        :param start_T: Starting temperature for the annealing process.
        :type start_T: float
        :param final_T: Final temperature for the annealing process.
        :type final_T: float
        :return: Modified annealing block with changes applied.
        :rtype: str
        """
        block = self._anneal_block(steps, start_T, final_T)

        return block.replace(
                        "velocity",
                        "minimize 0 5.0e-1 1000 1000000\n\nvelocity"
            )

    def _write_instruction(self,
                           type_opt: str,
                           FF: str,
                           steps: Optional[int]=None,
                           start_T: Optional[float]=None,
                           final_T: Optional[float]=None):
        """Master method to write out `instruction.in` based on type_opt."""
        header = textwrap.dedent("""\
            units      real
            atom_style charge
            boundary   p p p

            read_data  structure.data

        """)
        with open("instruction.in","w") as f:
            f.write(header)
            f.write(self._ff_block(FF))
            match type_opt:
                case "minimize":
                    f.write(self._minimize_block())
                case "anneal":
                    assert steps and start_T and final_T, "must supply steps, start_T, final_T"
                    f.write(self._anneal_block(steps, start_T, final_T))
                case "anneal_with_min":
                    assert steps and start_T and final_T
                    f.write(self._anneal_with_min_block(steps, start_T, final_T))
                case "final":
                    assert steps and start_T and final_T
                    f.write(self._anneal_block(steps, start_T, final_T))
                case _:
                    raise ValueError(f"Unknown opt type {type_opt!r}")

    def _write_in_file_minimize(self, FF):
        """Write out `instruction.in` for minimization."""
        self._write_instruction("minimize", FF)

    def _write_in_file_anneal(self, steps, st, fi, FF):
        """Write out `instruction.in` for annealing."""
        self._write_instruction("anneal", FF, steps, st, fi)

    def _write_in_file_anneal_with_min(self, steps, st, fi, FF):
        """Write out `instruction.in` for annealing with minimization."""
        self._write_instruction("anneal_with_min", FF, steps, st, fi)

    def _run(self):
        """
        Run LAMMPS using instruction.in, writing all output & errors to lmp.out.
        Raises CalledProcessError on non-zero exit.
        """
        cmd = [self.lammps_cmd, "-in", "instruction.in"]
        with open("lmp.out", "w") as outfile:
            subprocess.run(
                cmd,
                stdout=outfile,
                stderr=subprocess.STDOUT,
                check=True,
                text=True
            )


    def _read_final_struc_data(self, data_file: Optional[Path] = None) -> None:
        """
        Read the final LAMMPS data file into self.atoms.
        data_file: path to final_struc.data
        """
        if data_file is None:
            data_file = Path("final_struc.data")
        if not data_file.exists():
            raise FileNotFoundError(f"{data_file!r} not found")

        self.atoms = read(str(data_file), format="lammps-data")
