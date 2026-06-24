import os
from abc import ABC
from pathlib import Path
from typing import Any

from ase import Atoms, units
from ase.io import write
from ase.calculators.calculator import Calculator
from ase.optimize import LBFGS  # Or LBFGS, FIRE, etc.
from ase.md.langevin import Langevin


class AnnealingLangevin(Langevin):
    """
    Custom ASE Molecular Dynamics class that linearly ramps down 
    the temperature from T_ini to T_fin over a given number of steps.
    """
    def __init__(self, atoms: Atoms, timestep: float, T_ini: float, T_fin: float, 
                 total_steps: int, friction: float, **kwargs):
        # Initialize with the starting temperature
        super().__init__(atoms, timestep, temperature_K=T_ini, friction=friction, **kwargs)
        self.T_ini = T_ini
        self.T_fin = T_fin
        self.total_steps = total_steps
        self.current_step = 0

    def step(self, forces=None):
        """Override the step method to update the temperature dynamically."""
        # Calculate the interpolation fraction (protect against division by zero)
        frac = self.current_step / max(1, self.total_steps - 1)
        
        # Calculate and set the new temperature
        current_T = self.T_ini + frac * (self.T_fin - self.T_ini)
        self.set_temperature(temperature_K=current_T)
        
        self.current_step += 1
        
        # Call the parent class step to actually move the atoms
        super().step(forces)


class CalculatorInterface(ABC):
    """Base class for calculator interfaces with standard MD/Opt methods."""
    dump_path: Path
    calc: Calculator

    def _attach_trajectory(self, run, atoms: Atoms,
                           filename: str, fmt: str = "xyz",
                           interval: int = 1, workdir: "Path | None" = None):
        """
        Attaches a modular trajectory writer to an optimizer or MD engine.
        Handles both native ASE .traj files and appended text formats (like .xyz).

        ``workdir`` overrides ``self.dump_path`` so a single slab writes its trajectory
        into its own directory; this keeps concurrent runs from sharing one file.
        """
        out_dir = Path(workdir) if workdir is not None else self.dump_path
        out_dir.mkdir(parents=True, exist_ok=True)

        full_filename = f"{filename}.{fmt}"
        filepath = out_dir / full_filename
        if Path(filepath).exists():
            os.remove(filepath)
        
        def write_frame():
            write(filepath, atoms, append=True, format=fmt)
        run.attach(write_frame, interval=interval)


    def optimize(self, atoms: Atoms, fmax: float = 2.0, max_steps: int = 50,
                 logfile: str = "log.log", traj_name: str | None = None, traj_fmt: str | None = None,
                 workdir: "Path | None" = None,
                 **kwargs: Any) -> Atoms:
        """
        Optimize the geometry of the structure using BFGS.

        ``workdir`` overrides ``self.dump_path`` for the log/trajectory of this call so
        concurrent slabs never write to the same files.
        """
        print("Starting Optimization")
        traj_interval = kwargs.pop('interval', 1)
        out_dir = Path(workdir) if workdir is not None else self.dump_path
        out_dir.mkdir(parents=True, exist_ok=True)

        atoms.calc = self.calc
        opt = LBFGS(atoms, logfile=out_dir/logfile, **kwargs)
        if traj_name:
            self._attach_trajectory(opt, atoms, traj_name, traj_fmt, interval=traj_interval, workdir=out_dir)

        opt.run(fmax=fmax, steps=max_steps)
        
        atoms.get_potential_energy()
        
        return atoms


    def anneal(self, atoms: Atoms, T_ini: float, T_fin: float,
               steps: int = 500, dt: float = 1.0 * units.fs, friction: float = 0.002,
               logfile: str = "log.log", traj_name: str | None = None, traj_fmt: str | None = None,
               workdir: "Path | None" = None,
               **kwargs: Any) -> Atoms:
        """
        Anneal the structure using a custom slowly decreasing Langevin thermostat.

        ``workdir`` overrides ``self.dump_path`` for the log/trajectory of this call so
        concurrent slabs never write to the same files.
        """
        print("Starting Anneal")
        traj_interval = kwargs.pop('interval', 1)
        out_dir = Path(workdir) if workdir is not None else self.dump_path
        out_dir.mkdir(parents=True, exist_ok=True)

        # Ensure the atoms object uses the interface's calculator
        atoms.calc = self.calc

        # Initialize our custom Annealing MD class
        dyn = AnnealingLangevin(
            atoms=atoms,
            timestep=dt,
            T_ini=T_ini,
            T_fin=T_fin,
            total_steps=steps,
            friction=friction,
            logfile=out_dir/logfile,
            **kwargs
        )
        if traj_name:
            self._attach_trajectory(dyn, atoms, traj_name, traj_fmt, interval=traj_interval, workdir=out_dir)

        # Run the annealing process
        dyn.run(steps)
        
        return atoms