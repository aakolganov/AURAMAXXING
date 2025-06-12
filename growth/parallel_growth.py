#!/usr/bin/env python3
"""
parallel_growth.py

Provides two main functions:
  - worker_generate(...): spawns a single‐alpha slab generation in its own working directory
  - main_generation_routine(...): orchestrates many parallel worker_generate tasks, each with its own alpha

Usage (from another script):
    from parallel_crystal import main_generation_routine

    base_output_dir = "runs"
    alphas = [0.01, 0.1, 1.0, 10.0]  # for example
    max_workers = 4
    timeout_seconds = 5 * 60

    main_generation_routine(
        base_output_dir=base_output_dir,
        alphas=alphas,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
        total_desired_atoms=501,
        n_m=4,
        target_ratio={"Si":7, "Al":3},
        traj_filename="growth_trajectory.xyz",
        final_struc_template="SiO2_Al2O3_alpha_{alpha}.cif"
    )
"""
import os
import time
import shutil
from multiprocessing import Process, Queue, Manager
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from ase import Atoms
from growth.Amorphous_Structure_Growth import generate_amorphous_structure
from tqdm import tqdm

COLOUR_PALETTE = ["red","green","yellow","blue","magenta","cyan","white"]
n_colours = len(COLOUR_PALETTE)

def worker_generate(
    total_desired_atoms: int,
    alpha: float,
    n_m: int,
    starting_struc: Optional[Atoms],
    target_ratio: Dict[str, int],
    traj_filename: str,
    final_struc_template: str,
    work_dir: Path,
    bar_color: str,
    bar_position: int,
    result_queue: Queue,
    calculator = "lammps"
) -> None:
    """
    Worker function to generate one slab with a given alpha.
    Runs in its own process.  Invokes ASE_crystal_gen.generate_crystal(...) inside work_dir.

    On success, renames "final_struc.cif" to use the alpha in its filename.
    Always removes the "LAMMPS" subfolder before exiting.

    Parameters
    ----------
    total_desired_atoms : int
        Total number of atoms in the final slab (Si+O+Al).
    alpha : float
        Surface roughness parameter (attenuation factor for the Fourier-based surface).
    n_m : int
        Maximum Fourier mode along each axis.
    starting_struc : Optional[Atoms]
        Starting ASE Atoms object (if None, starts from scratch).
    target_ratio : Dict[str,int]
        Desired cation ratio, e.g. {"Si":7, "Al":3}.
    traj_filename : str
        Name of the XYZ trajectory file to write out during growth.
    final_struc_template : str
        Python format string for renaming the final CIF, e.g. "SiO2_Al2O3_alpha_{alpha}.cif"
    work_dir : Path
        Directory in which this worker will run.
    result_queue : Queue
        Multiprocessing Queue for returning a result dict at the end.
    """
    # record start times
    start_wall = time.time()
    start_cpu = time.process_time()

    status = "success"
    error_msg = ""
    wall_time = 0.0
    cpu_time = 0.0

    cif_file = ""

    try:
        # 1) Create and switch into work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(work_dir)

        # 2) Clean any leftover data
        xyz_path = work_dir / traj_filename
        if xyz_path.exists():
            xyz_path.unlink()
        lammps_data = work_dir / "LAMMPS" / "final_struc.data"
        if lammps_data.exists():
            lammps_data.unlink()

        # 3) Tqdm progress bar
        pbar = tqdm(
            total=total_desired_atoms,
            desc=f"α={alpha:.3f}",
            position=bar_position,
            colour=bar_color,
            leave=False
        )

        def progress_callback(delta: int=1):
            """updates the progress bar"""
            pbar.update(delta)

        # 4) Call the main generation routine
        #    Pass starting_struc (could be None), the target_ratio, etc.
        crystal = generate_amorphous_structure(
            total_desired_atoms=total_desired_atoms,
            alpha=float(alpha),
            n_m=n_m,
            starting_struc=starting_struc,
            TARGET_RATIO=target_ratio,
            traj_file=traj_filename,
            progress_cb=progress_callback,
            demo_mode=False,
            calculator=calculator
        )


        # 5) Rename the final CIF to embed alpha in the filename
        old_cif = Path("final_struc.cif")

        if old_cif.exists():
            alpha_str = f"{alpha:.5f}".replace(".", "_")
            final_filename = final_struc_template.format(alpha=alpha_str)
            Path(final_filename).unlink(missing_ok=True)
            old_cif.rename(final_filename)
            cif_file = str(Path.cwd() / final_filename)
        else:
            status = "error"
            error_msg = "final_struc.cif not found after generation"
            cif_file = ""  #

    except Exception as e:
        status = "error"
        error_msg = str(e)

    finally:
        # compute elapsed times
        end_wall = time.time()
        end_cpu = time.process_time()
        wall_time = end_wall - start_wall
        cpu_time = end_cpu - start_cpu

        # remove the LAMMPS folder if it exists
        lammps_folder = work_dir / "LAMMPS"
        if lammps_folder.exists():
            shutil.rmtree(lammps_folder, ignore_errors=True)

        #close progress bar
        try:
            pbar.close()
        except:
            pass

        # push result into the queue
        result_queue.put({
            "alpha": alpha,
            "status": status,
            "error_msg": error_msg,
            "wall_time": wall_time,
            "cpu_time": cpu_time,
            "work_dir": str(work_dir),
            "cif_file":cif_file
        })

        # Cleanup
        #result_queue.close()  #closing the que
        #result_queue.join_thread()  # waiting for the


def main_generation_routine(
    base_output_dir: str,
    alphas: List[float],
    max_workers: int,
    timeout_seconds: int,
    total_desired_atoms: int,
    starting_struc: Optional[Atoms],
    n_m: int,
    target_ratio: Dict[str, int],
    traj_filename: str,
    final_struc_template: str,
    calculator = "lammps"
) -> None:
    """
    Orchestrates parallel generation of multiple slabs, each with a different alpha.
    Spawns up to `max_workers` concurrent processes of `worker_generate`. Each process has a
    per-task timeout of `timeout_seconds`. Summarizes successes/timeouts/errors at the end.

    Parameters
    ----------
    base_output_dir : str
        Directory under which all subdirectories (one per alpha) will be created.  If it already
        exists, it will be deleted and recreated from scratch.
    alphas : List[float]
        List of alpha values (surface roughness parameters).  One separate run for each.
    max_workers : int
        Maximum number of parallel worker processes.
    timeout_seconds : int
        If any worker exceeds this many seconds of wall‐time, it will be terminated and marked as "timeout".
    total_desired_atoms : int
        Number of atoms in each generated slab.
    n_m : int
        Maximum Fourier mode parameter (passed to generate_crystal).
    starting_struc: Optional[Atoms]
        Initial structure (passed to generate_crystal).
    target_ratio : Dict[str,int]
        Cation ratio (e.g. {"Si":7, "Al":3}).
    traj_filename : str
        Name of the trajectory‐XYZ file (passed to generate_crystal).
    final_struc_template : str
        Format string for naming the final CIF, e.g. "SiO2_Al2O3_alpha_{alpha}.cif"
    """

    # 0) inter-process Lock via Manager (inter‐process Lock)
    manager = Manager()
    lock = manager.Lock()
    tqdm.set_lock(lock)

    base_dir = Path(base_output_dir)

    # 1) Create (or recreate) base_output_dir
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 2) Build (alpha, work_dir) list
    tasks = []
    for alpha in alphas:
        alpha_str = f"{alpha:.3f}".replace(".", "_")  # e.g. 0.01000 -> "0_01000"
        work_dir = base_dir / f"alpha_{alpha_str}"
        tasks.append((alpha, work_dir))

    # 3) Keep track of running processes
    results: List[Dict[str, Any]] = []
    running_procs: List[Dict[str, Any]] = []

    idx = 0
    total_tasks = len(tasks)

    # 4) Loop until all tasks are scheduled and all running procs have finished
    while idx < total_tasks or running_procs:
        # 4a) Launch new processes if we have capacity
        while len(running_procs) < max_workers and idx < total_tasks:
            colour = COLOUR_PALETTE[idx % n_colours]
            position = idx % max_workers
            alpha, work_dir = tasks[idx]
            result_q = Queue()
            p = Process(
                target=worker_generate,
                args=(
                    total_desired_atoms,
                    alpha,
                    n_m,
                    starting_struc,
                    target_ratio,
                    traj_filename,
                    final_struc_template,
                    work_dir,
                    colour,
                    position,
                    result_q,
                    calculator
                ),
                daemon=False
            )
            p.start()
            running_procs.append({
                "process": p,
                "queue": result_q,
                "start_time": time.time(),
                "alpha": alpha,
                "work_dir": work_dir
            })
            idx += 1

        # 4b) Brief sleep to avoid tight‐loop
        time.sleep(1.0)

        # 4c) Check each running process for timeout or completion
        for entry in running_procs[:]:
            p: Process = entry["process"]
            elapsed = time.time() - entry["start_time"]

            # 4c.i) If still alive and exceeded timeout → terminate & mark “timeout”
            if p.is_alive() and elapsed > timeout_seconds:
                p.terminate()
                p.join()
                results.append({
                    "alpha": entry["alpha"],
                    "status": "timeout",
                    "error_msg": f"Exceeded {timeout_seconds} s",
                    "wall_time": timeout_seconds,
                    "cpu_time": None,
                    "work_dir": str(entry["work_dir"])
                })
                # Clean up LAMMPS folder if it exists
                lammpath = entry["work_dir"] / "LAMMPS"
                if lammpath.exists():
                    shutil.rmtree(lammpath, ignore_errors=True)

                running_procs.remove(entry)
                continue

            # 4c.ii) If process has finished normally → collect its result
            if not p.is_alive():
                p.join()
                try:
                    result = entry["queue"].get_nowait()
                except Exception:
                    # queue might be empty if something went wrong
                    result = {
                        "alpha": entry["alpha"],
                        "status": "error",
                        "error_msg": "No result returned (possibly crashed)",
                        "wall_time": elapsed,
                        "cpu_time": None,
                        "work_dir": str(entry["work_dir"]),
                        "cif_file": ""
                    }
                if result["status"] == "success" and result.get("cif_file", "") == "":
                    found = list(Path(result["work_dir"]).glob("*.cif"))
                    if len(found) >= 1:
                        result["cif_file"] = str(found[0])
                    else:
                        result["status"] = "error"
                        result["error_msg"] = "final_struc.cif not found in work_dir"

                results.append(result)
                running_procs.remove(entry)


    # 5) At this point, all tasks have completed (success, error, or timeout)
    successes = [r for r in results if r["status"] == "success"]
    timeouts = [r for r in results if r["status"] == "timeout"]
    errors   = [r for r in results if r["status"] == "error"]

    # 6) Compute average times for successful runs
    avg_wall = np.mean([r["wall_time"] for r in successes]) if successes else 0.0
    avg_cpu  = np.mean([r["cpu_time"] for r in successes if r["cpu_time"] is not None]) if successes else 0.0

    # 7) Write a summary file under base_output_dir
    summary_path = base_dir / "generation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Parallel Generation Summary\n")
        f.write("===========================\n\n")
        f.write(f"Total tasks scheduled: {total_tasks}\n")
        f.write(f"Successful:            {len(successes)}\n")
        f.write(f"Timeout failures:      {len(timeouts)}\n")
        f.write(f"Error failures:        {len(errors)}\n\n")
        f.write(f"Average wall time (s) [successes]: {avg_wall:.2f}\n")
        f.write(f"Average CPU  time (s) [successes]: {avg_cpu:.2f}\n\n")

        f.write("Successful runs and their output files:\n")
        for r in successes:
            alpha_str = f"{r['alpha']:.5f}"
            cif = r.get("cif_file", "")
            f.write(f"  alpha={alpha_str:<8}  {cif}\n")

        f.write("\nFailures due to timeout:\n")
        for r in timeouts:
            alpha_str = f"{r['alpha']:.5f}"
            f.write(f"  alpha={alpha_str}: {r['error_msg']}\n")

        f.write("\nFailures due to errors:\n")
        for r in errors:
            alpha_str = f"{r['alpha']:.5f}"
            f.write(f"  alpha={alpha_str}: {r['error_msg']}\n")
    print("All tasks completed.")
    print(f"Summary written to: {summary_path}")
    manager.shutdown()