"""Optional: a ready-to-edit SLURM array submission script for the generated DFT inputs.

One array job per package, one task per generated structure. The array size and the
per-task input directory are wired in automatically; the resource directives, module loads
and the exact binary are left as clearly-marked ``# EDIT`` placeholders for the target
cluster (matching the style of ``examples/slurm/``).
"""
from __future__ import annotations

from pathlib import Path

# Per-package launch line (the only package-specific bit); edited on the cluster.
_RUN_LINE = {
    "vasp": "srun vasp_std",
    "cp2k": "srun cp2k.psmp -i {project}.inp -o cp2k.out",
}
_EXTRA_NOTE = {
    "vasp": "# Assemble POTCAR in each input dir (see POTCAR.order) before submitting.\n",
    "cp2k": "",
}


def _render(package: str, n: int, joblist: Path, project: str) -> str:
    run_line = _RUN_LINE[package].format(project=project)
    return f"""#!/bin/bash
#SBATCH --job-name=dft-{package}
#SBATCH --array=0-{n - 1}                 # one task per generated structure
#SBATCH --nodes=1                        # EDIT for your job size
#SBATCH --ntasks-per-node=32             # EDIT: MPI ranks per node
#SBATCH --time=24:00:00                  # EDIT
##SBATCH --partition=PARTITION           # EDIT: uncomment + set
##SBATCH --account=ACCOUNT               # EDIT: uncomment + set
#SBATCH --output=dft-{package}-%A_%a.out
#
# Ready-to-edit {package.upper()} geometry-optimization array job, one task per structure.
# EDIT the #SBATCH directives above and the module/launch lines below for your cluster.
# Paths in the job list are absolute as generated; regenerate or edit if you relocate inputs.
{_EXTRA_NOTE[package]}set -euo pipefail

# module load {package}                    # EDIT: load your {package.upper()}

JOBLIST="{joblist}"
DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBLIST")
cd "$DIR"
{run_line}                               # EDIT: your binary / launcher
"""


def write_slurm_scripts(output_root, structure_dft_dirs, packages,
                        cp2k_project: str = "amorphous_oxide") -> list[Path]:
    """Write a ready-to-edit ``submit_dft_<pkg>.sbatch`` + ``dft_<pkg>_jobs.txt`` per package
    under ``output_root``.

    ``structure_dft_dirs`` are the per-structure ``dft`` directories; a package's input dir is
    ``<dft_dir>/<package>``. Packages with no generated inputs on disk are skipped (so the
    array size always matches what actually got written). Returns the scripts written."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for package in packages:
        job_dirs = [Path(d) / package for d in structure_dft_dirs
                    if (Path(d) / package).is_dir()]
        if not job_dirs:
            continue
        joblist = (output_root / f"dft_{package}_jobs.txt").resolve()
        joblist.write_text("\n".join(str(d.resolve()) for d in job_dirs) + "\n")
        script = output_root / f"submit_dft_{package}.sbatch"
        script.write_text(_render(package, len(job_dirs), joblist, cp2k_project))
        written.append(script)
    return written
