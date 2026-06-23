#!/bin/bash
#SBATCH --job-name=auramaxxing
#SBATCH --array=0-3              # N array tasks -> N shards (set to num_shards-1)
#SBATCH --nodes=1               # one node per array task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      # cores available to each task (= cores_per_node here)
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%A_%a.out
#
# Slab generation as a SLURM job array. Each array task runs one shard of the plan
# (a disjoint slice of the seed/roughness sweep), so the whole sweep is covered exactly
# once across the array. The single pooled report is built by a separate reduce step
# after the array finishes (see the dependent --pool-only job at the bottom).
#
# Budgeting rule: workers * threads <= cpus-per-task, or the workers oversubscribe cores.

set -euo pipefail
CONFIG=${1:?usage: sbatch generate_array.sbatch <config.yaml>}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

# --- choose ONE of the two scenarios -------------------------------------------------

# Scenario 1 -- CPU-only LAMMPS/BKS: many single-threaded slabs at once on the node.
WORKERS=$SLURM_CPUS_PER_TASK
THREADS=1

# Scenario 2 -- CPU-only MACE: one slab per node, MACE uses every core per evaluation.
# WORKERS=1
# THREADS=$SLURM_CPUS_PER_TASK

# Belt-and-suspenders for BLAS backends that read their thread count at interpreter
# start (the --threads flag also sets these, but exporting here covers eager init).
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS

python -m runner "$CONFIG" \
    --num-shards "$NUM_SHARDS" --shard "$SHARD" \
    --workers "$WORKERS" --threads "$THREADS"

# --- Scenario 3 -- MACE on GPU + CPU workers (replaces the call above) ---------------
# Add `#SBATCH --gres=gpu:1` at the top. One evaluator process owns the model on the GPU;
# --workers CPU processes run placement and proxy their MACE evals to it (--threads each).
#
#   python -m runner "$CONFIG" --remote --device cuda \
#       --num-shards "$NUM_SHARDS" --shard "$SHARD" \
#       --workers "$SLURM_CPUS_PER_TASK" --threads 1

# --- reduce step (run ONCE, after all array tasks finish) ----------------------------
# Gathers every per-structure metrics.json into the single pooled report and merges the
# per-shard manifests into manifest.json. Submit as a dependent job, e.g.:
#
#   ARRAY_ID=$(sbatch --parsable generate_array.sbatch config.yaml)
#   sbatch --dependency=afterok:$ARRAY_ID --wrap "python -m runner config.yaml --pool-only"
