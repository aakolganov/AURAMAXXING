#!/bin/bash
#SBATCH --job-name=aura-s1
#SBATCH --array=0-3                 # N array tasks -> N shards (one shard per task)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # cores per node (= the in-node pool size below)
#SBATCH --time=08:00:00
#SBATCH --output=s1-%A_%a.out
#
# Scenario 1 -- CPU-only LAMMPS/BKS. Each array task is one node running its shard of the
# sweep across an in-node pool of single-threaded workers, so many slabs grow at once per
# node. Budget: workers * threads <= cpus-per-task.
set -euo pipefail
CONFIG=examples/config/scenario1_lammps_cpu.yaml
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -m runner "$CONFIG" \
    --num-shards "$NUM_SHARDS" --shard "$SHARD" \
    --workers "$SLURM_CPUS_PER_TASK" --threads 1

# After the whole array finishes, build the single pooled report + merge manifests once:
#   python -m runner "$CONFIG" --pool-only
