#!/bin/bash
#SBATCH --job-name=aura-s2
#SBATCH --array=0-7                 # one shard per task; more tasks = more nodes in flight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # whole node feeds one MACE evaluation at a time
#SBATCH --time=24:00:00
#SBATCH --output=s2-%A_%a.out
#
# Scenario 2 -- CPU-only MACE, one model per node. Each task loads the model once and runs
# its shard ONE slab at a time (workers=1), letting MACE use every core per evaluation.
# Parallelism comes from the array, not an in-node pool.
set -euo pipefail
CONFIG=examples/config/siral70_mace_cpu.yaml
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
THREADS=$SLURM_CPUS_PER_TASK

export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS

python -m runner "$CONFIG" \
    --num-shards "$NUM_SHARDS" --shard "$SHARD" \
    --workers 1 --threads "$THREADS"

# After the whole array finishes, build the single pooled report + merge manifests once:
#   python -m runner "$CONFIG" --pool-only
