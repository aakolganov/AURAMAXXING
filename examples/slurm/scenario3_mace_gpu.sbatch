#!/bin/bash
#SBATCH --job-name=aura-s3
#SBATCH --array=0-3                 # one shard per task (one GPU node each)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           # CPU workers that feed the GPU
#SBATCH --gres=gpu:1               # one GPU per task for the evaluator
#SBATCH --time=12:00:00
#SBATCH --output=s3-%A_%a.out
#
# Scenario 3 -- MACE on GPU + CPU workers. One evaluator process owns the model on the GPU;
# --workers CPU processes run placement (and the local LAMMPS/BKS growth) and proxy their
# MACE force/energy calls to it. The GPU stays busy while the CPUs do the control flow.
# Needs e3nn<0.5 for MACE checkpoints to load (see requirements.txt).
set -euo pipefail
CONFIG=examples/config/scenario3_mace_gpu.yaml
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -m runner "$CONFIG" --remote --device cuda \
    --num-shards "$NUM_SHARDS" --shard "$SHARD" \
    --workers "$SLURM_CPUS_PER_TASK" --threads 1

# After the whole array finishes, build the single pooled report + merge manifests once:
#   python -m runner "$CONFIG" --pool-only
