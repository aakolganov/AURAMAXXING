# Parallel run examples (SLURM)

Three ready-to-edit job-array templates, one per parallelization scenario. Each runs a
disjoint shard of the sweep per array task (`--num-shards`/`--shard`); after the whole array
finishes, build the single pooled report + merge the per-shard manifests with `--pool-only`
(commented at the bottom of each script). The rule everywhere is
`workers * threads <= cores_per_node`.

| Scenario | Script | Config | Per-node layout |
|----------|--------|--------|-----------------|
| 1. CPU LAMMPS/BKS | `scenario1_lammps_cpu.sbatch` | `config/scenario1_lammps_cpu.yaml` | pool of many 1-thread workers (`--workers cores --threads 1`) |
| 2. CPU MACE | `scenario2_mace_cpu.sbatch` | `config/siral70_mace_cpu.yaml` | one slab per node, MACE uses all cores (`--workers 1 --threads cores`) |
| 3. MACE on GPU + CPU | `scenario3_mace_gpu.sbatch` | `config/scenario3_mace_gpu.yaml` | one GPU evaluator + CPU workers proxy to it (`--remote --device cuda --workers cores --threads 1`) |

Submit, then chain the reduce step as a dependent job:

```bash
CONFIG=examples/config/scenario1_lammps_cpu.yaml
ARRAY=$(sbatch --parsable examples/slurm/scenario1_lammps_cpu.sbatch)
sbatch --dependency=afterok:$ARRAY --wrap "python -m runner $CONFIG --pool-only"
```

Scenario 3 was validated on an AWS g4dn.xlarge (Tesla T4): the remote evaluator matches a
single-process MACE run to machine precision, and the GPU runs at ~90% utilization while
several CPU workers feed it. It requires `e3nn<0.5` (see `requirements.txt`).
