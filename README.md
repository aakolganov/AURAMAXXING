# AURAMAXXING

**A**morphous s**U**rface **R**esearch **A**nd **M**odeling **A**nd o**X**ide e**X**ploration **I**ntegrated i**N** **G**eneration

Tools to efficiently generate amorphous oxide surfaces with ASE. Bare-surface growth *and* chemical saturation are composition-general (oxidation states, coordination numbers and bond-length distances are data-driven from a curated table + a covalent-radii model), relaxed with machine-learning potentials (MACE, UMA); a classical BKS/LAMMPS force field is kept as a lightweight generator for silica(-aluminas) only. Saturation caps are configurable 1-valent fragments (default –OH / H; or e.g. F, Na — see [Saturation](#saturation-capping-modes)).

Contributors:

- [Alexander Kolganov](https://github.com/aakolganov)
- [Mas Klein](https://github.com/MasKlein-1)

## What it does

A structure is built up atom-by-atom inside a configurable growth region, relaxed, and
then chemically terminated. The pipeline stages:

1. **Grow** — `growth.new_growth.grow_structure`: pick the next element from the target
   composition, choose an under-coordinated opposite-charge anchor, and place it on a sphere
   at a sampled bond length with KD-tree collision checking. On a steric jam the structure is
   melt-quenched (anneal) and re-sliced to the growth limits. Growth is composition-general
   (see below): oxidation states, coordination numbers and bond-length distances are
   data-driven, and the requested composition is made charge-neutral.
2. **Finalize** — `growth.new_growth.finalize_structure`: geometry optimization (LBFGS)
   with the chosen calculator.
3. **Saturate** — `saturation.new_sat.saturate_under_coordinated`: cap under-coordinated
   cations with –OH and anions with –H (data-driven from the per-element oxidation sign).
4. **Charge-correct** — `saturation.new_sat.correct_charge`: add H / OH until the net
   formal charge is zero (and `relabel_caps` swaps the caps for the run's fragments, e.g.
   F or Na — see [Saturation capping modes](#saturation-capping-modes)).
5. **Rules (optional)** — `rules.PeriodicStructureModifier` applies structural rules such
   as `AvoidMotifSwapRule` (e.g. avoid Al–O–Al) and `MinimumDistanceRule`.
6. **DFT inputs (optional)** — `dft.write_dft_inputs`: write ready-to-run VASP and/or CP2K
   geometry-optimization input sets (PBE-D3BJ, fixed cell, spin non-polarised) next to each
   structure for refinement. See [DFT-refinement inputs](#dft-refinement-inputs-step-3).

The growth region is defined by a flat bottom and a flat or Fourier-roughened top surface.

## Generation: composition & coordination

Bare-surface growth works for arbitrary oxides, not just silica/alumina. The composition is
given in any of these forms (see the configuration reference for the exact keys), and the cell
is made charge-neutral automatically:

- **brutto formula** — `formula: "SiO2"`
- **elemental ratio** — `ratio: {Si: 8, O: 2, Al: 1}`
- **structural mix** — `mix: {SiO2: 7, Al2O3: 1}` (also `"SiO2:Al2O3=7:1"` or `"70% SiO2 + 30% Al2O3"`)
- **explicit counts** — `counts: {Si: 160, O: 320}`

The first three take a `total_atoms`; the legacy `target_ratios` + `target_number_atoms` keys
still work. The resulting integer counts are adjusted to the nearest **charge-neutral**
composition (Σ count × oxidation = 0), keeping the requested ratios as close as possible, and
the adjustment is reported.

Per-element **oxidation states and coordination numbers** default from a curated table of
common oxide formers (`base/element_data.py`); per-pair **bond-length, collision and
bonding-cutoff distances** are derived from a covalent-radii model (ASE `covalent_radii`), so a
placed bond is always within its bonding cutoff. All of these are overridable in the
`coordination` block, and an element not in the curated table can be supplied there. A
`cn_distr` lets a fraction of an element take a different coordination number (and, optionally,
oxidation) — e.g. ~20% of Al grown at CN 6.

Relax arbitrary oxides with an **MLIP** (MACE/UMA). The classical **BKS/LAMMPS** backend is a
lightweight generator for **silica(-aluminas) only** (Si/Al/O) and raises a clear error for any
other element — and, because saturation adds H, it can never be the saturation calculator.
Chemical saturation/charge-correction is composition-general too (see
[Saturation capping modes](#saturation-capping-modes)).

## Installation

Python ≥ 3.10. Install the dependencies:

```bash
pip install -r requirements.txt
pip install -e .          # optional: also installs the `auramaxxing` command
```

This pulls NumPy, SciPy, NetworkX, ASE, matplotlib, PyYAML, and the relaxation backends
(`torch`, `mace-torch`, `lammps`). LAMMPS can alternatively be built from
[source](https://github.com/lammps/lammps); MACE models are distributed separately
([ACEsuit/mace](https://github.com/ACEsuit/mace)). The optional `UMAInterface`
additionally needs `fairchem-core`. The package is importable from the repo root
(top-level imports such as `from base import AmorphousStruc`).

## Quick start

The primary interface is a **YAML config file** run through the `runner`:

```bash
python -m runner examples/config/sio2.yaml            # generate
python -m runner --dry-run examples/config/sio2.yaml  # validate + print the plan only
python -m runner --threads 1 examples/config/sio2.yaml # cap compute threads per process
auramaxxing examples/config/sio2.yaml                 # same, if installed with `pip install -e .`
```

`--threads N` caps the compute threads this process uses (OMP/BLAS env + torch), to avoid
oversubscription when running many slabs at once. The rule is `workers * threads <=
cores_per_node`: use `--threads 1` for an in-node pool of many single-threaded workers, or
`--threads <cores>` to give one slab the whole node (e.g. one MACE job per node). For
backends whose threading initialises at interpreter start, also export `OMP_NUM_THREADS` in
your launch/sbatch script.

A minimal config — grow an amorphous SiO₂ slab with BKS/LAMMPS and write a POSCAR:

```yaml
# examples/config/sio2.yaml
structure:
  cell: [22.0, 22.0, 40.0]            # orthogonal box (Å); z holds slab + vacuum
composition:
  target_ratios: {Si: 1, O: 2}        # relative stoichiometric weights
  target_number_atoms: 432            # grow until the slab has this many atoms
limits:
  bottom: {type: flat,    z: 12.0}    # flat floor at z = 12
  top:    {type: fourier, z_av: 24.0, alpha: 0.3}   # rough ceiling (smaller alpha = rougher)
  fix: bottom
calculators:
  growth: {type: lammps, dump_path: dump_lmp}
run:
  seeds: [0]
  output_dir: output/sio2
```

## Configuration reference

Every parameter that drives a run lives in the config file. Only `structure`,
`composition`, `limits` and `calculators` are required; everything else has the defaults
shown below.

```yaml
structure:                  # start from a blank box OR a loaded file (exactly one)
  cell: [22.0, 22.0, 40.0]
  # pbc: [true, true, true]
  # from_file: POSCAR_bare_gAl_110   # grow on top of an existing surface
  # ase_read_kwargs: {}

composition:                # exactly one input form; all reduce to charge-neutral integer counts
  # (a) elemental ratio (relative weights) + total atom count:
  ratio: {Si: 4, Al: 2, O: 11}
  total_atoms: 440
  # (b) brutto formula + total:        formula: "SiO2"        / total_atoms: 432
  # (c) structural mix + total:        mix: {SiO2: 7, Al2O3: 1}    (or "SiO2:Al2O3=7:1",
  #                                         or "70% SiO2 + 30% Al2O3") / total_atoms: 440
  # (d) explicit per-element counts (no total needed):   counts: {Si: 160, O: 320}
  # Legacy keys 'target_ratios' + 'target_number_atoms' still work (alias of form (a)).
  # The counts are adjusted to the nearest charge-neutral integer composition (sum of
  # counts x oxidation states = 0), keeping the requested ratios as close as possible; the
  # adjustment is reported. target_number_atoms counts any pre-loaded substrate atoms.

limits:
  # each side is: flat(z) | fourier(z_av, alpha) | surface (top of a loaded slab).
  # 'offset' places z relative to the loaded surface instead of an absolute height.
  bottom: {type: flat,    z: 12.0}
  top:    {type: fourier, z_av: 24.0, alpha: 0.3, n_max: 6, m_max: 6}
  fix: bottom                              # bottom | top | null

coordination:               # optional; partial overrides merge onto the derived defaults
  # Defaults are derived for the run's element set: oxidation states + coordination numbers
  # from a curated table of common oxide formers (base/element_data.py), and per-pair
  # bond-length / collision / bonding-cutoff distances from a covalent-radii model (ASE
  # covalent_radii). Any element not in the curated table must be given oxidation + max_cn +
  # min_cn here (or added to the table).
  oxidation: {Ti: 4}                       # per-element signed oxidation state (sign = cation/anion)
  max_cn: {Al: 4}                          # per-element max coordination
  min_cn: {Al: 3}                          # per-element saturation floor
  distance: {bond_factor: 1.0, bond_dev: 0.15, cutoff_pad: 0.25, collision_factor: 0.75}
                                           # covalent-radii distance-model knobs (all optional)
  cut_offs: {Si-O: 1.95}                   # raw "El-El" cutoff override (must stay >= window upper)
  cn_distr:                                # expected coordination-number distribution per element
    Si: 4                                  # a single CN -> all Si are CN 4 (the only expected CN)
    Al: {4: 0.8, 6: 0.2}                   # 80% of Al at CN 4, 20% at CN 6
    Fe:                                    # list form carries a per-variant oxidation state
      - {cn: 6, fraction: 0.9}             # 90% Fe at CN 6 (default oxidation)
      - {cn: 4, fraction: 0.1, oxidation: 2}  # 10% as a 2+ / CN-4 site (same atoms get both)
    # A CN sets the atom's max coordination; the optional per-variant oxidation overrides the
    # element default for that fraction; any remainder (fractions sum < 1) falls back to the
    # curated default CN. NB: the charge-neutral target uses per-element oxidation, so
    # fractional-oxidation variants shift the realised charge slightly (settled by saturation).

calculators:                # type: lammps | mace | uma; extra keys are backend kwargs
  # NB: the lammps/BKS backend is a lightweight generator for silica(-aluminas) only -- it
  # supports Si/Al/O and raises on any other element. Use mace/uma for arbitrary oxides. BKS
  # also can't be the saturation calculator (saturation adds H, which BKS has no params for).
  growth:     {type: lammps, dump_path: dump_lmp}
  saturation: {type: mace, mace_model_path: model.model, device: cuda, dump_path: dump_mace}
              # optional; defaults to the growth calculator. mace needs mace_model_path;
              # uma needs uma_model_path (a pretrained name like "uma-s-1p2" that fairchem
              # downloads, or a local checkpoint path) and takes an optional task (default
              # omat; one of omol/omat/odac/oc20/oc25/omc).

growth:                     # all optional (defaults shown)
  max_placement_attempts: 1000
  per_anchor_attempts: 100
  num_samples: 100
  anneal: {T_ini: 2000, T_fin: 300, steps: 250, interval: 10}

finalize:                   # optional
  fmax: 0.1
  max_steps: 500

saturation:        {enabled: false, num_samples: 250}
# saturation caps default to –OH (cations) / H (anions); override with 1-valent fragments to
# get the other modes, e.g.  {enabled: true, negative_fragment: F}  (F caps cations).
# See "Saturation capping modes" below for the positive_fragment / negative_fragment / mode keys.
charge_correction: {enabled: false, max_iterations: 1000, num_samples: 250, move_alpha: 0.5}

statistics:        {enabled: true, per_structure: true, pooled: true}   # analysis plots + stats.json

debug:             {write_growth_dumps: false, write_trajectories: false} # optional; off by default
# write_growth_dumps: per-atom xyz snapshot after every placement (many files, heavy I/O)
# write_trajectories: anneal + finalize trajectory files. Both are debug-only; keep off
# for large/parallel ensembles. Trajectories/logs are written into each structure's own dir.

rules:                      # optional list, applied after saturation
  - {type: avoid_motif, edge_element: Al, center_element: O, swap_candidate: Si}
  - {type: min_distance, min_dist: 1.4}

run:
  seeds: [0, 1, 2]          # swept
  roughness: [0.01, 0.1, 1.0]   # optional; overrides limits.top.alpha (requires fourier top)
  output_dir: output
  output_format: vasp       # vasp | xyz

dft:                        # optional step 3: write VASP/CP2K geo-opt inputs per structure
  enabled: false
  packages: [vasp, cp2k]    # any subset
  slurm: false              # also drop a ready-to-edit SLURM array script per package
  vasp:
    incar:   {ENCUT: 550, MY_TAG: 1}    # override/add INCAR tags (null drops one)
    kpoints: {mode: gamma, grid: [1, 1, 1], shift: [0, 0, 0]}
  cp2k:
    project: amorphous_oxide
    basis_set_file: BASIS_MOLOPT
    potential_file: GTH_POTENTIALS
    input:   {FORCE_EVAL: {DFT: {MGRID: {CUTOFF: 600}}}}   # deep-merged onto the defaults
```

`run.seeds` × `run.roughness` is swept, producing one structure per combination written
to a per-run subdirectory (`output_dir/seed{seed}_alpha{alpha}/structure.<ext>`); a single
combination is written straight to `output_dir/structure.<ext>`. Unknown or missing keys
raise a clear error naming the offending path — use `--dry-run` to validate quickly without
LAMMPS/MACE. See `examples/config/sio2.yaml` and `examples/config/siral70.yaml`.

## Saturation capping modes

Saturation always runs the validated, charge-neutralising **–OH / H** engine first, then
swaps the cap identities for the run's **fragments**. Every fragment is **1-valent** (the
engine swaps whole caps, so a +1/–1 fragment keeps the slab neutral), written as an ordered
element list (the first atom bonds to the surface). Defaults: `positive_fragment: H` (caps
anions, net +1) and `negative_fragment: [O, H]` (caps cations, net –1). Override one or both:

| Mode | Config | Result |
| --- | --- | --- |
| **(a) neutral** | `negative_fragment: F` + `positive_fragment: Na` | cations → F, anions → Na (a net-neutral NaF pair) |
| **(b) anion_cap** | `negative_fragment: F` (positive stays H) | electronegative cap on cations; anions keep H |
| **(c) cation_cap** | `positive_fragment: Na` (negative stays –OH) | electropositive cap on anions; cations keep –OH |

```yaml
saturation:
  enabled: true
  mode: anion_cap            # optional; asserts the fragments match the named mode
  negative_fragment: F       # e.g. cap surface cations with F instead of –OH
charge_correction: {enabled: true}
```

A non-network cap element (F, Na, …) is auto-registered in the run's element tables with
its oxidation and a terminal coordination of 1; F and Cl are curated, others need an
`oxidation`/`min_cn`/`max_cn` entry in the `coordination` block. Because saturation adds
H/F/Na, the cap relax needs an **MLIP** saturation calculator (BKS is rejected). See
`examples/config/sio2_fluorine_cap.yaml`, `sio2_sodium_cap.yaml`, and `sio2_naf_cap.yaml`.

## DFT-refinement inputs (step 3)

The generated structures are heuristic, so the natural next step is a DFT geometry
optimization. With `dft.enabled` the runner writes a ready-to-run input set next to each
structure — `output_dir/.../dft/vasp/` and/or `.../dft/cp2k/` — built from the project's
standard method: **PBE-D3BJ**, **fixed-cell** ionic relaxation, **spin non-polarised**, no
wavefunction/charge files kept.

- **VASP** (`INCAR`, `KPOINTS`, `POSCAR`): `ENCUT 500`, `ISMEAR 0 / SIGMA 0.05`, `IVDW 12`
  (D3-BJ), `ISPIN 1`, `ISIF 2`, `LCHARG/LWAVE = .FALSE.`, Γ-only k-points. The `POSCAR`
  orders species **lightest-to-heaviest** (H, O, Al, Si …) and a `POTCAR.order` note records
  that order — assemble your `POTCAR` (PBE PAW) to match. `FixAtoms` is written as Selective
  Dynamics. py4vasp is *not* used here (it only reads VASP output); use it later for analysis.
- **CP2K** (`<project>.inp`, `coord.xyz`): QUICKSTEP/GPW, GTH-PBE + `DZVP-MOLOPT-SR-GTH`,
  `CUTOFF 500 / REL_CUTOFF 50` Ry, `EPS_SCF 1.0E-6` Ha, OT/DIIS, D3(BJ); `RUN_TYPE GEO_OPT`
  with any frozen substrate emitted as `&FIXED_ATOMS`.

Every tag is overridable from YAML: `vasp.incar` / `vasp.kpoints` are merged onto our INCAR/
KPOINTS defaults, and `cp2k.input` is **deep-merged** onto the CP2K section tree — a `null`
value drops a tag, a new key adds one, so you can both adjust our settings and add your own.

With `dft.slurm: true` the run also drops a ready-to-edit **SLURM array** script per package
at the output root (`submit_dft_vasp.sbatch` / `submit_dft_cp2k.sbatch`, with a matching
`dft_<pkg>_jobs.txt`). The array size and per-task input directory are wired in automatically;
the `#SBATCH` resources, module loads and the exact binary are left as `# EDIT` placeholders
(default launch is `srun vasp_std` / `srun cp2k.psmp -i <project>.inp`). Submit with
`sbatch submit_dft_vasp.sbatch` once you've assembled the POTCARs. (For a sharded sweep the
script is written by the `--pool-only` reduce step, like the pooled statistics.)

The same generator runs standalone on already-saved structures (no regeneration):

```bash
auramaxxing-dft output/siral70                      # every structure.vasp under the dir
auramaxxing-dft structure.vasp --package cp2k       # one file, CP2K only
auramaxxing-dft output/siral70 --slurm              # also write the submit_dft_*.sbatch array script
auramaxxing-dft output/siral70 --config examples/config/siral70.yaml   # apply the dft: overrides
```

(or `from dft import write_dft_inputs` in Python).

## Running in parallel (HPC)

Each slab in the sweep is independent, so a large ensemble parallelises two ways that
compose:

- **Across nodes** — `--num-shards N --shard I` runs a disjoint slice of the plan
  (`plan[I::N]`), one SLURM array task per shard. The union of shards is the whole sweep
  with no overlap.
- **Within a node** — `--workers W` runs W slabs at once in a process pool; each worker
  builds its own calculator.

```bash
# one node, 8 slabs at a time, single-threaded each (CPU LAMMPS)
python -m runner config.yaml --workers 8 --threads 1
# this node's share of a 4-way sharded sweep
python -m runner config.yaml --num-shards 4 --shard 0 --workers 8 --threads 1
```

Keep `workers * threads <= cores_per_node`. For CPU MACE, prefer one slab per node
(`--workers 1 --threads <cores>`) so each evaluation uses the whole node and the model is
loaded once per node; parallelism then comes from the array.

After a sharded run, build the single pooled report (and merge the per-shard manifests)
once with `python -m runner config.yaml --pool-only`.

**MACE on GPU + CPU workers (`--remote`).** With a GPU, `--remote` evaluates the MACE
calculator in one process that owns the model on the device, while `--workers` CPU processes
run the placement/optimization control flow and proxy their force/energy calls to it:

```bash
python -m runner config.yaml --remote --device cuda --workers 8 --threads 1
```

The GPU is fed by all workers (so it stays busy), the workers never touch CUDA, and any
non-MACE stage (e.g. LAMMPS/BKS growth) still runs locally on the worker's CPU. The evaluator
coalesces up to `--batch-size` concurrently-queued requests (default = `--workers`) into one
batched forward pass; `--batch-size 1` disables it. Sharding, manifests and `--pool-only` work
the same way, and `--resume` skips already-written slabs. (`--remote` needs `e3nn<0.5` for MACE
checkpoints to load — see `requirements.txt`.)

Ready-to-edit job-array templates for all three scenarios are in `examples/slurm/` (see its
README); the configs they use are `examples/config/scenario1_lammps_cpu.yaml`,
`examples/config/siral70_mace_cpu.yaml`, and `examples/config/scenario3_mace_gpu.yaml`.

## Statistics & plots

With `statistics.enabled` (on by default) the runner writes a set of diagnostic plots +
a `stats.json` next to each structure, and a pooled set across the sweep in `output_dir`:

- **coordination.png** — coordination-number distribution per element (fixed substrate
  atoms excluded).
- **homo_element_distance.png** — nearest same-element distance per element, with the
  bonding cutoff marked and any homo-element "bonds" below it flagged (O–O, Si–Si … defects).
- **element_O_distance.png** — nearest element–O distance per element.
- **tau4.png** — τ₄ and τ₄′ geometry indices for 4-coordinate centres (1 = tetrahedral,
  0 = square planar).
- **cap_distance.png**, **caps_per_cation.png** — saturation runs only: cap bond-length
  distributions per cap pair (O–H, Si–F, O–Na, …) and the number of capping groups per cation.

For large ensembles, set `statistics.per_structure: false` to skip the per-structure plots
(which multiply with every structure) while still getting the single pooled set; or
`statistics.enabled: false` to turn analysis off entirely.

The pooled report is built by gathering the per-structure `metrics.json` files off disk, so
it is correct no matter how the structures were produced. After a parallel/sharded run (where
each task generates a subset of the slabs into a shared `output_dir`), build the single pooled
report once with:

```bash
python -m runner config.yaml --pool-only   # gather metrics.json under output_dir -> pooled report
```

The same analysis runs standalone on any saved structure:

```bash
python -m stats output/**/structure.vasp --out stats_dir   # add --saturation for the cap plots
```
(or `from stats import write_report` / `analyze_structure` in Python).

## Python API

The same config can be driven from Python, or the pipeline functions can be called
directly for full programmatic control:

```python
from runner import run_from_config
run_from_config("examples/config/sio2.yaml")        # -> list of written paths
```

```python
# Calling the pipeline directly
from base.initialize import initialize_structure_blank
from base.config import CoordinationConfig
from base.limits import make_limit_flat, make_limits_fourier, fix_limits
from growth.new_growth import grow_structure, finalize_structure
from interfaces.LAMMPS_Interface import LMPInterface
from default_constants import SIRAL_CN_DISTR        # {"Al": {6: 0.2}}  (~20% of Al at CN 6)

# Coordination limits + the coordination-number distribution (omit `config` for the defaults).
cfg = CoordinationConfig(cn_distr=SIRAL_CN_DISTR)
struct = initialize_structure_blank(cell=[22.0, 22.0, 40.0], config=cfg)
struct.set_seed(42)                                  # reproducible: same seed -> same structure

make_limit_flat(struct, z_val=12.0, is_for="bottom")
make_limits_fourier(struct, z_av=24.0, alpha=0.3, is_for="top")
fix_limits(struct.limits, hard_limit="bottom")

calc = LMPInterface(dump_path="dump_lmp")
grow_structure(struct, target_number_atoms=432, target_ratios={"Si": 1, "O": 2}, calculator=calc)
finalize_structure(struct, calculator=calc)
struct.atoms.write("SiO2_slab.vasp", format="vasp")
```

To grow on a loaded substrate, use `initialize_structure_file("POSCAR", ase_read_kwargs={})`
and place the limits above it (config equivalent: `structure.from_file` + a `surface`/`offset`
bottom limit).

## Examples

- `examples/config/sio2.yaml` — amorphous SiO₂ (BKS), single structure.
- `examples/config/siral70.yaml` — Si/Al with a MACE saturation backend and a seed × roughness sweep.
- `examples/config/sio2_fluorine_cap.yaml`, `sio2_sodium_cap.yaml`, `sio2_naf_cap.yaml` —
  the three [saturation capping modes](#saturation-capping-modes) (F on cations, Na on anions,
  a net-neutral NaF pair).
- `examples/Generation/`, `examples/Saturation/` — equivalent scripted (Python) runs,
  including growth on a bare γ-Al₂O₃(110) substrate whose POSCAR `Selective Dynamics`
  (F F F) flags ASE reads into a `FixAtoms` constraint that keeps the substrate fixed.