# AURAMAXXING

**A**morphous s**U**rface **R**esearch **A**nd **M**odeling **A**nd o**X**ide e**X**ploration **I**ntegrated i**N** **G**eneration

Tools to efficiently generate mixed Si/Al (for now) amorphous oxide surfaces with ASE, using a classical BKS/LAMMPS force field and/or machine-learning potentials (MACE, UMA) for relaxation.

Contributors:

- [Alexander Kolganov](https://github.com/aakolganov)
- [Mas Klein](https://github.com/MasKlein-1)

## What it does

A structure is built up atom-by-atom inside a configurable growth region, relaxed, and
then chemically terminated. The pipeline stages:

1. **Grow** — `growth.new_growth.grow_structure`: pick the next atom type from the
   target stoichiometry, choose an under-coordinated anchor, and place it on a sphere
   at a sampled bond length with KD-tree collision checking. On a steric jam the
   structure is melt-quenched (anneal) and re-sliced to the growth limits.
2. **Finalize** — `growth.new_growth.finalize_structure`: geometry optimization (LBFGS)
   with the chosen calculator.
3. **Saturate** — `saturation.new_sat.saturate_under_coordinated`: cap under-coordinated
   cations with –OH and anions with –H.
4. **Charge-correct** — `saturation.new_sat.correct_charge`: add H / OH until the net
   formal charge is zero.
5. **Rules (optional)** — `rules.PeriodicStructureModifier` applies structural rules such
   as `AvoidMotifSwapRule` (e.g. avoid Al–O–Al) and `MinimumDistanceRule`.

The growth region is defined by a flat bottom and a flat or Fourier-roughened top surface.

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
auramaxxing examples/config/sio2.yaml                 # same, if installed with `pip install -e .`
```

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

composition:
  target_ratios: {Si: 4, Al: 2, O: 11}   # relative weights (here a charge-neutral Si/Al oxide)
  target_number_atoms: 440                # counts any pre-loaded substrate atoms

limits:
  # each side is: flat(z) | fourier(z_av, alpha) | surface (top of a loaded slab).
  # 'offset' places z relative to the loaded surface instead of an absolute height.
  bottom: {type: flat,    z: 12.0}
  top:    {type: fourier, z_av: 24.0, alpha: 0.3, n_max: 6, m_max: 6}
  fix: bottom                              # bottom | top | null

coordination:               # optional; partial overrides merge onto the defaults
  max_cn: {Al: 4}                          # per-element max coordination
  min_cn: {Al: 3}                          # per-element saturation floor
  cut_offs: {Si-O: 1.95}                   # "El-El" bonding cutoffs (Å)
  overcoord_policy:                        # let a random fraction grow over-coordinated
    Al: {max_cn: 6, fraction: 0.2}         # ~20% of Al may reach CN 6, rest cap at 4

calculators:                # type: lammps | mace | uma; extra keys are backend kwargs
  growth:     {type: lammps, dump_path: dump_lmp}
  saturation: {type: mace, mace_model_path: model.model, device: cuda, dump_path: dump_mace}
              # optional; defaults to the growth calculator. mace needs mace_model_path;
              # uma needs uma_model_path.

growth:                     # all optional (defaults shown)
  max_placement_attempts: 1000
  per_anchor_attempts: 100
  num_samples: 100
  anneal: {T_ini: 2000, T_fin: 300, steps: 250, interval: 10}

finalize:                   # optional
  fmax: 0.1
  max_steps: 500

saturation:        {enabled: false, num_samples: 250}
charge_correction: {enabled: false, max_iterations: 1000, num_samples: 250, move_alpha: 0.5}

rules:                      # optional list, applied after saturation
  - {type: avoid_motif, edge_element: Al, center_element: O, swap_candidate: Si}
  - {type: min_distance, min_dist: 1.4}

run:
  seeds: [0, 1, 2]          # swept
  roughness: [0.01, 0.1, 1.0]   # optional; overrides limits.top.alpha (requires fourier top)
  output_dir: output
  output_format: vasp       # vasp | xyz
```

`run.seeds` × `run.roughness` is swept, producing one structure per combination written
to a per-run subdirectory (`output_dir/seed{seed}_alpha{alpha}/structure.<ext>`); a single
combination is written straight to `output_dir/structure.<ext>`. Unknown or missing keys
raise a clear error naming the offending path — use `--dry-run` to validate quickly without
LAMMPS/MACE. See `examples/config/sio2.yaml` and `examples/config/siral70.yaml`.

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
from default_constants import SIRAL_OVERCOORD       # {"Al": {"max_cn": 6, "fraction": 0.2}}

# Coordination limits + overcoordination policy (omit `config` for the defaults).
cfg = CoordinationConfig(overcoord_policy=SIRAL_OVERCOORD)
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
- `examples/Generation/`, `examples/Saturation/` — equivalent scripted (Python) runs,
  including growth on a bare γ-Al₂O₃(110) substrate whose POSCAR `Selective Dynamics`
  (F F F) flags ASE reads into a `FixAtoms` constraint that keeps the substrate fixed.