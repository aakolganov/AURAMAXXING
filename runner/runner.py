"""Execute a surface-generation run described by a `RunConfig`.

`run_from_config` expands the seed/roughness sweep, builds the calculators once, and
drives the existing pipeline (`initialize_structure_* -> make_limits -> grow_structure
-> finalize_structure -> [saturate -> correct_charge] -> [rules] -> write`) for each
combination. `resolve_plan` produces the per-combination plan without building
calculators or running anything (used by `--dry-run`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from base.initialize import initialize_structure_blank, initialize_structure_file
from base.limits import make_limit_flat, make_limits_fourier, fix_limits
from growth.new_growth import grow_structure, finalize_structure
from saturation.new_sat import saturate_under_coordinated, correct_charge
from rules import PeriodicStructureModifier, AvoidMotifSwapRule, MinimumDistanceRule

from .config import RunConfig, RunSpec, CalculatorSpec, LimitSideSpec, load_config

_EXT = {"vasp": "vasp", "xyz": "xyz"}


# --- calculator factory -------------------------------------------------------------

def build_calculator(spec: CalculatorSpec):
    """Instantiate a calculator backend from its spec. Backends are imported lazily so
    that loading/validating a config never requires LAMMPS, torch or MACE."""
    opts = dict(spec.options)
    if spec.type == "lammps":
        from interfaces.LAMMPS_Interface import LMPInterface
        return LMPInterface(**opts)
    if spec.type == "mace":
        if "mace_model_path" not in opts:
            raise ValueError("calculators: a 'mace' calculator needs 'mace_model_path'")
        from interfaces.MACE_interface import MACEInterface
        return MACEInterface(**opts)
    if spec.type == "uma":
        if "uma_model_path" not in opts:
            raise ValueError("calculators: a 'uma' calculator needs 'uma_model_path'")
        from interfaces.UMA_interface import UMAInterface
        return UMAInterface(**opts)
    raise ValueError(f"calculators: unknown calculator type {spec.type!r}")


# --- sweep / plan -------------------------------------------------------------------

def _alpha_label(alpha: Optional[float]) -> str:
    return f"{alpha:.3f}" if alpha is not None else "flat"


def _top_alpha_for(cfg: RunConfig, roughness: Optional[float]) -> Optional[float]:
    """The fourier-top roughness actually used for a combo (swept value overrides the
    config's own alpha); None when the top surface is flat."""
    if cfg.limits.top.type != "fourier":
        return None
    return roughness if roughness is not None else cfg.limits.top.alpha


def resolve_plan(cfg: RunConfig) -> list[dict]:
    """Expand seeds x roughness into a list of {seed, alpha, output_path} entries.

    Does not touch any calculator or structure. Raises on inconsistent sweep settings.
    """
    run: RunSpec = cfg.run
    roughs = run.roughness if run.roughness is not None else [None]
    if run.roughness is not None and cfg.limits.top.type != "fourier":
        raise ValueError("run.roughness sweeps the top fourier surface, but limits.top.type "
                         "is not 'fourier'")
    combos = [(seed, rough) for seed in run.seeds for rough in roughs]

    ext = _EXT[run.output_format]
    out_root = Path(run.output_dir)
    single = len(combos) == 1
    plan = []
    for seed, rough in combos:
        alpha = _top_alpha_for(cfg, rough)
        if single:
            out_path = out_root / f"structure.{ext}"
        else:
            out_path = out_root / f"seed{seed}_alpha{_alpha_label(alpha)}" / f"structure.{ext}"
        plan.append({"seed": seed, "alpha": alpha, "output_path": out_path})
    return plan


# --- limits -------------------------------------------------------------------------

def _surface_top(struct) -> float:
    if len(struct.atoms) == 0:
        raise ValueError("limits use 'surface'/'offset' but the structure is empty "
                         "(use absolute 'z'/'z_av', or start from a loaded file)")
    return float(struct.atoms.get_positions()[:, 2].max())


def _apply_limit_side(struct, side: LimitSideSpec, is_for: str,
                      alpha_override: Optional[float], surf_top_cache: dict) -> None:
    def surf():
        if "v" not in surf_top_cache:
            surf_top_cache["v"] = _surface_top(struct)
        return surf_top_cache["v"]

    if side.type == "surface":
        z = surf() + (side.offset or 0.0)
        make_limit_flat(struct, z_val=z, is_for=is_for)
    elif side.type == "flat":
        z = side.z if side.z is not None else surf() + side.offset
        make_limit_flat(struct, z_val=z, is_for=is_for)
    else:  # fourier
        z_av = side.z_av if side.z_av is not None else surf() + side.offset
        alpha = alpha_override if (is_for == "top" and alpha_override is not None) else side.alpha
        make_limits_fourier(struct, z_av=z_av, alpha=alpha, is_for=is_for,
                            n_max=side.n_max, m_max=side.m_max)


def _build_limits(struct, cfg: RunConfig, roughness: Optional[float]) -> None:
    surf_top_cache: dict = {}
    _apply_limit_side(struct, cfg.limits.bottom, "bottom", None, surf_top_cache)
    _apply_limit_side(struct, cfg.limits.top, "top", roughness, surf_top_cache)
    fix_limits(struct.limits, hard_limit=cfg.limits.fix)


# --- rules --------------------------------------------------------------------------

def _build_rule(spec):
    if spec.type == "avoid_motif":
        return AvoidMotifSwapRule(**spec.options)
    if spec.type == "min_distance":
        return MinimumDistanceRule(**spec.options)
    raise ValueError(f"rules: unknown rule type {spec.type!r}")


# --- run ----------------------------------------------------------------------------

def _init_structure(cfg: RunConfig):
    s = cfg.structure
    if s.from_file is not None:
        return initialize_structure_file(s.from_file, ase_read_kwargs=s.ase_read_kwargs,
                                         config=cfg.coordination)
    return initialize_structure_blank(cell=s.cell, pbc=s.pbc, config=cfg.coordination)


def _generate_one(cfg: RunConfig, seed: int, roughness: Optional[float],
                  out_path: Path, growth_calc, sat_calc):
    struct = _init_structure(cfg)
    struct.set_seed(seed)
    _build_limits(struct, cfg, roughness)

    g, f = cfg.growth, cfg.finalize
    grow_structure(
        amorphous_struct=struct,
        target_number_atoms=cfg.composition.target_number_atoms,
        target_ratios=cfg.composition.target_ratios,
        calculator=growth_calc,
        max_placement_attempts=g.max_placement_attempts,
        per_anchor_attempts=g.per_anchor_attempts,
        num_samples=g.num_samples,
        anneal_params=g.anneal,
        output_dir=out_path.parent / "growth",
    )
    finalize_structure(struct, calculator=growth_calc, fmax=f.fmax,
                       max_steps=f.max_steps, traj_interval=f.traj_interval)

    if cfg.saturation.enabled:
        saturate_under_coordinated(struct, num_samples=cfg.saturation.num_samples)
        finalize_structure(struct, calculator=sat_calc, fmax=f.fmax,
                           max_steps=f.max_steps, traj_interval=f.traj_interval)
        if cfg.charge_correction.enabled:
            cc = cfg.charge_correction
            correct_charge(struct, max_iterations=cc.max_iterations,
                           num_samples=cc.num_samples, move_alpha=cc.move_alpha)
            finalize_structure(struct, calculator=sat_calc, fmax=f.fmax,
                               max_steps=f.max_steps, traj_interval=f.traj_interval)

    if cfg.rules:
        modifier = PeriodicStructureModifier(struct)
        for rule_spec in cfg.rules:
            modifier.add_rule(_build_rule(rule_spec))
        modifier.optimize()
        finalize_structure(struct, calculator=sat_calc, fmax=f.fmax,
                           max_steps=f.max_steps, traj_interval=f.traj_interval)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    struct.atoms.write(str(out_path), format=cfg.run.output_format)
    return struct


def run_from_config(source: Union[str, Path, dict, RunConfig]) -> list[Path]:
    """Run the full pipeline for every seed/roughness combination in the config.

    Returns the list of written output paths. A combination that fails is logged and
    skipped rather than aborting the whole sweep.
    """
    cfg = source if isinstance(source, RunConfig) else load_config(source)
    plan = resolve_plan(cfg)

    growth_calc = build_calculator(cfg.calculators.growth)
    sat_calc = build_calculator(cfg.calculators.saturation) if cfg.calculators.saturation else growth_calc

    written: list[Path] = []
    for entry in plan:
        out_path = entry["output_path"]
        try:
            _generate_one(cfg, entry["seed"], entry["alpha"], out_path, growth_calc, sat_calc)
            written.append(out_path)
            print(f"[runner] wrote {out_path}")
        except Exception as exc:   # keep the sweep going; report what was skipped
            print(f"[runner] FAILED seed={entry['seed']} alpha={entry['alpha']}: "
                  f"{type(exc).__name__}: {exc}")
    return written
