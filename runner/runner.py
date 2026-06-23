"""Execute a surface-generation run described by a `RunConfig`.

`run_from_config` expands the seed/roughness sweep, builds the calculators once, and
drives the existing pipeline (`initialize_structure_* -> make_limits -> grow_structure
-> finalize_structure -> [saturate -> correct_charge] -> [rules] -> write`) for each
combination. `resolve_plan` produces the per-combination plan without building
calculators or running anything (used by `--dry-run`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

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

def _combo_seed(seed: int, alpha: Optional[float]) -> np.random.Generator:
    """A distinct, reproducible RNG per (seed, roughness) combo. Growth seeds numpy's
    global RNG from this generator, so mixing the roughness in keeps two combos that
    share a seed but differ in alpha from drawing correlated bond lengths."""
    alpha_bits = 0 if alpha is None else int(np.array([alpha], dtype=np.float64).view(np.uint64)[0])
    return np.random.default_rng([int(seed), alpha_bits])


def _init_structure(cfg: RunConfig):
    s = cfg.structure
    if s.from_file is not None:
        return initialize_structure_file(s.from_file, ase_read_kwargs=s.ase_read_kwargs,
                                         config=cfg.coordination)
    return initialize_structure_blank(cell=s.cell, pbc=s.pbc, config=cfg.coordination)


def _generate_one(cfg: RunConfig, seed: int, roughness: Optional[float],
                  out_path: Path, growth_calc, sat_calc):
    struct = _init_structure(cfg)
    struct.set_seed(_combo_seed(seed, roughness))
    _build_limits(struct, cfg, roughness)

    # Each slab writes its logs/trajectories into its own directory, so concurrent
    # runs never share a file.
    workdir = out_path.parent
    g, f, dbg = cfg.growth, cfg.finalize, cfg.debug

    def _finalize(calc):
        finalize_structure(struct, calculator=calc, fmax=f.fmax, max_steps=f.max_steps,
                           traj_interval=f.traj_interval, workdir=workdir,
                           write_trajectories=dbg.write_trajectories)

    grow_structure(
        amorphous_struct=struct,
        target_number_atoms=cfg.composition.target_number_atoms,
        target_ratios=cfg.composition.target_ratios,
        calculator=growth_calc,
        max_placement_attempts=g.max_placement_attempts,
        per_anchor_attempts=g.per_anchor_attempts,
        num_samples=g.num_samples,
        anneal_params=g.anneal,
        output_dir=workdir / "growth",
        workdir=workdir,
        write_growth_dumps=dbg.write_growth_dumps,
        write_trajectories=dbg.write_trajectories,
    )
    _finalize(growth_calc)

    if cfg.saturation.enabled:
        saturate_under_coordinated(struct, num_samples=cfg.saturation.num_samples)
        _finalize(sat_calc)
        if cfg.charge_correction.enabled:
            cc = cfg.charge_correction
            correct_charge(struct, max_iterations=cc.max_iterations,
                           num_samples=cc.num_samples, move_alpha=cc.move_alpha)
            _finalize(sat_calc)

    if cfg.rules:
        modifier = PeriodicStructureModifier(struct)
        for rule_spec in cfg.rules:
            modifier.add_rule(_build_rule(rule_spec))
        modifier.optimize()
        _finalize(sat_calc)

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
    manifest: list[dict] = []
    for entry in plan:
        out_path = entry["output_path"]
        record = {"seed": entry["seed"], "alpha": entry["alpha"],
                  "output_path": str(out_path), "status": "ok"}
        try:
            struct = _generate_one(cfg, entry["seed"], entry["alpha"], out_path, growth_calc, sat_calc)
            written.append(out_path)
            print(f"[runner] wrote {out_path}")
            if cfg.statistics.enabled:
                # Per structure: plots+stats.json (gated by per_structure) and a small
                # metrics.json (gated by pooled) that the disk-scan pooler gathers.
                _write_stats(struct, out_path.parent, cfg.saturation.enabled,
                             label=out_path.parent.name,
                             write_files=cfg.statistics.per_structure,
                             write_metrics_file=cfg.statistics.pooled)
        except Exception as exc:   # keep the sweep going; report what was skipped
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[runner] FAILED seed={entry['seed']} alpha={entry['alpha']}: "
                  f"{type(exc).__name__}: {exc}")
        manifest.append(record)

    _write_manifest(cfg, manifest)

    if cfg.statistics.enabled and cfg.statistics.pooled:
        _pool_stats(cfg)
    return written


def _write_manifest(cfg: RunConfig, records: list[dict]) -> Path:
    """Write a per-run ``manifest.json`` recording each combo's status/output/error.

    Makes a sweep auditable and resumable: a re-run can skip combos already marked
    ``ok`` and failures are visible without scraping stdout. Never aborts the run."""
    out_root = Path(cfg.run.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "manifest.json"
    payload = {
        "output_dir": str(out_root),
        "total": len(records),
        "succeeded": sum(1 for r in records if r["status"] == "ok"),
        "failed": sum(1 for r in records if r["status"] == "failed"),
        "combos": records,
    }
    try:
        path.write_text(json.dumps(payload, indent=2))
        print(f"[runner] wrote manifest to {path}")
    except Exception as exc:
        print(f"[runner] manifest FAILED: {type(exc).__name__}: {exc}")
    return path


def _write_stats(struct, out_dir, is_saturation: bool, label: str,
                 write_files: bool = True, write_metrics_file: bool = True):
    """Write per-structure statistics next to the structure. ``write_files`` controls the
    plots + stats.json; ``write_metrics_file`` controls the lightweight metrics.json that
    the disk-scan pooler later gathers. Analyses the structure once and reuses it for
    both. Never aborts the run on a stats failure."""
    if not (write_files or write_metrics_file):
        return None
    try:
        from stats import analyze_structure, write_metrics, write_report
        metrics = analyze_structure(struct, is_saturation=is_saturation)
        if write_metrics_file:
            write_metrics(out_dir, metrics, is_saturation=is_saturation, label=label)
        if write_files:
            write_report(struct, out_dir, is_saturation=is_saturation, label=label, metrics=metrics)
        return metrics
    except Exception as exc:
        print(f"[runner] statistics FAILED for {out_dir}: {type(exc).__name__}: {exc}")
        return None


def _pool_stats(cfg: RunConfig) -> None:
    """Build the pooled report by gathering per-structure metrics from disk. Reading off
    disk (not an in-memory list) makes pooling correct for parallel runs: structures
    produced by separate workers / SLURM array tasks are all included. Never aborts."""
    try:
        from stats import pool_reports_from_dir
        merged = pool_reports_from_dir(cfg.run.output_dir, cfg.saturation.enabled)
        if merged is not None:
            print(f"[runner] wrote pooled statistics to {cfg.run.output_dir}")
        else:
            print(f"[runner] pooled statistics skipped: fewer than 2 metrics.json under "
                  f"{cfg.run.output_dir}")
    except Exception as exc:
        print(f"[runner] pooled statistics FAILED: {type(exc).__name__}: {exc}")


def pool_from_config(source: Union[str, Path, dict, RunConfig]) -> None:
    """Reduce step: gather per-structure metrics already on disk under the config's
    output_dir and (re)write the pooled report. Run once after a parallel/sharded sweep
    (where each task generates its slabs but the single pooled report is built here)."""
    cfg = source if isinstance(source, RunConfig) else load_config(source)
    if not (cfg.statistics.enabled and cfg.statistics.pooled):
        print("[runner] statistics.pooled is disabled; nothing to pool")
        return
    _pool_stats(cfg)
