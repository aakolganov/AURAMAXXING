"""Execute a surface-generation run described by a `RunConfig`.

`run_from_config` expands the seed/roughness sweep, builds the calculators once, and
drives the existing pipeline (`initialize_structure_* -> make_limits -> grow_structure
-> finalize_structure -> [saturate -> correct_charge] -> [rules] -> write`) for each
combination. `resolve_plan` produces the per-combination plan without building
calculators or running anything (used by `--dry-run`).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Union

import numpy as np

from base.initialize import initialize_structure_blank, initialize_structure_file
from base.limits import make_limit_flat, make_limits_fourier, fix_limits
from growth.new_growth import grow_structure, finalize_structure
from saturation.new_sat import saturate_under_coordinated, correct_charge, relabel_caps
from rules import PeriodicStructureModifier, AvoidMotifSwapRule, MinimumDistanceRule

from .config import (RunConfig, RunSpec, CalculatorSpec, LimitSideSpec, load_config,
                     DEFAULT_NEG_FRAGMENT, DEFAULT_POS_FRAGMENT)

_EXT = {"vasp": "vasp", "xyz": "xyz"}


# --- calculator factory -------------------------------------------------------------

def build_calculator(spec: CalculatorSpec):
    """Instantiate a calculator backend from its spec. Backends are imported lazily so
    that loading/validating a config never requires LAMMPS, torch or MACE."""
    opts = {k: v for k, v in spec.options.items() if k not in {"type", "addons"}}
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


def _build_addon(calc_type: str, opts: dict):
    """Instantiate a bare ASE Calculator from calculators."""
    import importlib
    mod = importlib.import_module("calculators")
    cls = getattr(mod, calc_type)
    return cls(**opts)


def _compose_addons(interface, addons: list) -> None:
    """Build each addon calculator and compose it onto *interface* via add_calc.

    Addons are bare ASE Calculator objects from calculators, not
    full interfaces. Each is instantiated directly from its type string and
    kwargs, then summed into the interface's internal calculator.
    """
    for addon_spec_dict in addons:
        addon_type = addon_spec_dict.get("type")
        if addon_type is None:
            raise ValueError("addon calculator spec missing required type")
        addon = _build_addon(addon_type, {k: v for k, v in addon_spec_dict.items() if k != "type"})
        interface.add_calc(addon)


# --- sweep / plan -------------------------------------------------------------------

def _alpha_label(alpha: Optional[float]) -> str:
    # 5 decimals so a dense roughness sweep (e.g. hundreds of log-spaced alphas over
    # 0.01-1.0, nearest-neighbour gap ~1e-4) keeps a unique directory per combo. The old
    # 3-decimal label silently collided near the low end (0.0100, 0.0102 -> both "0.010").
    return f"{alpha:.5f}" if alpha is not None else "flat"


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

    # A combo's directory is seed{seed}_alpha{label}; if two combos round to the same
    # label they would silently overwrite each other's structure. Fail loudly instead.
    seen: dict = {}
    for e in plan:
        key = e["output_path"]
        if key in seen:
            s0, a0 = seen[key]
            raise ValueError(
                f"run: sweep combos (seed={s0}, alpha={a0}) and (seed={e['seed']}, "
                f"alpha={e['alpha']}) map to the same output path {key} -- their directory "
                f"labels collide. Use roughness values that stay distinct to 5 decimals."
            )
        seen[key] = (e["seed"], e["alpha"])
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

    if f.two_step_opt:

        def _finalize(calc, traj_name: str = "final_opt.xyz"):
            # Step 1 using FIRE
            finalize_structure(
                struct,
                calculator=calc,
                fmax=f.two_step_fmax_1,  # eV/AA
                max_steps=f.max_steps,
                traj_interval=f.traj_interval,
                workdir=workdir,
                write_trajectories=dbg.write_trajectories,
                traj_name=traj_name,
                optimizer=f.two_step_optimizer_1,
            )
            # Step 2 using LBFGS
            finalize_structure(
                struct,
                calculator=calc,
                fmax=f.fmax,  # eV/AA
                max_steps=f.max_steps,
                traj_interval=f.traj_interval,
                workdir=workdir,
                write_trajectories=dbg.write_trajectories,
                traj_name=traj_name,
                optimizer="BFGS",
            )

    else:

        def _finalize(
            calc,
            traj_name: str = "final_opt.xyz",
        ):
            finalize_structure(
                struct,
                calculator=calc,
                fmax=f.fmax,
                max_steps=f.max_steps,
                traj_interval=f.traj_interval,
                workdir=workdir,
                write_trajectories=dbg.write_trajectories,
                traj_name=traj_name,
                optimizer="LBFGS",
            )

    grow_structure(
        amorphous_struct=struct,
        target_number_atoms=cfg.composition.target_number_atoms,
        target_ratios=cfg.composition.target_ratios,
        calculator=growth_calc,
        max_placement_attempts=g.max_placement_attempts,
        per_anchor_attempts=g.per_anchor_attempts,
        num_samples=g.num_samples,
        anneal_params=g.anneal,
        mode=g.mode,
        bridge_bias=g.bridge_bias,
        output_dir=workdir / "growth",
        workdir=workdir,
        write_growth_dumps=dbg.write_growth_dumps,
        write_trajectories=dbg.write_trajectories,
    )
    _finalize(growth_calc, traj_name="opt_growth")

    # Dump pre-saturation structure if requested
    if dbg.pre_saturation_dump:
        pre_sat_path = out_path.parent / f"pre_sat_structure.{_EXT[cfg.run.output_format]}"
        struct.atoms.write(str(pre_sat_path), format=cfg.run.output_format)
        print(f"[runner] wrote pre-saturation structure to {pre_sat_path}")

    if cfg.saturation.enabled:
        sat = cfg.saturation
        _dump = dbg.write_growth_dumps
        saturate_under_coordinated(struct, num_samples=sat.num_samples,
                                   dump_dir=(workdir / "saturation" if _dump else None))
        _finalize(sat_calc, traj_name="opt_saturation")
        if cfg.charge_correction.enabled:
            cc = cfg.charge_correction
            correct_charge(struct, max_iterations=cc.max_iterations,
                           num_samples=cc.num_samples, move_alpha=cc.move_alpha,
                           dump_dir=(workdir / "charge" if _dump else None))
            _finalize(sat_calc, traj_name="opt_charge")
        # Swap the engine's -OH/H reference caps for the run's fragments (e.g. F, Na). No-op for
        # the default OH/H, so the default path's geometry and relax cost are unchanged.
        if (sat.negative_fragment != DEFAULT_NEG_FRAGMENT
                or sat.positive_fragment != DEFAULT_POS_FRAGMENT):
            relabel_caps(struct, negative_fragment=sat.negative_fragment,
                         positive_fragment=sat.positive_fragment)
            _finalize(sat_calc, traj_name="opt_relabel")

    if cfg.rules:
        modifier = PeriodicStructureModifier(struct)
        for rule_spec in cfg.rules:
            modifier.add_rule(_build_rule(rule_spec))
        modifier.optimize()
        _finalize(sat_calc, traj_name="opt_rules")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    struct.atoms.write(str(out_path), format=cfg.run.output_format)

    # Step 3 (optional): write VASP/CP2K DFT-refinement inputs next to the structure. The
    # writer Z-sorts a copy internally, so it is independent of the output above.
    if cfg.dft.enabled:
        from dft.api import write_dft_inputs
        write_dft_inputs(struct.atoms, cfg.dft, out_path.parent / "dft")
    return struct


def _shard_plan(plan: list, shard: Optional[int], num_shards: Optional[int]) -> list:
    """Return this task's slice of the plan. ``plan[shard::num_shards]`` is a deterministic
    round-robin partition: the union over shard in [0, num_shards) is the whole plan with no
    overlap, so independent SLURM array tasks cover everything exactly once."""
    if num_shards is None and shard is None:
        return plan
    if num_shards is None or shard is None:
        raise ValueError("--shard and --num-shards must be given together")
    if num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if not (0 <= shard < num_shards):
        raise ValueError(f"--shard must be in [0, {num_shards}), got {shard}")
    return plan[shard::num_shards]


def _partition_resume(plan: list, resume: bool) -> tuple[list, list]:
    """Split the plan into (todo, already_done). With ``resume`` on, an entry whose output
    file already exists is skipped, so an interrupted/failed sweep can be re-run without
    redoing finished slabs. Skipped entries still appear in the manifest (status 'skipped')
    and their on-disk metrics.json are still picked up by the pooled report."""
    if not resume:
        return plan, []
    todo, done = [], []
    for entry in plan:
        (done if entry["output_path"].exists() else todo).append(entry)
    return todo, done


def _skipped_record(entry: dict) -> dict:
    return {"seed": entry["seed"], "alpha": entry["alpha"],
            "output_path": str(entry["output_path"]), "status": "skipped"}


def _run_entry(cfg: RunConfig, entry: dict, growth_calc, sat_calc) -> dict:
    """Generate one slab and write its statistics. Returns a manifest record (never raises);
    a failure is captured in the record so a parallel sweep keeps going."""
    out_path = entry["output_path"]
    record = {"seed": entry["seed"], "alpha": entry["alpha"],
              "output_path": str(out_path), "status": "ok"}
    try:
        struct = _generate_one(cfg, entry["seed"], entry["alpha"], out_path, growth_calc, sat_calc)
        print(f"[runner] wrote {out_path}")
        if cfg.statistics.enabled:
            # Per structure: plots+stats.json (gated by per_structure) and a small
            # metrics.json (gated by pooled) that the disk-scan pooler gathers.
            surf = cfg.statistics.surface
            _write_stats(struct, out_path.parent, cfg.saturation.enabled,
                         label=out_path.parent.name,
                         write_files=cfg.statistics.per_structure,
                         write_metrics_file=cfg.statistics.pooled,
                         surface_opts={"enabled": surf.enabled, "probe": surf.probe,
                                       "n_points": surf.n_points, "overrides": surf.vdw_overrides})
    except Exception as exc:   # keep the sweep going; report what was skipped
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[runner] FAILED seed={entry['seed']} alpha={entry['alpha']}: "
              f"{type(exc).__name__}: {exc}")
    return record


def _default_calc_provider(cfg: RunConfig):
    """Build the (growth, saturation) calculators from the config.

    Each calculator's addons are built and composed via add_calc,
    so all energies/forces are summed before use.  Calculators
    (LAMMPS/torch) don't pickle, so this runs inside each worker
    rather than in the parent.
    """
    growth = build_calculator(cfg.calculators.growth)
    _compose_addons(growth, cfg.calculators.growth.addons)
    if cfg.calculators.saturation is None:
        return growth, growth

    sat = build_calculator(cfg.calculators.saturation)
    _compose_addons(sat, cfg.calculators.saturation.addons)
    return growth, sat



# Per-worker state for the in-node process pool: the calculators are built once per worker
# process (see _default_calc_provider) and stored here.
_WORKER: dict = {}


def _worker_init(cfg: RunConfig, threads: Optional[int], calc_provider) -> None:
    from .threads import configure_threads
    configure_threads(threads)
    growth, sat = calc_provider(cfg)
    _WORKER.update(cfg=cfg, growth=growth, sat=sat)


def _worker_run(entry: dict) -> dict:
    return _run_entry(_WORKER["cfg"], entry, _WORKER["growth"], _WORKER["sat"])


def run_from_config(source: Union[str, Path, dict, RunConfig], *,
                    workers: int = 1, threads: Optional[int] = None,
                    shard: Optional[int] = None, num_shards: Optional[int] = None,
                    resume: bool = False, calc_provider=None) -> list[Path]:
    """Run the full pipeline for every seed/roughness combination in the config.

    ``shard``/``num_shards`` restrict this run to a deterministic slice of the plan (one
    SLURM array task per shard). ``workers`` runs that slice across an in-node process pool
    (each worker builds its own calculators); ``threads`` caps compute threads per process.
    ``resume`` skips combinations whose output already exists. ``calc_provider(cfg) ->
    (growth_calc, sat_calc)`` overrides how calculators are built; it must be a module-level
    (picklable) callable when ``workers > 1``.

    Returns the list of present output paths (newly written + resumed). A combination that
    fails is logged and skipped rather than aborting the whole sweep. When sharded, the pooled
    report is left to a final ``--pool-only`` reduce (no single task sees every structure).
    """
    cfg = source if isinstance(source, RunConfig) else load_config(source)
    plan = _shard_plan(resolve_plan(cfg), shard, num_shards)
    todo, done = _partition_resume(plan, resume)
    if done:
        print(f"[runner] resume: skipping {len(done)} already-written of {len(plan)} structures")
    provider = calc_provider or _default_calc_provider

    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                 initargs=(cfg, threads, provider)) as ex:
            records = list(ex.map(_worker_run, todo))
    else:
        from .threads import configure_threads
        configure_threads(threads)
        growth_calc, sat_calc = provider(cfg)
        records = [_run_entry(cfg, entry, growth_calc, sat_calc) for entry in todo]

    records += [_skipped_record(e) for e in done]
    written = [Path(r["output_path"]) for r in records if r["status"] in ("ok", "skipped")]
    _write_manifest(cfg, records, shard=shard, num_shards=num_shards)

    # A single shard only ever sees its own slabs, so pooling there would be incomplete and
    # would race other shards on stats_pooled.json. Pool after generation in the unsharded
    # case; otherwise run `--pool-only` once after all array tasks finish.
    if cfg.statistics.enabled and cfg.statistics.pooled:
        if num_shards is None:
            _pool_stats(cfg)
        else:
            print(f"[runner] sharded run: skipping pooled statistics; run "
                  f"`--pool-only` over {cfg.run.output_dir} after all shards finish")

    # Sweep-level DFT SLURM array script(s): like pooled stats, only the unsharded run sees
    # every structure; a sharded sweep emits them from `--pool-only` once all tasks finish.
    if cfg.dft.enabled and cfg.dft.slurm:
        if num_shards is None:
            _write_dft_slurm(cfg)
        else:
            print(f"[runner] sharded run: skipping DFT SLURM script; run `--pool-only` over "
                  f"{cfg.run.output_dir} after all shards finish")
    return written


def _write_manifest(cfg: RunConfig, records: list[dict],
                    shard: Optional[int] = None, num_shards: Optional[int] = None) -> Path:
    """Write a manifest recording each combo's status/output/error.

    Makes a sweep auditable and resumable: a re-run can skip combos already marked ``ok``
    and failures are visible without scraping stdout. Sharded tasks each write their own
    ``manifest.shard{i}of{N}.json`` (no overwrite); ``--pool-only`` later merges them into
    ``manifest.json``. Never aborts the run."""
    out_root = Path(cfg.run.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    name = "manifest.json" if num_shards is None else f"manifest.shard{shard}of{num_shards}.json"
    path = out_root / name
    payload = {
        "output_dir": str(out_root),
        "shard": shard,
        "num_shards": num_shards,
        "total": len(records),
        "succeeded": sum(1 for r in records if r["status"] == "ok"),
        "skipped": sum(1 for r in records if r["status"] == "skipped"),
        "failed": sum(1 for r in records if r["status"] == "failed"),
        "combos": records,
    }
    try:
        path.write_text(json.dumps(payload, indent=2))
        print(f"[runner] wrote manifest to {path}")
    except Exception as exc:
        print(f"[runner] manifest FAILED: {type(exc).__name__}: {exc}")
    return path


def _merge_manifests(out_root: Path) -> Optional[Path]:
    """Combine per-shard manifests under ``out_root`` into a single ``manifest.json``.
    Idempotent; returns the merged path, or None if there were no shard manifests."""
    shard_files = sorted(out_root.glob("manifest.shard*of*.json"))
    if not shard_files:
        return None
    combos: list = []
    for f in shard_files:
        combos.extend(json.loads(f.read_text()).get("combos", []))
    payload = {
        "output_dir": str(out_root),
        "shards": len(shard_files),
        "total": len(combos),
        "succeeded": sum(1 for c in combos if c["status"] == "ok"),
        "skipped": sum(1 for c in combos if c["status"] == "skipped"),
        "failed": sum(1 for c in combos if c["status"] == "failed"),
        "combos": combos,
    }
    path = out_root / "manifest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _write_stats(struct, out_dir, is_saturation: bool, label: str,
                 write_files: bool = True, write_metrics_file: bool = True,
                 surface_opts: Optional[dict] = None):
    """Write per-structure statistics next to the structure. ``write_files`` controls the
    plots + stats.json; ``write_metrics_file`` controls the lightweight metrics.json that
    the disk-scan pooler later gathers. ``surface_opts`` parameterises the cap areal-density
    metric. Analyses the structure once and reuses it for both. Never aborts the run on a
    stats failure."""
    if not (write_files or write_metrics_file):
        return None
    try:
        from stats import analyze_structure, write_metrics, write_report
        metrics = analyze_structure(struct, is_saturation=is_saturation, surface_opts=surface_opts)
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


def _write_dft_slurm(cfg: RunConfig) -> None:
    """Write the ready-to-edit DFT SLURM array script(s) over the full plan. Built from the
    DFT input dirs that exist on disk, so it covers exactly what was generated (across all
    shards/workers). Never aborts the run."""
    try:
        from dft.slurm import write_slurm_scripts
        dft_dirs = [entry["output_path"].parent / "dft" for entry in resolve_plan(cfg)]
        project = cfg.dft.cp2k.get("project", "amorphous_oxide")
        written = write_slurm_scripts(Path(cfg.run.output_dir), dft_dirs,
                                      cfg.dft.packages, cp2k_project=project)
        if written:
            print(f"[runner] wrote DFT SLURM array script(s): "
                  f"{', '.join(p.name for p in written)}")
    except Exception as exc:
        print(f"[runner] DFT SLURM script generation FAILED: {type(exc).__name__}: {exc}")


def pool_from_config(source: Union[str, Path, dict, RunConfig]) -> None:
    """Reduce step: gather per-structure metrics already on disk under the config's
    output_dir and (re)write the pooled report. Run once after a parallel/sharded sweep
    (where each task generates its slabs but the single pooled report is built here)."""
    cfg = source if isinstance(source, RunConfig) else load_config(source)
    merged = _merge_manifests(Path(cfg.run.output_dir))
    if merged is not None:
        print(f"[runner] merged shard manifests -> {merged}")
    if cfg.dft.enabled and cfg.dft.slurm:
        _write_dft_slurm(cfg)
    if not (cfg.statistics.enabled and cfg.statistics.pooled):
        print("[runner] statistics.pooled is disabled; nothing to pool")
        return
    _pool_stats(cfg)
