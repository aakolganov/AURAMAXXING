"""Declarative configuration for a surface-generation run.

`load_config(path)` parses a YAML file into a validated `RunConfig` (a tree of
dataclasses). The runner (`runner.runner`) consumes a `RunConfig`; nothing here imports
the heavy calculator backends, so configs can be loaded/validated without LAMMPS, torch
or MACE installed.

The schema is intentionally focused: geometry, composition, coordination, per-stage
calculators and the pipeline knobs. Deep physics (BKS parameters, bond-length
distributions, formal charges) stays in ``default_constants.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml

from base.config import CoordinationConfig


# --- validation helpers -------------------------------------------------------------

def _as_dict(value, where: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _check_keys(d: dict, allowed: set, where: str, required: tuple = ()) -> None:
    """Reject unknown keys and demand required ones, naming the offending path."""
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"{where}: missing required key(s) {missing}")


def _check_choice(value, choices: set, where: str):
    if value not in choices:
        raise ValueError(f"{where}: {value!r} is not one of {sorted(choices)}")
    return value


# --- section dataclasses ------------------------------------------------------------

@dataclass
class StructureSpec:
    """Starting point: an empty box (``cell``) or a loaded file (``from_file``)."""
    cell: Optional[list] = None
    pbc: Optional[list] = None
    from_file: Optional[str] = None
    ase_read_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "StructureSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"cell", "pbc", "from_file", "ase_read_kwargs"}, where)
        if bool(d.get("cell")) == bool(d.get("from_file")):
            raise ValueError(f"{where}: provide exactly one of 'cell' (blank box) or 'from_file'")
        return cls(
            cell=d.get("cell"),
            pbc=d.get("pbc"),
            from_file=d.get("from_file"),
            ase_read_kwargs=d.get("ase_read_kwargs", {}) or {},
        )


@dataclass
class CompositionSpec:
    target_ratios: dict
    target_number_atoms: int

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "CompositionSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"target_ratios", "target_number_atoms"}, where,
                    required=("target_ratios", "target_number_atoms"))
        return cls(target_ratios=dict(d["target_ratios"]),
                   target_number_atoms=int(d["target_number_atoms"]))


@dataclass
class LimitSideSpec:
    """One growth boundary (top or bottom)."""
    type: str = "flat"             # flat | fourier | surface
    z: Optional[float] = None      # flat: absolute height
    z_av: Optional[float] = None   # fourier: absolute mean height
    offset: Optional[float] = None # height relative to the loaded surface top
    alpha: Optional[float] = None  # fourier roughness (smaller = rougher)
    n_max: int = 6
    m_max: int = 6

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "LimitSideSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"type", "z", "z_av", "offset", "alpha", "n_max", "m_max"}, where)
        kind = _check_choice(d.get("type", "flat"), {"flat", "fourier", "surface"}, f"{where}.type")
        if kind == "flat" and d.get("z") is None and d.get("offset") is None:
            raise ValueError(f"{where}: a flat limit needs 'z' (or 'offset' above a loaded surface)")
        if kind == "fourier":
            if d.get("alpha") is None:
                raise ValueError(f"{where}: a fourier limit needs 'alpha'")
            if d.get("z_av") is None and d.get("offset") is None:
                raise ValueError(f"{where}: a fourier limit needs 'z_av' or 'offset'")
        return cls(type=kind, z=d.get("z"), z_av=d.get("z_av"), offset=d.get("offset"),
                   alpha=d.get("alpha"), n_max=int(d.get("n_max", 6)), m_max=int(d.get("m_max", 6)))


@dataclass
class LimitsSpec:
    bottom: LimitSideSpec
    top: LimitSideSpec
    fix: Optional[str] = None      # bottom | top | None

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "LimitsSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"bottom", "top", "fix"}, where, required=("bottom", "top"))
        fix = d.get("fix")
        if fix is not None:
            _check_choice(fix, {"bottom", "top"}, f"{where}.fix")
        return cls(bottom=LimitSideSpec.from_dict(d["bottom"], f"{where}.bottom"),
                   top=LimitSideSpec.from_dict(d["top"], f"{where}.top"),
                   fix=fix)


@dataclass
class CalculatorSpec:
    """A calculator backend: ``type`` plus backend-specific options (kwargs)."""
    type: str
    options: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "CalculatorSpec":
        d = _as_dict(d, where)
        if "type" not in d:
            raise ValueError(f"{where}: missing required key 'type'")
        kind = _check_choice(d["type"], {"lammps", "mace", "uma"}, f"{where}.type")
        return cls(type=kind, options={k: v for k, v in d.items() if k != "type"})


@dataclass
class CalculatorsSpec:
    growth: CalculatorSpec
    saturation: Optional[CalculatorSpec] = None   # None -> reuse the growth calculator

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "CalculatorsSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"growth", "saturation"}, where, required=("growth",))
        sat = d.get("saturation")
        return cls(growth=CalculatorSpec.from_dict(d["growth"], f"{where}.growth"),
                   saturation=CalculatorSpec.from_dict(sat, f"{where}.saturation") if sat else None)


@dataclass
class GrowthSpec:
    max_placement_attempts: int = 1000
    per_anchor_attempts: int = 100
    num_samples: int = 100
    anneal: Optional[dict] = None   # {T_ini, T_fin, steps, interval}; None -> grow defaults

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "GrowthSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"max_placement_attempts", "per_anchor_attempts", "num_samples", "anneal"}, where)
        anneal = d.get("anneal")
        if anneal is not None:
            _check_keys(_as_dict(anneal, f"{where}.anneal"),
                        {"T_ini", "T_fin", "steps", "interval"}, f"{where}.anneal")
        return cls(max_placement_attempts=int(d.get("max_placement_attempts", 1000)),
                   per_anchor_attempts=int(d.get("per_anchor_attempts", 100)),
                   num_samples=int(d.get("num_samples", 100)),
                   anneal=anneal)


@dataclass
class FinalizeSpec:
    fmax: float = 0.1
    max_steps: int = 500
    traj_interval: int = 1

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "FinalizeSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"fmax", "max_steps", "traj_interval"}, where)
        return cls(fmax=float(d.get("fmax", 0.1)),
                   max_steps=int(d.get("max_steps", 500)),
                   traj_interval=int(d.get("traj_interval", 1)))


@dataclass
class SaturationSpec:
    enabled: bool = False
    num_samples: int = 250

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "SaturationSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"enabled", "num_samples"}, where)
        return cls(enabled=bool(d.get("enabled", False)),
                   num_samples=int(d.get("num_samples", 250)))


@dataclass
class ChargeCorrectionSpec:
    enabled: bool = False
    max_iterations: int = 1000
    num_samples: int = 250
    move_alpha: float = 0.5

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "ChargeCorrectionSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"enabled", "max_iterations", "num_samples", "move_alpha"}, where)
        return cls(enabled=bool(d.get("enabled", False)),
                   max_iterations=int(d.get("max_iterations", 1000)),
                   num_samples=int(d.get("num_samples", 250)),
                   move_alpha=float(d.get("move_alpha", 0.5)))


@dataclass
class StatisticsSpec:
    enabled: bool = True               # master switch for any statistics output
    per_structure: bool = True         # write per-structure plots + stats.json in each dir
    pooled: bool = True                # write one pooled plot set across the sweep

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "StatisticsSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"enabled", "per_structure", "pooled"}, where)
        return cls(enabled=bool(d.get("enabled", True)),
                   per_structure=bool(d.get("per_structure", True)),
                   pooled=bool(d.get("pooled", True)))


@dataclass
class RuleSpec:
    type: str
    options: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "RuleSpec":
        d = _as_dict(d, where)
        if "type" not in d:
            raise ValueError(f"{where}: missing required key 'type'")
        kind = _check_choice(d["type"], {"avoid_motif", "min_distance"}, f"{where}.type")
        return cls(type=kind, options={k: v for k, v in d.items() if k != "type"})


@dataclass
class RunSpec:
    seeds: list = field(default_factory=lambda: [0])
    roughness: Optional[list] = None   # None -> use the top limit's own alpha
    output_dir: str = "output"
    output_format: str = "vasp"        # vasp | xyz

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "RunSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"seeds", "roughness", "output_dir", "output_format"}, where)
        fmt = _check_choice(d.get("output_format", "vasp"), {"vasp", "xyz"}, f"{where}.output_format")
        seeds = d.get("seeds", [0])
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"{where}.seeds: expected a non-empty list of ints")
        rough = d.get("roughness")
        if rough is not None and (not isinstance(rough, list) or not rough):
            raise ValueError(f"{where}.roughness: expected a non-empty list of floats or omit it")
        return cls(seeds=[int(s) for s in seeds],
                   roughness=[float(a) for a in rough] if rough is not None else None,
                   output_dir=str(d.get("output_dir", "output")),
                   output_format=fmt)


# --- coordination merge -------------------------------------------------------------

def _parse_cut_offs(raw: dict, where: str) -> dict:
    """Expand ``{"Si-O": 2.0}`` into symmetric tuple-keyed pairs {(Si,O):2.0,(O,Si):2.0}."""
    out = {}
    for key, val in _as_dict(raw, where).items():
        parts = str(key).split("-")
        if len(parts) != 2:
            raise ValueError(f"{where}: cut_off key {key!r} must be 'Element-Element' (e.g. 'Si-O')")
        a, b = parts
        out[(a, b)] = float(val)
        out[(b, a)] = float(val)
    return out


def _build_coordination(d: Optional[dict], where: str) -> CoordinationConfig:
    """Merge a partial coordination block onto the constant defaults."""
    cfg = CoordinationConfig()   # fresh copies of the defaults
    if not d:
        return cfg
    d = _as_dict(d, where)
    _check_keys(d, {"max_cn", "min_cn", "cut_offs", "overcoord_policy"}, where)
    if "max_cn" in d:
        cfg.max_cn.update({k: int(v) for k, v in _as_dict(d["max_cn"], f"{where}.max_cn").items()})
    if "min_cn" in d:
        cfg.min_cn.update({k: int(v) for k, v in _as_dict(d["min_cn"], f"{where}.min_cn").items()})
    if "cut_offs" in d:
        cfg.cut_offs.update(_parse_cut_offs(d["cut_offs"], f"{where}.cut_offs"))
    if "overcoord_policy" in d:
        cfg.overcoord_policy.update(_as_dict(d["overcoord_policy"], f"{where}.overcoord_policy"))
    return cfg


# --- top-level config ---------------------------------------------------------------

@dataclass
class RunConfig:
    structure: StructureSpec
    composition: CompositionSpec
    limits: LimitsSpec
    calculators: CalculatorsSpec
    coordination: CoordinationConfig = field(default_factory=CoordinationConfig)
    growth: GrowthSpec = field(default_factory=GrowthSpec)
    finalize: FinalizeSpec = field(default_factory=FinalizeSpec)
    saturation: SaturationSpec = field(default_factory=SaturationSpec)
    charge_correction: ChargeCorrectionSpec = field(default_factory=ChargeCorrectionSpec)
    statistics: StatisticsSpec = field(default_factory=StatisticsSpec)
    rules: list = field(default_factory=list)
    run: RunSpec = field(default_factory=RunSpec)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = _as_dict(d, "config")
        _check_keys(
            d,
            {"structure", "composition", "limits", "calculators", "coordination",
             "growth", "finalize", "saturation", "charge_correction", "statistics",
             "rules", "run"},
            "config",
            required=("structure", "composition", "limits", "calculators"),
        )
        rules = d.get("rules", []) or []
        if not isinstance(rules, list):
            raise ValueError("config.rules: expected a list of rule mappings")
        return cls(
            structure=StructureSpec.from_dict(d["structure"], "structure"),
            composition=CompositionSpec.from_dict(d["composition"], "composition"),
            limits=LimitsSpec.from_dict(d["limits"], "limits"),
            calculators=CalculatorsSpec.from_dict(d["calculators"], "calculators"),
            coordination=_build_coordination(d.get("coordination"), "coordination"),
            growth=GrowthSpec.from_dict(d.get("growth", {}), "growth"),
            finalize=FinalizeSpec.from_dict(d.get("finalize", {}), "finalize"),
            saturation=SaturationSpec.from_dict(d.get("saturation", {}), "saturation"),
            charge_correction=ChargeCorrectionSpec.from_dict(d.get("charge_correction", {}), "charge_correction"),
            statistics=StatisticsSpec.from_dict(d.get("statistics", {}), "statistics"),
            rules=[RuleSpec.from_dict(r, f"rules[{i}]") for i, r in enumerate(rules)],
            run=RunSpec.from_dict(d.get("run", {}), "run"),
        )


def load_config(source: Union[str, Path, dict]) -> RunConfig:
    """Load and validate a run configuration from a YAML file path or a dict."""
    if isinstance(source, dict):
        data = source
    else:
        with open(source) as fh:
            data = yaml.safe_load(fh)
        if data is None:
            raise ValueError(f"{source}: empty config file")
    return RunConfig.from_dict(data)
