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

import math
import re
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Optional, Union

import yaml
from ase.formula import Formula

from base.config import CoordinationConfig
from base.element_data import (OXIDE_ELEMENTS, build_element_tables, element_record,
                               normalize_cn_distr, DEFAULT_DISTANCE_KNOBS)


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
        from_file = d.get("from_file")
        if from_file is not None and not Path(from_file).exists():
            raise FileNotFoundError(f"{where}.from_file: structure file not found: {from_file}")
        return cls(
            cell=d.get("cell"),
            pbc=d.get("pbc"),
            from_file=d.get("from_file"),
            ase_read_kwargs=d.get("ase_read_kwargs", {}) or {},
        )


@dataclass
class CompositionSpec:
    """Canonical, charge-neutral integer per-element target counts for growth.

    Built from one input form (formula / elemental ratio / structural mix / explicit counts,
    or the legacy target_ratios), then adjusted to the nearest charge-neutral integer
    composition. ``target_ratios`` and ``target_number_atoms`` are exposed for the grow loop:
    the integer counts are the relative weights the deficit picker converges to, and their
    sum is the target size.
    """
    target_counts: dict
    requested_counts: dict
    adjusted: bool

    @property
    def target_number_atoms(self) -> int:
        return sum(self.target_counts.values())

    @property
    def target_ratios(self) -> dict:
        return dict(self.target_counts)


# --- composition parsing ------------------------------------------------------------

def _formula_counts(formula, where: str) -> dict:
    """Element counts of a brutto formula (e.g. 'Al2O3' -> {Al: 2, O: 3}) via ase.formula."""
    try:
        counts = Formula(str(formula)).count()
    except Exception as exc:
        raise ValueError(f"{where}: could not parse formula {formula!r}: {exc}")
    return {el: int(n) for el, n in counts.items()}


def _parse_mix_string(s: str, where: str) -> list:
    """Parse a structural-mix string: 'SiO2:Al2O3=7:1' or '70% SiO2 + 30% Al2O3'."""
    s = s.strip()
    if "=" in s:                                   # 'A:B=x:y'
        comps, ratios = s.split("=", 1)
        comp_list = [c.strip() for c in comps.split(":")]
        ratio_list = [r.strip() for r in ratios.split(":")]
        if len(comp_list) != len(ratio_list):
            raise ValueError(f"{where}.mix: '{s}' has {len(comp_list)} components but "
                             f"{len(ratio_list)} ratios")
        return [(c, float(r)) for c, r in zip(comp_list, ratio_list)]
    if "+" in s or "%" in s:                       # 'x% A + y% B'
        items = []
        for token in s.split("+"):
            token = token.strip()
            m = re.match(r"^([0-9]*\.?[0-9]+)\s*%?\s*(.+)$", token)
            if not m:
                raise ValueError(f"{where}.mix: could not parse component {token!r} in {s!r}")
            items.append((m.group(2).strip(), float(m.group(1))))
        return items
    raise ValueError(f"{where}.mix: unrecognised mix {s!r}; use 'A:B=x:y', 'x% A + y% B', "
                     f"or a mapping {{formula: weight}}")


def _parse_mix(mix, where: str) -> dict:
    """Expand a structural mix into element weights (sum of each component's formula counts
    scaled by its weight)."""
    if isinstance(mix, dict):
        items = [(str(f), float(w)) for f, w in mix.items()]
    elif isinstance(mix, str):
        items = _parse_mix_string(mix, where)
    else:
        raise ValueError(f"{where}.mix: expected a mapping or a string, got {type(mix).__name__}")
    weights: dict = {}
    for formula, w in items:
        for el, c in _formula_counts(formula, f"{where}.mix").items():
            weights[el] = weights.get(el, 0.0) + w * c
    return weights


def _parse_composition_block(d: dict, where: str) -> dict:
    """Parse a composition block into element weights (+ total) or explicit counts. Exactly
    one of formula / ratio / mix / counts (or the legacy target_ratios) must be present."""
    d = _as_dict(d, where)
    _check_keys(d, {"formula", "ratio", "mix", "counts", "total_atoms",
                    "target_ratios", "target_number_atoms"}, where)
    forms = [k for k in ("formula", "ratio", "mix", "counts", "target_ratios") if k in d]
    if len(forms) != 1:
        raise ValueError(f"{where}: provide exactly one of 'formula', 'ratio', 'mix', "
                         f"'counts' (or legacy 'target_ratios'); got {forms}")
    form = forms[0]

    # The total-atoms key must match the form, so a mismatched/extra total key isn't silently
    # ignored: 'counts' takes neither; legacy 'target_ratios' takes 'target_number_atoms';
    # the other forms take 'total_atoms'.
    expected_total = None if form == "counts" else ("target_number_atoms" if form == "target_ratios" else "total_atoms")
    stray_total = ({"total_atoms", "target_number_atoms"} - {expected_total}) & set(d)
    if stray_total:
        raise ValueError(f"{where}: composition form {form!r} does not accept {sorted(stray_total)}"
                         + (f"; use {expected_total!r}" if expected_total else " (explicit counts need no total)"))

    if form == "counts":                           # explicit counts: no total needed
        counts = {str(k): int(v) for k, v in _as_dict(d["counts"], f"{where}.counts").items()}
        if any(v < 0 for v in counts.values()) or sum(counts.values()) == 0:
            raise ValueError(f"{where}.counts: need positive integer atom counts")
        return {"weights": None, "counts": counts, "total": None, "elements": set(counts)}

    total_key = expected_total
    if total_key not in d:
        raise ValueError(f"{where}: composition form {form!r} needs '{total_key}'")
    total = int(d[total_key])
    if total <= 0:
        raise ValueError(f"{where}.{total_key}: must be a positive integer")

    if form == "formula":
        weights = {el: float(c) for el, c in _formula_counts(d["formula"], where).items()}
    elif form in ("ratio", "target_ratios"):
        weights = {str(k): float(v) for k, v in _as_dict(d[form], f"{where}.{form}").items()}
    else:                                          # mix
        weights = _parse_mix(d["mix"], where)

    if not weights or sum(weights.values()) <= 0:
        raise ValueError(f"{where}: composition has no positive element weights")
    return {"weights": weights, "counts": None, "total": total, "elements": set(weights)}


def _nearest_neutral_counts(x: dict, oxidation: dict, where: str):
    """Integer per-element counts closest (least-squares) to the targets ``x`` whose net
    formal charge is zero. Brute-forces the small integer neighbourhood of round(x); exact for
    the handful of species in an oxide composition. Returns (counts, requested, adjusted)."""
    elems = list(x)
    ox = [oxidation[e] for e in elems]
    n0 = [max(0, int(round(x[e]))) for e in elems]

    def search(kmax):
        best = None
        for deltas in product(range(-kmax, kmax + 1), repeat=len(elems)):
            n = [n0[i] + deltas[i] for i in range(len(elems))]
            if any(c < 0 for c in n) or sum(n) == 0:
                continue
            if sum(n[i] * ox[i] for i in range(len(elems))) != 0:
                continue
            cost = sum((n[i] - x[elems[i]]) ** 2 for i in range(len(elems)))
            key = (cost, sum(abs(dd) for dd in deltas))
            if best is None or key < best[0]:
                best = (key, n)
        return best

    # Grow the search box until it contains ANY neutral composition (an off-stoichiometry
    # request may need a large adjustment before the charge-balance lattice is reachable), then
    # expand once more to the radius that provably contains the *closest* one and search that.
    # For the least-squares optimum n*, (n*_i - x_i)**2 <= cost so |n*_i - x_i| <= sqrt(cost),
    # and n0 is within 0.5 of x, hence |n*_i - n0_i| <= sqrt(cost) + 0.5. Searching that radius
    # returns the true nearest neutral composition rather than merely the first the growing box
    # happened to hit (the old fixed (4, 8, 16) ladder both stopped short of the optimum and
    # raised a spurious "single charge sign" error for reachable-but-distant compositions).
    best = None
    searched = 0
    for kmax in (4, 8, 16, 32, 64, 128, 256):
        searched = kmax
        best = search(kmax)
        if best is not None:
            break
    if best is None:
        raise ValueError(
            f"{where}: cannot reach a charge-neutral integer composition for "
            f"{dict(zip(elems, ox))}; it must contain both cations (oxidation > 0) and "
            f"anions (oxidation < 0).")
    guaranteed = min(512, int(math.ceil(math.sqrt(best[0][0]) + 0.5)))
    if guaranteed > searched:
        best = search(guaranteed)
    counts = {e: c for e, c in zip(elems, best[1])}
    requested = {e: c for e, c in zip(elems, n0)}
    return counts, requested, (best[1] != n0)


def _finalize_composition(parsed: dict, oxidation: dict, where: str) -> CompositionSpec:
    if parsed["counts"] is not None:
        x = {e: float(c) for e, c in parsed["counts"].items()}
    else:
        weights = parsed["weights"]
        tot_w = sum(weights.values())
        x = {e: weights[e] / tot_w * parsed["total"] for e in weights}
    counts, requested, adjusted = _nearest_neutral_counts(x, oxidation, where)
    if adjusted:
        print(f"[composition] adjusted to the nearest charge-neutral integer counts: "
              f"{requested} -> {counts} (net formal charge 0)")
    return CompositionSpec(target_counts=counts, requested_counts=requested, adjusted=adjusted)


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
    addons: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "CalculatorSpec":
        d = _as_dict(d, where)
        if "type" not in d:
            raise ValueError(f"{where}: missing required key 'type'")

        opts = {k: v for k, v in d.items() if k not in {"type", "addons"}}
        addons_raw = d.get("addons")
        if addons_raw is not None and not isinstance(addons_raw, list):
            raise ValueError(f"{where}.addons: expected a list of calculator specs, got {type(addons_raw).__name__}")
        return cls(type=d["type"], options=opts, addons=[_as_dict(a, f"{where}.addons") for a in (addons_raw or [])])


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


DEFAULT_POS_FRAGMENT = ("H",)
DEFAULT_NEG_FRAGMENT = ("O", "H")


def _parse_fragment(raw, where: str) -> tuple:
    """A saturation fragment: an ordered list of element symbols (or a bare symbol). The first
    atom bonds to the surface site, each next atom to the previous one (e.g. ``[O, H]`` is the
    -OH group)."""
    if isinstance(raw, str):
        frag = (raw,)
    elif isinstance(raw, (list, tuple)) and raw and all(isinstance(e, str) for e in raw):
        frag = tuple(raw)
    else:
        raise ValueError(f"{where}: expected an element symbol or a non-empty list of symbols, "
                         f"got {raw!r}")
    return frag


def _validate_mode(mode: str, pos: tuple, neg: tuple, where: str) -> None:
    """A given ``mode`` must be consistent with the fragments (mode is otherwise inferred from
    them). anion_cap = cap cations with a non-default electronegative fragment, anions keep H;
    cation_cap = cap anions with a non-default electropositive fragment, cations keep -OH."""
    if mode == "anion_cap" and (pos != DEFAULT_POS_FRAGMENT or neg == DEFAULT_NEG_FRAGMENT):
        raise ValueError(f"{where}.mode 'anion_cap': set a non-default negative_fragment "
                         f"(e.g. F) and leave positive_fragment at the default H.")
    if mode == "cation_cap" and (neg != DEFAULT_NEG_FRAGMENT or pos == DEFAULT_POS_FRAGMENT):
        raise ValueError(f"{where}.mode 'cation_cap': set a non-default positive_fragment "
                         f"(e.g. Na) and leave negative_fragment at the default -OH ([O, H]).")


@dataclass
class SaturationSpec:
    enabled: bool = False
    num_samples: int = 250
    positive_fragment: tuple = DEFAULT_POS_FRAGMENT
    negative_fragment: tuple = DEFAULT_NEG_FRAGMENT
    mode: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "SaturationSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"enabled", "num_samples", "positive_fragment", "negative_fragment",
                        "mode"}, where)
        pos = _parse_fragment(d.get("positive_fragment", list(DEFAULT_POS_FRAGMENT)),
                              f"{where}.positive_fragment")
        neg = _parse_fragment(d.get("negative_fragment", list(DEFAULT_NEG_FRAGMENT)),
                              f"{where}.negative_fragment")
        mode = d.get("mode")
        if mode is not None:
            mode = _check_choice(mode, {"neutral", "anion_cap", "cation_cap"}, f"{where}.mode")
            _validate_mode(mode, pos, neg, where)
        return cls(enabled=bool(d.get("enabled", False)),
                   num_samples=int(d.get("num_samples", 250)),
                   positive_fragment=pos, negative_fragment=neg, mode=mode)


def _saturation_fragment_elements(sat_block: Optional[dict]) -> set:
    """Element symbols named in the saturation block's fragments, so the run's element tables
    include them (a non-network cap like F/Na needs distance/cutoff rows and an oxidation).
    Returns the full set; the caller subtracts the network elements to find the cap-only ones."""
    if not sat_block or not isinstance(sat_block, dict):
        return set()
    elems: set = set()
    for key in ("positive_fragment", "negative_fragment"):
        raw = sat_block.get(key)
        if raw is None:
            continue
        # A malformed (non-str/non-list) value is left for SaturationSpec.from_dict/_parse_fragment
        # to reject with the clear error, rather than raising a raw TypeError from list(raw) here.
        if isinstance(raw, str):
            frag = [raw]
        elif isinstance(raw, (list, tuple)):
            frag = list(raw)
        else:
            continue
        elems.update(str(e) for e in frag if isinstance(e, str))
    return elems


def _validate_fragments(sat: SaturationSpec, oxidation: dict, where: str) -> None:
    """Saturation fragments must be 1-valent (the engine swaps whole caps, so a +1/-1 fragment
    keeps the slab neutral), and a non-default fragment must be a single atom (the -OH default
    is the only multi-atom fragment supported). Net charge is summed from the run's oxidation."""
    for label, frag, want, default in (
            ("positive_fragment", sat.positive_fragment, +1, DEFAULT_POS_FRAGMENT),
            ("negative_fragment", sat.negative_fragment, -1, DEFAULT_NEG_FRAGMENT)):
        if frag != default and len(frag) != 1:
            raise ValueError(f"{where}.{label}: a non-default fragment must be a single element "
                             f"(1-valent cap); got {list(frag)}.")
        charge = sum(oxidation[e] for e in frag)
        if charge != want:
            raise ValueError(f"{where}.{label}: fragment {list(frag)} has net formal charge "
                             f"{charge:+d}, but saturation fragments must be 1-valent "
                             f"({want:+d}).")


def _validate_cn_distr_oxidation(cn_distr: dict, oxidation: dict, cap_elements: set,
                                 where: str) -> None:
    """Reject per-atom oxidation overrides (from cn_distr) that the saturation/charge stage
    cannot reconcile:

    - An override on a cap element (O, H, or a fragment element, in ``cap_elements``) puts a
      non-default charge on an atom the engine reuses as a saturation cap, which breaks the
      count-based charge-correction / relabel balance (the cap's formal charge no longer cancels
      its partner). Only checked when saturation is enabled (the caller passes an empty set
      otherwise).
    - An override whose sign differs from the element's base oxidation mis-classifies the atom as
      cation vs anion for both growth attachment and saturation site selection -- always rejected.
    """
    for el, variants in (cn_distr or {}).items():
        base = oxidation.get(el, 0)
        for v in variants:
            ox = v.get("oxidation")
            if ox is None:
                continue
            if el in cap_elements:
                raise ValueError(
                    f"{where}: a cn_distr 'oxidation' override on {el!r} is not allowed -- it is "
                    f"used as a saturation cap, and a per-atom charge on a cap breaks the "
                    f"charge-correction / relabel charge balance.")
            if base and (ox > 0) != (base > 0):
                raise ValueError(
                    f"{where}: cn_distr 'oxidation' {ox:+d} on {el!r} has the opposite sign of its "
                    f"base oxidation {base:+d}; that mis-classifies it as a "
                    f"{'cation' if ox > 0 else 'anion'} for growth and saturation.")


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
class SurfaceSpec:
    """Surface cap-group areal-concentration metric: caps per nm², normalised by the true
    van-der-Waals surface area (periodic Shrake–Rupley). ``probe`` 0 = bare VdW surface,
    ~1.4 Å = solvent-accessible; ``n_points`` is the per-atom sphere sampling (accuracy vs speed);
    ``vdw_overrides`` is an optional ``{element: radius Å}`` map."""
    enabled: bool = True
    probe: float = 0.0
    n_points: int = 200
    vdw_overrides: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "SurfaceSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"enabled", "probe", "n_points", "vdw_overrides"}, where)
        overrides = _as_dict(d.get("vdw_overrides", {}) or {}, f"{where}.vdw_overrides")
        return cls(enabled=bool(d.get("enabled", True)),
                   probe=float(d.get("probe", 0.0)),
                   n_points=int(d.get("n_points", 200)),
                   vdw_overrides={str(k): float(v) for k, v in overrides.items()})


@dataclass
class StatisticsSpec:
    enabled: bool = True               # master switch for any statistics output
    per_structure: bool = True         # write per-structure plots + stats.json in each dir
    pooled: bool = True                # write one pooled plot set across the sweep
    surface: SurfaceSpec = field(default_factory=SurfaceSpec)   # cap areal-density sub-metric

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "StatisticsSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"enabled", "per_structure", "pooled", "surface"}, where)
        return cls(enabled=bool(d.get("enabled", True)),
                   per_structure=bool(d.get("per_structure", True)),
                   pooled=bool(d.get("pooled", True)),
                   surface=SurfaceSpec.from_dict(d.get("surface", {}), f"{where}.surface"))


@dataclass
class DebugSpec:
    """Debug-only outputs. Off by default: the per-atom growth snapshots and the
    MD/optimization trajectories are large and I/O-heavy on big ensembles, and are
    mostly useful for diagnosing a single run."""
    write_growth_dumps: bool = False   # per-atom xyz snapshot after every placement
    write_trajectories: bool = False   # anneal + finalize trajectory files
    pre_saturation_dump: bool = False  # write the optimized, dry surface as well

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "DebugSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"write_growth_dumps", "write_trajectories", "pre_saturation_dump"}, where)
        return cls(
            write_growth_dumps=bool(d.get("write_growth_dumps", False)),
            write_trajectories=bool(d.get("write_trajectories", False)),
            pre_saturation_dump=bool(d.get("pre_saturation_dump", False))
            )


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


@dataclass
class DFTSpec:
    """Optional pipeline step 3: write VASP/CP2K geometry-optimization inputs for each
    generated structure. The ``vasp``/``cp2k`` blocks are free-form passthrough so a user
    can override our default tags or add their own (see ``dft.common.deep_merge``)."""
    enabled: bool = False
    packages: list = field(default_factory=lambda: ["vasp", "cp2k"])
    vasp: dict = field(default_factory=dict)
    cp2k: dict = field(default_factory=dict)
    slurm: bool = False   # also emit a ready-to-edit SLURM array submission script per package

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "DFTSpec":
        d = _as_dict(d, where)
        _check_keys(d, {"enabled", "packages", "vasp", "cp2k", "slurm"}, where)
        packages = d.get("packages", ["vasp", "cp2k"])
        if not isinstance(packages, list) or not packages:
            raise ValueError(f"{where}.packages: expected a non-empty list of package names")
        for p in packages:
            _check_choice(p, {"vasp", "cp2k"}, f"{where}.packages")
        vasp = _as_dict(d.get("vasp", {}) or {}, f"{where}.vasp")
        cp2k = _as_dict(d.get("cp2k", {}) or {}, f"{where}.cp2k")
        # Validate only the wrapper keys; the tag dicts under them are free-form passthrough
        # so a user can add arbitrary INCAR/CP2K tags.
        _check_keys(vasp, {"incar", "kpoints"}, f"{where}.vasp")
        _check_keys(cp2k, {"project", "basis_set_file", "potential_file", "input"}, f"{where}.cp2k")
        return cls(enabled=bool(d.get("enabled", False)),
                   packages=list(packages), vasp=dict(vasp), cp2k=dict(cp2k),
                   slurm=bool(d.get("slurm", False)))


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


def _coord_override_elements(coord_block: Optional[dict]) -> set:
    """Element symbols named anywhere in a coordination block, so the derived tables include
    them even when they are absent from the composition (e.g. a max_cn override for Al)."""
    if not coord_block:
        return set()
    elems: set = set()
    for key in ("max_cn", "min_cn", "oxidation", "cn_distr"):
        sub = coord_block.get(key)
        if isinstance(sub, dict):
            elems.update(str(k) for k in sub)
    cut = coord_block.get("cut_offs")
    if isinstance(cut, dict):
        for pair in cut:
            parts = str(pair).split("-")
            if len(parts) == 2:                 # ignore malformed keys here; _parse_cut_offs
                elems.update(parts)             # raises the clear 'Element-Element' error
    return elems


def _loaded_elements(structure_block: dict) -> set:
    """Element symbols in a loaded ``from_file`` structure (empty for a blank-box run). Read at
    config load so the coordination derivation includes a substrate's elements as spectators.
    A missing file returns empty -- StructureSpec.from_dict raises the clear FileNotFoundError."""
    ff = structure_block.get("from_file")
    if not ff or not Path(ff).exists():
        return set()
    from ase.io import read
    atoms = read(ff, **(structure_block.get("ase_read_kwargs") or {}))
    return set(atoms.get_chemical_symbols())


def _resolve_oxidation(elements, coord_block: Optional[dict], where: str) -> dict:
    """Per-element signed oxidation state: curated default, overridden by the coordination
    block. Raises the clear missing-element error for a non-curated, non-overridden element."""
    over = {}
    if coord_block and "oxidation" in coord_block:
        over = {str(k): int(v)
                for k, v in _as_dict(coord_block["oxidation"], f"{where}.oxidation").items()}
    oxidation = {}
    for e in elements:
        if e in over:
            oxidation[e] = over[e]
        elif e in OXIDE_ELEMENTS:
            oxidation[e] = OXIDE_ELEMENTS[e]["oxidation"]
        else:
            element_record(e)   # raises the clear missing-element error
    return oxidation


def _parse_distance_knobs(raw, where: str) -> dict:
    d = _as_dict(raw, where)
    _check_keys(d, {"bond_factor", "bond_dev", "cutoff_pad", "collision_factor"}, where)
    return {k: float(v) for k, v in d.items()}


def _parse_cn_distr(raw, where: str) -> dict:
    """Validate + normalize the per-element coordination-number distribution. Each element maps
    to a single CN, a ``{cn: fraction}`` mapping, or a list of ``{cn, fraction, oxidation?}``;
    a fraction of that element is grown at each CN (and optional oxidation), the remainder at
    the curated default."""
    d = _as_dict(raw, where)
    out = {}
    for el, spec in d.items():
        try:
            out[str(el)] = normalize_cn_distr({el: spec})[str(el)]
        except ValueError as exc:
            raise ValueError(f"{where}.{el}: {exc}")
    return out


def _validate_cutoff_overrides(overrides: dict, cfg: CoordinationConfig, where: str) -> None:
    """A raw cut_off override must not fall below the derived bond-sampling upper bound for the
    pair, or a sampled bond could be committed yet scored as non-bonded (the coherence bug).
    Pairs whose elements are not in the derived set are left to the user."""
    for (a, b), val in overrides.items():
        rs = cfg.sample_dist.get(a)
        if rs is None or b not in rs:
            continue
        hi = float(dict.__getitem__(rs, b).support()[1])
        if val < hi:
            raise ValueError(
                f"{where}: cut_off {a}-{b}={val} is below the derived bond-sampling upper "
                f"bound {hi:.3f}; a sampled bond could then be scored as non-bonded. Raise the "
                f"cut_off, or use a 'distance' override to move the whole window.")


def _build_coordination(d: Optional[dict], where: str, *, elements, oxidation,
                        spectators=(), terminal_cn=()) -> CoordinationConfig:
    """Derive the covalent-radii distance tables + curated coordination defaults for the run's
    element set, then apply the partial user overrides (same merge convention as before).

    ``spectators`` are present-but-not-grown elements (a loaded foreign substrate); they get
    distance/cutoff rows so placement and the graph work, without needing a curated CN/oxidation.
    ``terminal_cn`` are non-network saturation-cap elements (e.g. F, Na used as fragments) that
    are forced to a terminal CN of 1 -- so a placed cap is never flagged under-coordinated and
    a curated network CN (e.g. Na's 4-6) doesn't apply to it. An explicit coordination-block
    max_cn/min_cn for the same element still wins."""
    d = _as_dict(d, where) if d is not None else None
    if d is not None:
        _check_keys(d, {"max_cn", "min_cn", "cut_offs", "cn_distr", "oxidation", "distance"}, where)
    distance_knobs = _parse_distance_knobs(d["distance"], f"{where}.distance") if d and "distance" in d else None
    terminal = {str(e): 1 for e in terminal_cn}
    max_cn_over = {**terminal,
                   **({str(k): int(v) for k, v in _as_dict(d["max_cn"], f"{where}.max_cn").items()}
                      if d and "max_cn" in d else {})}
    min_cn_over = {**terminal,
                   **({str(k): int(v) for k, v in _as_dict(d["min_cn"], f"{where}.min_cn").items()}
                      if d and "min_cn" in d else {})}

    tables = build_element_tables(elements, spectator_elements=spectators,
                                  distance_knobs=distance_knobs,
                                  oxidation=oxidation, max_cn=max_cn_over, min_cn=min_cn_over)
    cfg = CoordinationConfig(
        max_cn=tables["max_cn"], min_cn=tables["min_cn"], cut_offs=tables["cut_offs"],
        sample_dist=tables["sample_dist"], d_min_max=tables["d_min_max"],
        oxidation=tables["oxidation"],
        # carry the effective bond-window scale so the saturation stage places caps at the same
        # scale as the grown network (not always the 1.0x default).
        bond_factor=float((distance_knobs or {}).get("bond_factor",
                                                     DEFAULT_DISTANCE_KNOBS["bond_factor"])),
    )
    if d and "cn_distr" in d:
        cfg.cn_distr.update(_parse_cn_distr(d["cn_distr"], f"{where}.cn_distr"))
    if d and "cut_offs" in d:
        overrides = _parse_cut_offs(d["cut_offs"], f"{where}.cut_offs")
        _validate_cutoff_overrides(overrides, cfg, f"{where}.cut_offs")
        cfg.cut_offs.update(overrides)
    return cfg


# --- calculator/composition cross-check ---------------------------------------------

_BKS_SUPPORTED = {"Si", "Al", "O"}


def _check_bks_supported(calculators: CalculatorsSpec, comp_elements: set,
                         saturation_enabled: bool, where: str) -> None:
    """Fail fast at config load (so --dry-run catches it) when a lammps/BKS calculator can't
    handle the run. The LAMMPS interface guards this at build time too, but catching it here
    gives an immediate, clear error. Two cases:

    - growth with lammps + a composition outside Si/Al/O; and
    - saturation enabled with a lammps saturation calculator: saturation adds H, which BKS has
      no parameters for, so it can never relax a saturated cell (rejected for any composition).
    """
    unsupported = comp_elements - _BKS_SUPPORTED
    if calculators.growth.type == "lammps" and unsupported:
        raise ValueError(
            f"{where}: the lammps/BKS backend supports only {sorted(_BKS_SUPPORTED)} "
            f"(silica-aluminas), but the composition contains {sorted(unsupported)}. "
            f"Use a 'mace' or 'uma' calculator for this composition.")
    if saturation_enabled:
        effective_sat = calculators.saturation or calculators.growth
        if effective_sat.type == "lammps":
            raise ValueError(
                f"{where}: saturation adds H, which the lammps/BKS backend cannot parameterize, "
                f"so BKS cannot relax the saturated structure. Use a 'mace' or 'uma' saturation "
                f"calculator (or disable saturation).")


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
    debug: DebugSpec = field(default_factory=DebugSpec)
    rules: list = field(default_factory=list)
    run: RunSpec = field(default_factory=RunSpec)
    dft: DFTSpec = field(default_factory=DFTSpec)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = _as_dict(d, "config")
        _check_keys(
            d,
            {"structure", "composition", "limits", "calculators", "coordination",
             "growth", "finalize", "saturation", "charge_correction", "statistics",
             "debug", "rules", "run", "dft"},
            "config",
            required=("structure", "composition", "limits", "calculators"),
        )
        rules = d.get("rules", []) or []
        if not isinstance(rules, list):
            raise ValueError("config.rules: expected a list of rule mappings")

        # Composition and coordination are interdependent: parse the composition to learn the
        # element set, resolve oxidation (curated + overrides), neutralise the integer counts,
        # then derive the per-pair distance / coordination tables for that element set.
        comp_parsed = _parse_composition_block(d["composition"], "composition")
        coord_block = _as_dict(d["coordination"], "coordination") if d.get("coordination") is not None else None
        # Saturation fragments may name non-network cap elements (e.g. F, Na): they must be in
        # the element tables (distance/cutoff rows + an oxidation) and are forced to a terminal
        # CN of 1, so a placed cap is never re-flagged under-coordinated.
        sat_block = _as_dict(d["saturation"], "saturation") if d.get("saturation") is not None else None
        # Fragment cap elements are wired in (and validated) only when saturation actually runs,
        # consistent with the saturation BKS guard below -- a disabled saturation block is inert.
        sat_enabled = bool(sat_block.get("enabled", False)) if sat_block else False
        cap_elems = (_saturation_fragment_elements(sat_block) - (comp_parsed["elements"] | {"O", "H"})
                     if sat_enabled else set())
        elements = sorted(comp_parsed["elements"] | {"O", "H"}
                          | _coord_override_elements(coord_block) | cap_elems)
        oxidation = _resolve_oxidation(elements, coord_block, "coordination")
        composition = _finalize_composition(comp_parsed, oxidation, "composition")
        # A loaded substrate's elements are present-but-not-grown spectators: they need
        # distance/cutoff rows (else placement KeyErrors on them) but no curated CN/oxidation.
        spectators = _loaded_elements(_as_dict(d["structure"], "structure")) - set(elements)
        coordination = _build_coordination(coord_block, "coordination",
                                           elements=elements, oxidation=oxidation,
                                           spectators=sorted(spectators),
                                           terminal_cn=sorted(cap_elems))

        calculators = CalculatorsSpec.from_dict(d["calculators"], "calculators")
        saturation = SaturationSpec.from_dict(d.get("saturation", {}), "saturation")
        # cap elements (O, H, fragments) must not carry per-atom oxidation overrides when
        # saturation runs; the opposite-sign growth check applies regardless.
        cap_block = ({"O", "H"} | cap_elems) if saturation.enabled else set()
        _validate_cn_distr_oxidation(coordination.cn_distr, oxidation, cap_block, "coordination.cn_distr")
        if saturation.enabled:
            _validate_fragments(saturation, oxidation, "saturation")
        _check_bks_supported(calculators, set(composition.target_counts),
                             saturation.enabled, "calculators")

        return cls(
            structure=StructureSpec.from_dict(d["structure"], "structure"),
            composition=composition,
            limits=LimitsSpec.from_dict(d["limits"], "limits"),
            calculators=calculators,
            coordination=coordination,
            growth=GrowthSpec.from_dict(d.get("growth", {}), "growth"),
            finalize=FinalizeSpec.from_dict(d.get("finalize", {}), "finalize"),
            saturation=saturation,
            charge_correction=ChargeCorrectionSpec.from_dict(d.get("charge_correction", {}), "charge_correction"),
            statistics=StatisticsSpec.from_dict(d.get("statistics", {}), "statistics"),
            debug=DebugSpec.from_dict(d.get("debug", {}), "debug"),
            rules=[RuleSpec.from_dict(r, f"rules[{i}]") for i, r in enumerate(rules)],
            run=RunSpec.from_dict(d.get("run", {}), "run"),
            dft=DFTSpec.from_dict(d.get("dft", {}), "dft"),
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
