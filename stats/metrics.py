"""Structural statistics for a generated/saturated surface.

Pure computation — no matplotlib — so it is fast and trivially testable. Everything is
derived from an ``AmorphousStruc`` (its ASE atoms, coordination graph and bonding
cutoffs). Fixed atoms (a frozen substrate carried in via a ``FixAtoms`` constraint) are
excluded as the *subject* of every distribution; they may still appear as partners.
"""
from __future__ import annotations

import numpy as np

THETA_T = 109.5   # ideal tetrahedral angle, for the tau4 indices


def _tau4_indices(angles_deg: list[float]) -> tuple[float, float]:
    """tau4 and tau4' from the six L-centre-L angles of a 4-coordinate centre.

    beta >= alpha are the two largest angles. tau4 = 1 / tau4' = 1 for an ideal
    tetrahedron, 0 for square planar (Yang 2007; Okuniewski 2015).
    """
    beta, alpha = sorted(angles_deg, reverse=True)[:2]
    tau4 = (360.0 - (alpha + beta)) / (360.0 - 2.0 * THETA_T)
    tau4_prime = (beta - alpha) / (360.0 - THETA_T) + (180.0 - beta) / (180.0 - THETA_T)
    return tau4, tau4_prime


def analyze_structure(struct, is_saturation: bool = False,
                      surface_opts: dict | None = None) -> dict:
    """Compute the per-element distributions used by the plotting/report layer.

    Returns a dict of raw distributions (plain Python lists, JSON-friendly). When
    ``is_saturation`` is False the ``saturation`` entry is ``None``. ``surface_opts`` (from
    ``cfg.statistics.surface``: ``{enabled, probe, n_points, overrides}``) parameterises the
    saturation-only cap areal-density metric; when omitted it uses defaults (enabled, probe 0).
    """
    atoms = struct.atoms
    n = len(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    graph = struct.get_graph()
    fixed = struct.fixed_indices()
    mobile = np.array([i not in fixed for i in range(n)], dtype=bool)

    elements = sorted(set(symbols.tolist()))
    # Anion/cation identity is data-driven from the struct's per-element oxidation, so the
    # metrics work for any oxide (not just Si/Al/O/H): anions have oxidation < 0, cations > 0.
    ox = getattr(struct, "oxidation", {}) or {}
    anions = {e for e, v in ox.items() if v < 0}
    cations = {e for e, v in ox.items() if v > 0}
    out: dict = {
        "n_atoms": int(n),
        "n_fixed": int(len(fixed)),
        "composition": {el: int(np.count_nonzero(symbols == el)) for el in elements},
        "coordination": {},
        "homo_distance": {},
        "element_anion_distance": {},
        "tau4": {},
        "saturation": None,
        "cap_areal_density": None,
    }
    if n == 0:
        return out

    # Full minimum-image distance matrix (N <= a few hundred -> cheap, PBC-correct).
    dmat = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dmat, np.inf)   # so "nearest other atom" never picks self

    # 1) coordination number per element (mobile subjects)
    degrees = dict(graph.degree())
    for el in elements:
        idx = np.where((symbols == el) & mobile)[0]
        out["coordination"][el] = [int(degrees[i]) for i in idx]

    # 2) nearest homo-element distance per element (+ count below the bonding cutoff)
    for el in elements:
        same = np.where(symbols == el)[0]
        subj = np.where((symbols == el) & mobile)[0]
        if len(same) < 2 or len(subj) == 0:
            continue
        dists = [float(dmat[i, same].min()) for i in subj]
        cutoff = struct.cut_offs.get((el, el))
        homo_bonds = int(np.sum(np.array(dists) < cutoff)) if cutoff is not None else 0
        out["homo_distance"][el] = {
            "distances": dists,
            "cutoff": float(cutoff) if cutoff is not None else None,
            "homo_bond_count": homo_bonds,
        }

    # 3) nearest element-anion distance (mobile non-anion subjects; "element-O" for oxides)
    anion_idx = np.where(np.isin(symbols, list(anions)))[0] if anions else np.array([], dtype=int)
    if len(anion_idx) > 0:
        for el in elements:
            if el in anions:
                continue
            subj = np.where((symbols == el) & mobile)[0]
            if len(subj) == 0:
                continue
            out["element_anion_distance"][el] = [float(dmat[i, anion_idx].min()) for i in subj]

    # 4) tau4 / tau4' for 4-coordinate mobile centres
    for el in elements:
        subj = [i for i in np.where((symbols == el) & mobile)[0] if degrees[i] == 4]
        t4, t4p = [], []
        for i in subj:
            nbrs = list(graph.neighbors(i))
            angles = [atoms.get_angle(nbrs[a], i, nbrs[b], mic=True)
                      for a in range(4) for b in range(a + 1, 4)]
            a4, a4p = _tau4_indices(angles)
            t4.append(float(a4))
            t4p.append(float(a4p))
        if t4:
            out["tau4"][el] = {"tau4": t4, "tau4_prime": t4p}

    # 5) saturation-only: cap-bond distances + capping groups per cation. Generalised from the
    # old O-H / -OH metric so any 1-valent cap (H, F, Na, ...) is reported, not just hydroxyl.
    if is_saturation:
        # ``degrees`` was already materialised above (coordination pass); reuse it.
        # A cap is a mobile, monovalent (max_cn 1) terminal atom -- H, F, Na, ...; record its
        # bond to its single anchor, labelled "{anchor}-{cap}".
        cap_distances: dict = {}
        max_cn = getattr(struct, "max_cn", {}) or {}
        for i in range(n):
            if mobile[i] and degrees.get(i, 0) == 1 and max_cn.get(symbols[i], 99) == 1:
                a = next(iter(graph.neighbors(i)))
                d = dmat[i, a] if np.isfinite(dmat[i, a]) else atoms.get_distance(i, a, mic=True)
                cap_distances.setdefault(f"{symbols[a]}-{symbols[i]}", []).append(float(d))
        # Caps per cation: a neighbouring anion that is "finished" -- bonded to this cation and
        # otherwise only to monovalent caps (a hydroxyl's H, or a relabelled Na/other 1-valent
        # cap) or to nothing else (a terminal halide). The completing partner is any monovalent
        # (max_cn 1) atom, not hardcoded H, so a non-H positive relabel (e.g. O-Na) still counts.
        caps_per_cation: dict = {}
        for el in sorted(cations):
            subj = np.where((symbols == el) & mobile)[0]
            if len(subj) == 0:
                continue
            counts = []
            for i in subj:
                n_cap = sum(1 for nb in graph.neighbors(i)
                            if symbols[nb] in anions
                            and all(max_cn.get(symbols[o], 99) == 1
                                    for o in graph.neighbors(nb) if o != i))
                counts.append(int(n_cap))
            caps_per_cation[el] = counts
        out["saturation"] = {"cap_distances": cap_distances, "caps_per_cation": caps_per_cation}

        # Surface cap-group areal concentration (groups/nm²), normalised by the true VdW surface
        # area (rough tops make count/(Lx·Ly) meaningless). Gated + parameterised by surface_opts;
        # non-fatal so a degenerate slab / missing dependency never aborts the rest of the metrics.
        opts = surface_opts or {}
        if opts.get("enabled", True):
            try:
                from .surface import cap_areal_density
                cap_counts = {label: len(v) for label, v in cap_distances.items()}
                out["cap_areal_density"] = cap_areal_density(
                    struct, cap_counts, sum(cap_counts.values()),
                    probe=float(opts.get("probe", 0.0)),
                    n_points=int(opts.get("n_points", 200)),
                    overrides=opts.get("overrides") or None)
            except Exception as exc:   # keep the rest of the metrics dict intact
                out["cap_areal_density"] = {"error": f"{type(exc).__name__}: {exc}"}

    return out


def merge_metrics(metrics_list: list) -> dict:
    """Pool several per-structure metrics dicts into one (for the per-run plots)."""
    metrics_list = [m for m in metrics_list if m]
    merged: dict = {"n_atoms": 0, "n_fixed": 0, "n_structures": len(metrics_list),
                    "composition": {}, "coordination": {}, "homo_distance": {},
                    "element_anion_distance": {}, "tau4": {}, "saturation": None,
                    "cap_areal_density": None}
    sat_d, sat_c = {}, {}
    # Pooled areal density is the ratio of TOTALS (Σcaps / Σarea), not the mean of per-structure
    # densities: averaging ratios weights a small slab like a large one and is biased. Accumulate
    # counts and area, divide once at the end.
    dens_counts: dict = {}
    dens_area = 0.0
    dens_ncaps = 0
    dens_nstruct = 0
    for m in metrics_list:
        merged["n_atoms"] += m["n_atoms"]
        merged["n_fixed"] += m["n_fixed"]
        for el, c in m["composition"].items():
            merged["composition"][el] = merged["composition"].get(el, 0) + c
        for el, v in m["coordination"].items():
            merged["coordination"].setdefault(el, []).extend(v)
        for el, d in m["homo_distance"].items():
            tgt = merged["homo_distance"].setdefault(el, {"distances": [], "cutoff": d["cutoff"],
                                                          "homo_bond_count": 0})
            tgt["distances"].extend(d["distances"])
            tgt["homo_bond_count"] += d["homo_bond_count"]
            tgt["cutoff"] = tgt["cutoff"] if tgt["cutoff"] is not None else d["cutoff"]
        for el, v in m["element_anion_distance"].items():
            merged["element_anion_distance"].setdefault(el, []).extend(v)
        for el, d in m["tau4"].items():
            tgt = merged["tau4"].setdefault(el, {"tau4": [], "tau4_prime": []})
            tgt["tau4"].extend(d["tau4"])
            tgt["tau4_prime"].extend(d["tau4_prime"])
        if m.get("saturation"):
            # .get() so a pre-rename metrics.json (oh_distances/oh_per_element) on disk pools as
            # empty rather than KeyError-ing the whole pooled report.
            for label, v in m["saturation"].get("cap_distances", {}).items():
                sat_d.setdefault(label, []).extend(v)
            for el, v in m["saturation"].get("caps_per_cation", {}).items():
                sat_c.setdefault(el, []).extend(v)
        cad = m.get("cap_areal_density")
        if cad and "error" not in cad:
            dens_nstruct += 1
            dens_area += cad.get("area_nm2", 0.0)
            dens_ncaps += cad.get("n_caps", 0)
            for label, c in cad.get("counts", {}).items():
                dens_counts[label] = dens_counts.get(label, 0) + c
    if sat_d or sat_c:
        merged["saturation"] = {"cap_distances": sat_d, "caps_per_cation": sat_c}
    if dens_nstruct and dens_area > 0:
        merged["cap_areal_density"] = {
            "per_type": {label: c / dens_area for label, c in dens_counts.items()},
            "total": dens_ncaps / dens_area,
            "area_nm2": dens_area, "counts": dens_counts, "n_caps": dens_ncaps,
            "n_structures": dens_nstruct}
    return merged


def _summary_stats(values: list) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max())}


def summarize(metrics: dict) -> dict:
    """Scalar summary of a metrics dict, for the stats.json header."""
    s: dict = {
        "n_atoms": metrics["n_atoms"],
        "n_fixed": metrics["n_fixed"],
        "composition": metrics["composition"],
        "coordination_mean": {el: (float(np.mean(v)) if v else None)
                              for el, v in metrics["coordination"].items()},
        "homo_bonds": {el: d["homo_bond_count"] for el, d in metrics["homo_distance"].items()},
        "element_anion_distance": {el: _summary_stats(v) for el, v in metrics["element_anion_distance"].items()},
        "tau4_mean": {el: float(np.mean(d["tau4"])) for el, d in metrics["tau4"].items()},
    }
    if metrics.get("saturation"):
        sat = metrics["saturation"]
        s["cap_distance"] = {label: _summary_stats(v)
                             for label, v in sat.get("cap_distances", {}).items()}
        s["caps_per_cation_mean"] = {el: (float(np.mean(v)) if v else None)
                                     for el, v in sat.get("caps_per_cation", {}).items()}
    cad = metrics.get("cap_areal_density")
    if cad and "error" not in cad:
        s["cap_areal_density"] = {"total": cad["total"], "per_type": cad["per_type"],
                                  "area_nm2": cad["area_nm2"], "n_caps": cad["n_caps"]}
    return s
