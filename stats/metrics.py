"""Structural statistics for a generated/saturated surface.

Pure computation — no matplotlib — so it is fast and trivially testable. Everything is
derived from an ``AmorphousStruc`` (its ASE atoms, coordination graph and bonding
cutoffs). Fixed atoms (a frozen substrate carried in via a ``FixAtoms`` constraint) are
excluded as the *subject* of every distribution; they may still appear as partners.
"""
from __future__ import annotations

import numpy as np

from default_constants import OXIDATION_POS, OXIDATION_NEG

THETA_T = 109.5   # ideal tetrahedral angle, for the tau4 indices
_ANION = next(iter(OXIDATION_NEG))            # "O"
# cations that can carry an -OH group (X-O-H); H itself is excluded.
_OH_CATIONS = [e for e in OXIDATION_POS if e != "H"]


def fixed_indices(struct) -> set:
    """Indices held fixed by any ASE constraint (e.g. a FixAtoms substrate)."""
    fixed: set = set()
    for constraint in struct.atoms.constraints:
        if hasattr(constraint, "get_indices"):
            fixed.update(int(i) for i in constraint.get_indices())
    return fixed


def _tau4_indices(angles_deg: list[float]) -> tuple[float, float]:
    """tau4 and tau4' from the six L-centre-L angles of a 4-coordinate centre.

    beta >= alpha are the two largest angles. tau4 = 1 / tau4' = 1 for an ideal
    tetrahedron, 0 for square planar (Yang 2007; Okuniewski 2015).
    """
    beta, alpha = sorted(angles_deg, reverse=True)[:2]
    tau4 = (360.0 - (alpha + beta)) / (360.0 - 2.0 * THETA_T)
    tau4_prime = (beta - alpha) / (360.0 - THETA_T) + (180.0 - beta) / (180.0 - THETA_T)
    return tau4, tau4_prime


def analyze_structure(struct, is_saturation: bool = False) -> dict:
    """Compute the per-element distributions used by the plotting/report layer.

    Returns a dict of raw distributions (plain Python lists, JSON-friendly). When
    ``is_saturation`` is False the ``saturation`` entry is ``None``.
    """
    atoms = struct.atoms
    n = len(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    graph = struct.get_graph()
    fixed = fixed_indices(struct)
    mobile = np.array([i not in fixed for i in range(n)], dtype=bool)

    elements = sorted(set(symbols.tolist()))
    out: dict = {
        "n_atoms": int(n),
        "n_fixed": int(len(fixed)),
        "composition": {el: int(np.count_nonzero(symbols == el)) for el in elements},
        "coordination": {},
        "homo_distance": {},
        "element_O_distance": {},
        "tau4": {},
        "saturation": None,
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

    # 3) nearest element-O distance (mobile non-O subjects)
    o_idx = np.where(symbols == _ANION)[0]
    if len(o_idx) > 0:
        for el in elements:
            if el == _ANION:
                continue
            subj = np.where((symbols == el) & mobile)[0]
            if len(subj) == 0:
                continue
            out["element_O_distance"][el] = [float(dmat[i, o_idx].min()) for i in subj]

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

    # 5) saturation-only: O-H distances and OH groups per cation
    if is_saturation:
        oh_distances = []
        for u, v in graph.edges():
            pair = {symbols[u], symbols[v]}
            if pair == {_ANION, "H"}:
                oh_distances.append(float(dmat[u, v] if np.isfinite(dmat[u, v])
                                          else atoms.get_distance(u, v, mic=True)))
        oh_per_element: dict = {}
        for el in _OH_CATIONS:
            subj = np.where((symbols == el) & mobile)[0]
            if len(subj) == 0:
                continue
            counts = []
            for i in subj:
                n_oh = 0
                for o in graph.neighbors(i):
                    if symbols[o] == _ANION and any(symbols[h] == "H" for h in graph.neighbors(o)):
                        n_oh += 1
                counts.append(int(n_oh))
            oh_per_element[el] = counts
        out["saturation"] = {"oh_distances": oh_distances, "oh_per_element": oh_per_element}

    return out


def merge_metrics(metrics_list: list) -> dict:
    """Pool several per-structure metrics dicts into one (for the per-run plots)."""
    metrics_list = [m for m in metrics_list if m]
    merged: dict = {"n_atoms": 0, "n_fixed": 0, "n_structures": len(metrics_list),
                    "composition": {}, "coordination": {}, "homo_distance": {},
                    "element_O_distance": {}, "tau4": {}, "saturation": None}
    sat_d, sat_c = [], {}
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
        for el, v in m["element_O_distance"].items():
            merged["element_O_distance"].setdefault(el, []).extend(v)
        for el, d in m["tau4"].items():
            tgt = merged["tau4"].setdefault(el, {"tau4": [], "tau4_prime": []})
            tgt["tau4"].extend(d["tau4"])
            tgt["tau4_prime"].extend(d["tau4_prime"])
        if m.get("saturation"):
            sat_d.extend(m["saturation"]["oh_distances"])
            for el, v in m["saturation"]["oh_per_element"].items():
                sat_c.setdefault(el, []).extend(v)
    if sat_d or sat_c:
        merged["saturation"] = {"oh_distances": sat_d, "oh_per_element": sat_c}
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
        "element_O_distance": {el: _summary_stats(v) for el, v in metrics["element_O_distance"].items()},
        "tau4_mean": {el: float(np.mean(d["tau4"])) for el, d in metrics["tau4"].items()},
    }
    if metrics.get("saturation"):
        sat = metrics["saturation"]
        s["oh_distance"] = _summary_stats(sat["oh_distances"])
        s["oh_per_element_mean"] = {el: (float(np.mean(v)) if v else None)
                                    for el, v in sat["oh_per_element"].items()}
    return s
