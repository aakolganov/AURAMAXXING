"""Matplotlib figures for the structure statistics.

A metrics dict (from ``stats.metrics.analyze_structure`` or ``merge_metrics``) is turned
into figure files. Typography/sizing/output all come from ``stats.style`` (Agg backend,
TeX Gyre Heros font stack, DPI 300, tight bbox).
"""
from __future__ import annotations

from pathlib import Path

from . import style
style.apply_style()
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

_ELEMENT_COLORS = {"Si": "tab:blue", "Al": "tab:orange", "O": "tab:red", "H": "tab:green",
                   "F": "tab:purple", "Cl": "tab:olive", "Na": "tab:cyan"}


def _color(el):
    return _ELEMENT_COLORS.get(el, "tab:gray")


def _save(fig, path: Path):
    style.savefig_multi(fig, path)


def _empty(ax, msg="no data"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="gray")


def plot_coordination(metrics: dict, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    coord = {el: v for el, v in metrics["coordination"].items() if v}
    if not coord:
        _empty(ax)
    else:
        all_cn = [c for v in coord.values() for c in v]
        bins = np.arange(min(all_cn), max(all_cn) + 2) - 0.5
        elements = sorted(coord)
        ax.hist([coord[el] for el in elements], bins=bins,
                label=elements, color=[_color(el) for el in elements])
        ax.set_xticks(range(int(bins[0] + 0.5), int(bins[-1] + 0.5)))
        ax.legend(title="element")
    ax.set_xlabel("coordination number")
    ax.set_ylabel("count (mobile atoms)")
    ax.set_title("Coordination number distribution")
    _save(fig, path)


def plot_homo_distance(metrics: dict, path: Path):
    homo = metrics["homo_distance"]
    elements = sorted(homo)
    fig, axes = plt.subplots(len(elements) or 1, 1, figsize=(7, 2.4 * (len(elements) or 1)),
                             squeeze=False)
    if not elements:
        _empty(axes[0, 0], "no homo-element pairs")
    for ax, el in zip(axes[:, 0], elements, strict=False):
        d = homo[el]
        ax.hist(d["distances"], bins=30, color=_color(el), alpha=0.8)
        cutoff, n_bond = d["cutoff"], d["homo_bond_count"]
        if cutoff is not None:
            ax.axvline(cutoff, color="red", ls="--", lw=1,
                       label=f"bond cutoff {cutoff:.2f} Å")
            ax.legend(loc="upper right")
        if n_bond:
            ax.text(0.03, 0.92, f"⚠ {n_bond} homo-bond(s)", transform=ax.transAxes,
                    fontsize=style.ANNOTATION, color="red", va="top",
                    bbox=dict(boxstyle="round", fc="white", ec="red", alpha=0.9))
        ax.set_title(f"{el}–{el} nearest-neighbour distance")
        ax.set_ylabel("count")
    axes[-1, 0].set_xlabel("nearest same-element distance (Å)")
    _save(fig, path)


def plot_element_O(metrics: dict, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = {el: v for el, v in metrics["element_anion_distance"].items() if v}
    if not data:
        _empty(ax)
    else:
        for el in sorted(data):
            ax.hist(data[el], bins=30, alpha=0.55, label=f"{el}–anion", color=_color(el))
        ax.legend()
    ax.set_xlabel("nearest element–anion distance (Å)")
    ax.set_ylabel("count")
    ax.set_title("Element–anion closest-distance distribution")
    _save(fig, path)


def plot_tau4(metrics: dict, path: Path):
    tau = metrics["tau4"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    data = {el: v for el, v in tau.items() if v["tau4"]}
    if not data:
        _empty(ax1, "no 4-coordinate atoms")
        _empty(ax2, "")
    else:
        bins = np.linspace(0, 1, 26)
        for el in sorted(data):
            ax1.hist(data[el]["tau4"], bins=bins, alpha=0.55, label=el, color=_color(el))
            ax2.hist(data[el]["tau4_prime"], bins=bins, alpha=0.55, label=el, color=_color(el))
        ax1.legend(title="element")
    ax1.set_title(r"$\tau_4$ (1 = tetrahedral, 0 = square planar)")
    ax2.set_title(r"$\tau_4'$")
    for ax in (ax1, ax2):
        ax.set_xlabel("index")
    ax1.set_ylabel("count (4-coordinate atoms)")
    _save(fig, path)


def plot_cap_distance(metrics: dict, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = {label: v for label, v in
            (metrics.get("saturation") or {}).get("cap_distances", {}).items() if v}
    if not data:
        _empty(ax)
    else:
        for label in sorted(data):
            cap_el = label.split("-")[-1]   # colour by the cap element (O-H -> H, Si-F -> F)
            ax.hist(data[label], bins=30, alpha=0.6, label=label, color=_color(cap_el))
        ax.legend(title="cap bond")
    ax.set_xlabel("cap bond distance (Å)")
    ax.set_ylabel("count")
    ax.set_title("Saturation cap bond-length distribution")
    _save(fig, path)


def plot_caps_per_cation(metrics: dict, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    per = (metrics.get("saturation") or {}).get("caps_per_cation", {})
    per = {el: v for el, v in per.items() if v}
    if not per:
        _empty(ax)
    else:
        max_caps = max(max(v) for v in per.values())
        bins = np.arange(0, max_caps + 2) - 0.5
        elements = sorted(per)
        ax.hist([per[el] for el in elements], bins=bins,
                label=elements, color=[_color(el) for el in elements])
        ax.set_xticks(range(0, int(max_caps) + 1))
        ax.legend(title="element")
    ax.set_xlabel("number of capping groups on the atom")
    ax.set_ylabel("count")
    ax.set_title("Capping groups per cation")
    _save(fig, path)


def plot_cap_areal_density(metrics: dict, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    cad = metrics.get("cap_areal_density") or {}
    per = cad.get("per_type") or {}
    if "error" in cad or not per:
        _empty(ax)
    else:
        labels = sorted(per)
        heights = [per[label] for label in labels] + [cad.get("total", sum(per.values()))]
        # colour each cap-type bar by its cap element (O-H -> H, Si-F -> F); total in grey.
        colors = [_color(label.split("-")[-1]) for label in labels] + ["tab:gray"]
        xs = labels + ["total"]
        ax.bar(range(len(xs)), heights, color=colors)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=30, ha="right")
        area = cad.get("area_nm2")
        if area:
            ax.text(0.97, 0.95, f"VdW area {area:.1f} nm²", transform=ax.transAxes,
                    ha="right", va="top", fontsize=style.ANNOTATION, color="gray")
    ax.set_ylabel("areal concentration (groups / nm²)")
    ax.set_title("Surface cap-group areal concentration")
    _save(fig, path)
