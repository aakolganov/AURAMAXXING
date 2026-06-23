"""Write the statistics figures + a stats.json for a structure (or a pooled set)."""
from __future__ import annotations

import json
from pathlib import Path

from .metrics import analyze_structure, merge_metrics, summarize
from . import plots

# (filename, plotting function, saturation-only?)
_FIGURES = [
    ("coordination.png", plots.plot_coordination, False),
    ("homo_element_distance.png", plots.plot_homo_distance, False),
    ("element_O_distance.png", plots.plot_element_O, False),
    ("tau4.png", plots.plot_tau4, False),
    ("oh_distance.png", plots.plot_oh_distance, True),
    ("oh_per_element.png", plots.plot_oh_per_element, True),
]


def _write_figures(metrics: dict, out_dir: Path, is_saturation: bool):
    for fname, fn, sat_only in _FIGURES:
        if sat_only and not is_saturation:
            continue
        fn(metrics, out_dir / fname)


def write_report(struct, out_dir, is_saturation: bool = False, label: str | None = None) -> dict:
    """Analyse one structure: write its figures + stats.json into out_dir. Returns the
    metrics dict so a caller can pool several structures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = analyze_structure(struct, is_saturation=is_saturation)
    _write_figures(metrics, out_dir, is_saturation)
    with open(out_dir / "stats.json", "w") as fh:
        json.dump({"label": label, "is_saturation": is_saturation,
                   "summary": summarize(metrics), "metrics": metrics}, fh, indent=2)
    return metrics


def write_pooled_report(metrics_list: list, out_dir, is_saturation: bool = False) -> dict:
    """Pool several per-structure metrics dicts and write a combined figure set + json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = merge_metrics(metrics_list)
    _write_figures(merged, out_dir, is_saturation)
    summary = summarize(merged)
    summary["n_structures"] = merged.get("n_structures", len(metrics_list))
    with open(out_dir / "stats_pooled.json", "w") as fh:
        json.dump({"is_saturation": is_saturation, "summary": summary, "metrics": merged}, fh, indent=2)
    return merged
