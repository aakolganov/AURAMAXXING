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
    ("element_anion_distance.png", plots.plot_element_O, False),
    ("tau4.png", plots.plot_tau4, False),
    ("cap_distance.png", plots.plot_cap_distance, True),
    ("caps_per_cation.png", plots.plot_caps_per_cation, True),
]


_METRICS_FILE = "metrics.json"


def _write_figures(metrics: dict, out_dir: Path, is_saturation: bool):
    for fname, fn, sat_only in _FIGURES:
        if sat_only and not is_saturation:
            continue
        fn(metrics, out_dir / fname)


def write_report(struct, out_dir, is_saturation: bool = False, label: str | None = None,
                 metrics: dict | None = None) -> dict:
    """Analyse one structure: write its figures + stats.json into out_dir. Returns the
    metrics dict so a caller can pool several structures. Pass a precomputed ``metrics``
    to reuse an earlier ``analyze_structure`` instead of recomputing it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if metrics is None:
        metrics = analyze_structure(struct, is_saturation=is_saturation)
    _write_figures(metrics, out_dir, is_saturation)
    with open(out_dir / "stats.json", "w") as fh:
        json.dump({"label": label, "is_saturation": is_saturation,
                   "summary": summarize(metrics), "metrics": metrics}, fh, indent=2)
    return metrics


def write_metrics(out_dir, metrics: dict, is_saturation: bool = False,
                  label: str | None = None) -> Path:
    """Persist one structure's machine-readable metrics (``metrics.json``) without any
    plots. This small, always-writable artifact is what lets a pooled report be built
    from structures produced by separate processes or SLURM array tasks (see
    ``pool_reports_from_dir``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _METRICS_FILE
    with open(path, "w") as fh:
        json.dump({"label": label, "is_saturation": is_saturation, "metrics": metrics}, fh, indent=2)
    return path


def pool_reports_from_dir(root, is_saturation: bool = False) -> dict | None:
    """Reduce step for an ensemble: gather every per-structure ``metrics.json`` under
    ``root`` and write the pooled figure set + ``stats_pooled.json`` into ``root``.

    Reading results off disk (rather than from an in-memory list) is what makes pooling
    correct under parallelism: structures generated serially, by an in-node pool, or by
    independent SLURM array tasks all land as ``metrics.json`` files in the same output
    tree, and this gathers them uniformly. Returns the merged metrics, or ``None`` when
    fewer than two were found (nothing to pool)."""
    root = Path(root)
    metrics_list = []
    for path in sorted(root.glob(f"**/{_METRICS_FILE}")):
        with open(path) as fh:
            metrics_list.append(json.load(fh)["metrics"])
    if len(metrics_list) < 2:
        return None
    return write_pooled_report(metrics_list, root, is_saturation=is_saturation)


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
