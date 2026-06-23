"""Shared plot styling for the statistics figures.

Typography: TeX Gyre Heros (a Helvetica metric clone) with an Arial -> Helvetica ->
DejaVu Sans fallback (matplotlib silently walks the list to the first installed font).
Sizes are fixed in points so every figure matches.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless-safe; set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402

FONT_STACK = ["TeX Gyre Heros", "Arial", "Helvetica", "DejaVu Sans"]

# point sizes
AXIS_LABEL = 11
TITLE = 11
TICK = 10
LEGEND = 10
STATBOX = 10
ANNOTATION = 10        # in-figure value annotations (9-10)

DPI = 300


def apply_style() -> None:
    """Install the house style into matplotlib's rcParams (idempotent)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "axes.labelsize": AXIS_LABEL,
        "axes.titlesize": TITLE,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": LEGEND,
        "figure.titlesize": TITLE,
        "axes.facecolor": "white",     # white plot area even when the figure is transparent
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
    })


def wrap_label(text, width: int = 12) -> str:
    """Wrap a long axis-tick label onto two lines (no text is dropped; any overflow
    beyond the first line is kept on the second)."""
    wrapped = textwrap.wrap(str(text), width=width, break_long_words=False)
    if not wrapped:
        return str(text)
    if len(wrapped) <= 2:
        return "\n".join(wrapped)
    return wrapped[0] + "\n" + " ".join(wrapped[1:])


def wrap_xticklabels(ax, width: int = 12) -> None:
    """Wrap any existing string x-tick labels in place."""
    labels = [t.get_text() for t in ax.get_xticklabels()]
    if any(labels):
        ax.set_xticklabels([wrap_label(t, width) for t in labels])


def savefig_multi(fig, path, formats=("png",), transparent: bool = True) -> None:
    """Save a figure to one or more formats: DPI 300, tight bbox, transparent canvas
    (the axes keep their white background). Then close the figure."""
    stem = Path(path).with_suffix("")
    for fmt in formats:
        fig.savefig(f"{stem}.{fmt}", dpi=DPI, bbox_inches="tight", transparent=transparent)
    plt.close(fig)
