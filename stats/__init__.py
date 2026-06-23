"""Structure statistics: per-element distributions + diagnostic plots.

    from stats import analyze_structure, write_report
    write_report(struct, "out_dir", is_saturation=False)

or from the command line on a saved structure:

    python -m stats structure.vasp --out stats_dir [--saturation]
"""
from .metrics import analyze_structure, merge_metrics, summarize
from .report import write_report, write_pooled_report

__all__ = ["analyze_structure", "merge_metrics", "summarize",
           "write_report", "write_pooled_report"]
