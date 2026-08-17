"""Summarize a sweep_mlip.jsonl into the (mode, anneal, scale) grid, per-cell means.

Usage: python summarize.py [sweep_mlip.jsonl]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "sweep_mlip.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["calc"], r["mode"], r["anneal"], r["scale"])].append(r)

    print(f"{'calc':<6}{'mode':<12}{'anneal':<9}{'scale':<6}{'n':>2}{'compl':>6}"
          f"{'anneals':>8}{'t(s)':>7} | {'grown oo/ss/r2':>14} | "
          f"{'final oo/ss/r2':>14}{'defTot':>7}")
    for key in sorted(cells):
        rs = cells[key]
        def m(*p, rs=rs):
            return np.mean([r[p[0]] if len(p) == 1 else r[p[0]][p[1]] for r in rs])
        g = f"{m('grown','oo'):.1f}/{m('grown','sisi'):.1f}/{m('grown','ring2'):.1f}"
        f = f"{m('final','oo'):.1f}/{m('final','sisi'):.1f}/{m('final','ring2'):.1f}"
        tot = m("final", "oo") + m("final", "sisi") + m("final", "ring2")
        print(f"{key[0]:<6}{key[1]:<12}{key[2]:<9}{key[3]:<6}{len(rs):>2}"
              f"{int(sum(r['completed'] for r in rs)):>6}{m('anneals'):>8.1f}"
              f"{m('t_grow') + m('t_final'):>7.0f} | {g:>14} | {f:>14}{tot:>7.1f}")
    print("\ndefTot = final (oo<1.8) + (sisi<2.4) + ring2 per slab; "
          "grown = end of growth, pre-finalize.")


if __name__ == "__main__":
    main()
