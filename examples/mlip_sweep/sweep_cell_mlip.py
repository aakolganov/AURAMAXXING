"""One cell of the anneal-scheme x class-floor-scale sweep, MLIP edition.

Grows a production-like silica slab (Si150O300, 22x22x40, flat bottom z=12,
fourier top z_av=24 alpha=0.3) with the given growth mode, anneal scheme and
same-class exclusion scaling, then finalizes (LBFGS, fmax 0.1), and appends one
JSON line (anneal count, timings, defect census at grown + final) to --results.

The class-floor scale is applied by post-processing the derived tables (entries
with d_min_max[a][b][0] == [1] are the geometric same-class floors of the
cutoff_fixing branch), so no runner/config support is needed.

Run on the A10G box (see the header of run_grid.sh for the venv):
  PYTHONPATH=$HOME/AURAMAXXING ~/macevenv/bin/python sweep_cell_mlip.py \
      --anneal default --scale 0.9 --mode deposition --seed 1 --device cuda

BKS context (local sweep, deposition mode): scale <= 0.9 never jammed and grew a
defect-free network; every BKS melt-quench froze in Si-Si/2-rings. BKS has an
inherent O-O repulsion bias, hence this MLIP re-run of the grid.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from base.amorphous_structure import AmorphousStruc_factory
from base.config import CoordinationConfig
from base.element_data import build_element_tables
from base.limits import make_limit_flat, make_limits_fourier, fix_limits
from growth.new_growth import grow_structure, finalize_structure
from helpers.files_io import write_structure_to_file

CELL = [22.0, 22.0, 40.0]
ANNEALS = {
    "default": {"T_ini": 2000, "T_fin": 300, "steps": 250, "interval": 10},
    "cool":    {"T_ini": 1200, "T_fin": 300, "steps": 250, "interval": 10},
    "slow":    {"T_ini": 2000, "T_fin": 300, "steps": 1000, "interval": 10},
}


def make_calc(kind: str, model: str, device: str, dump_path: str):
    # lazy imports: the MACE venv has no fairchem and vice versa
    if kind == "mace":
        from interfaces.MACE_interface import MACEInterface
        return MACEInterface(mace_model_path=model, device=device, dump_path=dump_path)
    if kind == "uma":
        from interfaces.UMA_interface import UMAInterface
        return UMAInterface(uma_model_path=model, task="omat", device=device,
                            dump_path=dump_path)
    raise ValueError(f"unknown calc kind {kind!r}")


def build_struct(seed: int, scale: float):
    t = build_element_tables(["Si", "O", "H"])
    scaled = {a: {b: ([lo * scale, hi * scale] if lo == hi else [lo, hi])
                  for b, (lo, hi) in row.items()}
              for a, row in t["d_min_max"].items()}
    cfg = CoordinationConfig(max_cn=t["max_cn"], min_cn=t["min_cn"],
                             cut_offs=t["cut_offs"], oxidation=t["oxidation"],
                             sample_dist=t["sample_dist"], d_min_max=scaled)
    s = AmorphousStruc_factory(cell=CELL, pbc=True, seed=seed, config=cfg)
    make_limit_flat(s, z_val=12.0, is_for="bottom")
    make_limits_fourier(s, z_av=24.0, alpha=0.3, is_for="top")
    fix_limits(s.limits, hard_limit="bottom")
    return s


def census(atoms):
    cell = np.array(CELL)
    sym = np.array(atoms.get_chemical_symbols())
    pos = atoms.get_positions() % cell
    o = pos[sym == "O"]
    si = pos[sym == "Si"]
    to = cKDTree(o, boxsize=cell)
    ts = cKDTree(si, boxsize=cell)

    def dists(tree, pts, rmax):
        out = []
        for i, j in tree.query_pairs(r=rmax):
            d = pts[i] - pts[j]
            d -= cell * np.round(d / cell)
            out.append(float(np.linalg.norm(d)))
        return out

    oo = dists(to, o, 2.2)
    ss = dists(ts, si, 3.0)
    nbrs = {i: set(h) for i, h in enumerate(ts.query_ball_tree(to, r=2.17))}
    keys = sorted(nbrs)
    ring2 = sum(1 for i, a in enumerate(keys) for b in keys[i + 1:]
                if len(nbrs[a] & nbrs[b]) >= 2)
    return {"oo": sum(d < 1.8 for d in oo), "oo_m": sum(d >= 1.8 for d in oo),
            "sisi": sum(d < 2.4 for d in ss), "sisi_m": sum(d >= 2.4 for d in ss),
            "ring2": ring2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anneal", choices=sorted(ANNEALS), required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--mode", choices=["default", "deposition"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--calc", choices=["mace", "uma"], default="mace")
    ap.add_argument("--model", default="medium-omat-0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target", type=int, default=450,
                    help="target atoms (lower it for smoke tests)")
    ap.add_argument("--results", default="sweep_mlip.jsonl")
    ap.add_argument("--workdir", default="sweep_runs")
    args = ap.parse_args()

    tag = f"{args.calc}_{args.anneal}_s{args.scale}_{args.mode}_{args.seed}"
    wd = Path(args.workdir) / tag
    wd.mkdir(parents=True, exist_ok=True)

    s = build_struct(args.seed, args.scale)
    calc = make_calc(args.calc, args.model, args.device, str(wd / "dump"))
    n_anneal = {"n": 0}
    orig = calc.anneal
    def counting_anneal(*a, **k):
        n_anneal["n"] += 1
        return orig(*a, **k)
    calc.anneal = counting_anneal

    t0 = time.time()
    completed = grow_structure(
        s, target_number_atoms=args.target, target_ratios={"Si": 1, "O": 2},
        calculator=calc, mode=args.mode, anneal_params=ANNEALS[args.anneal],
        workdir=wd, output_dir=wd / "growth")
    t_grow = time.time() - t0
    grown = census(s.atoms)
    write_structure_to_file(s, wd / "grown", write_xyz=False)

    t0 = time.time()
    finalize_structure(s, calculator=calc, fmax=0.1, max_steps=500, workdir=wd)
    t_final = time.time() - t0
    final = census(s.atoms)
    write_structure_to_file(s, wd / "final", write_xyz=False)

    row = {"tag": tag, "calc": args.calc, "model": args.model,
           "anneal": args.anneal, "scale": args.scale, "mode": args.mode,
           "seed": args.seed, "target": args.target,
           "completed": bool(completed), "n_atoms": len(s),
           "anneals": n_anneal["n"],
           "t_grow": round(t_grow, 1), "t_final": round(t_final, 1),
           "grown": grown, "final": final}
    with open(args.results, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
