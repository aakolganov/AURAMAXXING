#!/bin/sh
# MACE-OMAT sweep grid for the A10G box (anneal scheme x class-floor scale x growth
# mode). Sequential -- one GPU. Resumable: a cell whose tag is already in the results
# file is skipped, so the script can be re-run after any interruption.
#
# Box setup (see memory/gpu-validation-instance + mace-omat-venv): the system mace on
# the A10G DLAMI is ABI-broken; build the clean venv once:
#   python3.12 -m venv ~/macevenv
#   ~/macevenv/bin/pip install -q --upgrade pip
#   ~/macevenv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128
#   ~/macevenv/bin/pip install -q -U mace-torch
#   ~/macevenv/bin/pip install -q ase numpy scipy networkx matplotlib pyyaml tqdm
# Then, from this directory (repo pulled to ~/AURAMAXXING, branch cutoff_fixing):
#   nohup sh run_grid.sh > grid.log 2>&1 &     # or run inside tmux
#
# Grid rationale (from the local BKS sweep -- BKS has an inherent O-O repulsion bias,
# hence this MLIP re-run): jam frequency is placement-side, so at scales <= 0.9 the
# anneal scheme is moot (expect ~0 anneals) -- those cells run with the default scheme
# only. The anneal-scheme axis is only probed at scale 1.0 where jams actually fire.
# The slow x deposition cells are the most expensive (most anneals x 4x steps) and run
# LAST so an interrupted session loses nothing else.

PY=${PY:-$HOME/macevenv/bin/python}
export PYTHONPATH=${PYTHONPATH:-$HOME/AURAMAXXING}
export PYTHONHASHSEED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
CALC=${CALC:-mace}
MODEL=${MODEL:-medium-omat-0}
RESULTS=${RESULTS:-sweep_mlip.jsonl}
SEEDS=${SEEDS:-"1 2"}

run_cell() {
    tag="${CALC}_$1_s$2_$3_$4"
    if [ -f "$RESULTS" ] && grep -q "\"tag\": \"$tag\"" "$RESULTS"; then
        echo "[skip] $tag (already in $RESULTS)"
        return
    fi
    echo "[run ] $tag  $(date '+%H:%M:%S')"
    "$PY" sweep_cell_mlip.py --anneal "$1" --scale "$2" --mode "$3" --seed "$4" \
        --calc "$CALC" --model "$MODEL" --device cuda --results "$RESULTS" \
        || echo "[FAIL] $tag"
}

# Tier 1 -- jam-free frontier check (fast: placement + one finalize each)
for mode in default deposition; do
    for scale in 0.85 0.9; do
        for seed in $SEEDS; do run_cell default "$scale" "$mode" "$seed"; done
    done
done

# Tier 2 -- anneal-scheme axis at the jammy scale
for mode in default deposition; do
    for anneal in default cool; do
        for seed in $SEEDS; do run_cell "$anneal" 1.0 "$mode" "$seed"; done
    done
done
for seed in $SEEDS; do run_cell slow 1.0 default "$seed"; done

# Tier 3 -- the expensive tail (slow quench x deposition; interruptible)
for seed in $SEEDS; do run_cell slow 1.0 deposition "$seed"; done

echo "[grid done] $(date '+%H:%M:%S')"
"$PY" summarize.py "$RESULTS" || true
