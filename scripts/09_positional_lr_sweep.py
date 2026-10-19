"""Logistic-regression grid over (dataset, readout, layer) for the positional arm.

Readouts, all over the same cached forward pass:

  mean        768-d   whole-sequence mean pooling -- the incumbent
  cls         768-d   position 0
  winW        W*768   the W bp windows concatenated
  winW_best   768-d   ONE window, chosen on dev; the chosen index is reported
  winWshuf    W*768   winW with its W blocks permuted per sequence

winW_best and winWshuf are derived from the winW cache, not stored.

winWshuf is the dimension control: it answers "is winW better because it is
positional, or merely because it has W times the features?" and is therefore
only informative where winW is actually read -- pass --readouts winWshuf
--layers <best> once the main pass has named each dataset's layer, instead of
paying for all 12.

Dev MCC selects within each cell; test MCC is computed once, for the selected
setting only. One CSV per cell under results/positional_lr_cells/ so the sweep
resumes after an interruption.

Usage:
    python scripts/09_positional_lr_sweep.py
    python scripts/09_positional_lr_sweep.py --readouts win8shuf --layers 6
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import dnabert2_gue as SL

CELL_DIR = os.path.join(SL.RESULTS_DIR, "positional_lr_cells")
OUT = os.path.join(SL.RESULTS_DIR, "positional_lr_sweep.csv")
W = SL.POSITIONAL_N_WINDOWS
WIN = f"win{W}"

# Lower than 03_lr_sweep.py's default on purpose: a winW cell is W x the
# feature width, so each worker holds roughly W x the matrix. At W=8 and
# 47,356 training rows that is ~1.2 GB of features per split before
# StandardScaler's copy, and this host has 31 GB shared with a desktop.
N_JOBS = int(os.environ.get("SWEEP_N_JOBS", 3))

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="*", default=list(SL.POSITIONAL_DATASETS))
ap.add_argument("--readouts", nargs="*",
                default=["mean", "cls", WIN, f"{WIN}_best"])
ap.add_argument("--layers", nargs="*", type=int,
                default=list(range(1, SL.N_LAYERS + 1)))
ap.add_argument("--n-jobs", type=int, default=N_JOBS)
args = ap.parse_args()

os.makedirs(CELL_DIR, exist_ok=True)


def cell_path(dataset, readout, layer):
    return os.path.join(CELL_DIR, f"{dataset}__{readout}__L{layer:02d}.csv")


def load_readout(dataset, split, layer, readout):
    """Feature matrix for one readout, derived from the cache where needed."""
    if readout in ("mean", "cls", WIN):
        return SL.load_split(dataset, split, layer, readout)
    X, y = SL.load_split(dataset, split, layer, WIN)
    if readout == f"{WIN}shuf":
        return SL.shuffle_windows(X, W, SL.POSITIONAL_SHUFFLE_SEED), y
    raise ValueError(readout)


def run_cell(dataset, readout, layer):
    p = cell_path(dataset, readout, layer)
    if os.path.exists(p):
        return p, 0.0
    t0 = time.time()
    meta = {"dataset": dataset, "task": SL.DATASET_TASK[dataset],
            "readout": readout, "pool": readout, "layer": layer,
            "n_windows": W if readout.startswith(WIN) else 0}

    if readout == f"{WIN}_best":
        # Fit each 768-d window separately and let dev choose one. Test is
        # scored once overall, not once per window: fit_lr_grid fills
        # test_mcc for each window's own best, so every row that is not the
        # global dev winner has that value cleared.
        rows = []
        for k in range(W):
            sl = slice(k * SL.HIDDEN, (k + 1) * SL.HIDDEN)
            splits = []
            for s in SL.SPLITS:
                X, y = SL.load_split(dataset, s, layer, WIN)
                splits += [X[:, sl], y]
            rows += SL.fit_lr_grid(tuple(splits), {**meta, "window": k})
        best = max(range(len(rows)), key=lambda i: rows[i]["dev_mcc"])
        for i, r in enumerate(rows):
            r["selected"] = (i == best)
            if i != best:
                r["test_mcc"] = np.nan
    else:
        splits = []
        for s in SL.SPLITS:
            X, y = load_readout(dataset, s, layer, readout)
            splits += [X, y]
        rows = SL.fit_lr_grid(tuple(splits), {**meta, "window": -1})

    tmp = p + ".tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, p)
    dt = time.time() - t0
    print(f"{dataset:24s} {readout:10s} L{layer:02d}  {dt:7.1f}s", flush=True)
    return p, dt


cells = [(d, r, L) for d in args.datasets
         for r in args.readouts for L in args.layers]
print(f"{len(cells)} cells, n_jobs={args.n_jobs}", flush=True)

t_all = time.time()
done = Parallel(n_jobs=args.n_jobs, verbose=0)(
    delayed(run_cell)(d, r, L) for d, r, L in cells)

grid = pd.concat([pd.read_csv(p) for p in sorted(
    os.path.join(CELL_DIR, f) for f in os.listdir(CELL_DIR)
    if f.endswith(".csv"))], ignore_index=True)
grid.to_csv(OUT, index=False)
print(f"{len(grid)} rows -> {OUT}   TOTAL {time.time() - t_all:.1f}s", flush=True)

sel = grid[grid["selected"]]
print(f"selected rows: {len(sel)}", flush=True)
edge = sel[sel["at_grid_edge"]]
print(f"selected cells at a C grid edge: {len(edge)} / {len(sel)}", flush=True)
if len(edge):
    print(edge.groupby(["readout", "C"]).size().to_string(), flush=True)
