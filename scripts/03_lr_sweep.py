"""Logistic-regression grid over every (dataset, pooling, layer) cell.

26 datasets x 2 poolings x 12 layers x (9 C values x 2 class_weight) = 11,232
fits. Dev MCC selects within each cell; test MCC is computed once, for the
selected setting only. Each cell is written to its own CSV under
results/lr_cells/ so the sweep resumes after an interruption.
"""

import os
import sys
import time

import pandas as pd
from joblib import Parallel, delayed

import dnabert2_gue as SL

CELL_DIR = os.path.join(SL.RESULTS_DIR, "lr_cells")
OUT = os.path.join(SL.RESULTS_DIR, "lr_layer_sweep.csv")
# 6 jobs x OMP_NUM_THREADS=3 was measured best on a 20-core box; scale that
# split to the host. SWEEP_N_JOBS overrides.
N_JOBS = int(os.environ.get("SWEEP_N_JOBS", max(1, (os.cpu_count() or 4) // 3)))

os.makedirs(CELL_DIR, exist_ok=True)


def cell_path(dataset, pool, layer):
    return os.path.join(CELL_DIR, f"{dataset}__{pool}__L{layer:02d}.csv")


def run_cell(dataset, pool, layer):
    p = cell_path(dataset, pool, layer)
    if os.path.exists(p):
        return p, 0.0
    t0 = time.time()
    rows = SL.grid_search_lr(dataset, pool, layer)
    tmp = p + ".tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, p)
    dt = time.time() - t0
    print(f"{dataset:24s} {pool:4s} L{layer:02d}  {dt:6.1f}s", flush=True)
    return p, dt


cells = [(d, p, L)
         for _, d, _ in SL.manifest()
         for p in SL.POOLS
         for L in range(1, SL.N_LAYERS + 1)]
print(f"{len(cells)} cells, n_jobs={N_JOBS}", flush=True)

t_all = time.time()
done = Parallel(n_jobs=N_JOBS, verbose=0)(
    delayed(run_cell)(d, p, L) for d, p, L in cells)

grid = pd.concat([pd.read_csv(p) for p, _ in done], ignore_index=True)
grid.to_csv(OUT, index=False)
print(f"{len(grid)} rows -> {OUT}   TOTAL {time.time() - t_all:.1f}s", flush=True)

sel = grid[grid["selected"]]
print(f"selected rows: {len(sel)} (expected {len(cells)})", flush=True)
print(f"at grid edge:  {int(sel['at_grid_edge'].sum())}", flush=True)
print(sel.groupby("task")["test_mcc"].agg(["count", "mean", "max"]).to_string(),
      flush=True)
