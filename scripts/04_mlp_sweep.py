"""MLP arm: the notebook's TaperedMLP on the layers chosen by the LR sweep.

26 datasets x 2 poolings x 3 selected layers x 3 seeds = 468 runs, reading the
same .npy features the LR arm consumed. Per-run rows go to
results/mlp_results.csv and per-epoch curves to results/mlp_histories.csv.gz;
both resume from partial state.

Usage: mlp_sweep.py [batch_size]
"""

import os
import sys
import time

import pandas as pd

import dnabert2_gue as SL

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 32
SEEDS = (42, 43, 44)
OUT = os.path.join(SL.RESULTS_DIR, "mlp_results.csv")
# gzipped CSV rather than parquet: the venv has no pyarrow/fastparquet and the
# table is only ~14k rows, so a parquet engine is not worth installing.
HIST = os.path.join(SL.RESULTS_DIR, "mlp_histories.csv.gz")

sel = pd.read_csv(os.path.join(SL.RESULTS_DIR, "selected_layers.csv"))
layers_for_task = sel.groupby("task")["layer"].apply(list).to_dict()

done = set()
rows = []
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    rows = prev.to_dict("records")
    done = set(zip(prev.dataset, prev.pool, prev.layer, prev.seed))
    print(f"resuming: {len(done)} runs already present", flush=True)

hists = []
if os.path.exists(HIST):
    hists = [pd.read_csv(HIST)]

plan = [(task, d, p, L, s)
        for task, d, _ in SL.manifest()
        for p in SL.POOLS
        for L in layers_for_task[task]
        for s in SEEDS]
print(f"{len(plan)} runs planned, batch_size={BATCH}", flush=True)

t_all = time.time()
for i, (task, d, p, L, s) in enumerate(plan, 1):
    if (d, p, L, s) in done:
        continue
    t0 = time.time()
    res, hist = SL.run_mlp(d, p, L, s, batch_size=BATCH)
    res["wall_s"] = round(time.time() - t0, 1)
    rows.append(res)

    h = pd.DataFrame(hist)
    h["dataset"], h["pool"], h["layer"], h["seed"] = d, p, L, s
    hists.append(h)

    print(f"[{i:3d}/{len(plan)}] {d:24s} {p:4s} L{L:02d} s{s} "
          f"dev={res['dev_mcc']:.4f} test={res['test_mcc']:.4f} "
          f"ep={res['best_epoch']:3d}/{res['epochs_run']:3d} "
          f"{res['wall_s']:6.1f}s", flush=True)

    if i % 20 == 0 or i == len(plan):
        pd.DataFrame(rows).to_csv(OUT, index=False)
        pd.concat(hists, ignore_index=True).to_csv(HIST, index=False)

pd.DataFrame(rows).to_csv(OUT, index=False)
pd.concat(hists, ignore_index=True).to_csv(HIST, index=False)
print(f"{len(rows)} runs -> {OUT}   TOTAL {time.time() - t_all:.1f}s", flush=True)
