"""Bag-of-tokens floor: k-mer and BPE-vocabulary baselines on the same 26 datasets.

Five featurizations (kmer3/4/5/6 and DNABERT-2's own 4,096-token BPE vocabulary)
x two heads (the frozen arm's LR grid and its TaperedMLP), run through the
identical downstream protocol as the embedding arms so that only the input
encoding differs.

Why both heads: the ceiling this is a floor for -- best_frozen_vs_paper.csv --
is itself selected per dataset across LR *and* MLP, so a floor selected by a
different rule would not be apples-to-apples. The LR half is also the cheap one
(1,872 fits, minutes), so dropping it would save nothing.

Why the featurization is selected on dev rather than fixed: k is a
hyperparameter exactly like C. The embedding arm puts all 18 C x class_weight
points in the history table with dev_selected marking the winner; the k-mer arm
puts 4 k x 9 C x 2 class_weight = 72 in and marks one. Note the budgets land the
right way round -- the embedding arm gets 432 dev evaluations per dataset
(9 x 2 x 12 layers x 2 poolings), the bag arm 72. A floor should not be better
tuned than the thing it is a floor for.

Outputs results/bag_lr_grid.csv (every grid point) and results/bag_results.csv
(the per-dataset selected rows for both heads, the file 06_build_tables.py
reads). Both halves resume. LR is CPU-parallel; the MLP half needs torch and
will use the GPU if one is present.

Usage: 07_bag_baseline.py [--heads lr mlp] [--feats kmer3 ... bpe] [--jobs N]
"""

import argparse
import os
import sys
import time

import pandas as pd
from joblib import Parallel, delayed

import dnabert2_gue as SL
from dnabert2_gue.bags import FEATS, grid_search_bag, run_bag_mlp

LR_CELLS = os.path.join(SL.RESULTS_DIR, "bag_lr_cells")
LR_GRID = os.path.join(SL.RESULTS_DIR, "bag_lr_grid.csv")
OUT = os.path.join(SL.RESULTS_DIR, "bag_results.csv")
HIST = os.path.join(SL.RESULTS_DIR, "bag_histories.csv.gz")
SEEDS = (42, 43, 44)

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--heads", nargs="+", choices=["lr", "mlp"], default=["lr", "mlp"])
ap.add_argument("--feats", nargs="+", choices=list(FEATS), default=list(FEATS))
ap.add_argument("--jobs", type=int, default=int(os.environ.get("SWEEP_N_JOBS",
                                                 max(1, (os.cpu_count() or 4) // 3))))
ap.add_argument("--debug", action="store_true")
args = ap.parse_args()

os.makedirs(LR_CELLS, exist_ok=True)
NEEDS_TOK = "bpe" in args.feats


def _tok():
    """DNABERT-2's tokenizer -- no model, no forward pass, CPU only."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(SL.MODEL_NAME, trust_remote_code=True)


# ---- LR half: one CSV per (dataset, feat) cell, so it resumes ---------------

def lr_cell(dataset, feat):
    p = os.path.join(LR_CELLS, f"{dataset}__{feat}.csv")
    if os.path.exists(p):
        return p
    t0 = time.time()
    rows = grid_search_bag(dataset, feat, tok=_tok() if feat == "bpe" else None)
    tmp = p + ".tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, p)
    sel = [r for r in rows if r["selected"]][0]
    print(f"  LR  {dataset:24s} {feat:6s} {sel['n_features']:5d}f "
          f"dev={sel['dev_mcc']:.4f} test={sel['test_mcc']:.4f} "
          f"C={sel['C']:g} {time.time() - t0:6.1f}s", flush=True)
    return p


if "lr" in args.heads:
    cells = [(d, f) for _, d, _ in SL.manifest() for f in args.feats]
    print(f"LR half: {len(cells)} cells ({len(args.feats)} feats x 26 datasets), "
          f"n_jobs={args.jobs}", flush=True)
    t0 = time.time()
    paths = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(lr_cell)(d, f) for d, f in cells)
    grid = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    grid.to_csv(LR_GRID, index=False)
    print(f"LR half: {len(grid)} grid rows -> {LR_GRID}  "
          f"TOTAL {time.time() - t0:.1f}s", flush=True)

# ---- MLP half: one row per (dataset, feat, seed), flush every 20 ------------

if "mlp" in args.heads:
    rows, done, hists = [], set(), []
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        prev = prev[prev.head_type == "MLP"]
        rows = prev.to_dict("records")
        done = set(zip(prev.dataset, prev.feat, prev.seed))
        print(f"MLP half: resuming, {len(done)} runs present", flush=True)
    if os.path.exists(HIST):
        hists = [pd.read_csv(HIST)]

    tok = _tok() if NEEDS_TOK else None
    plan = [(d, f, s) for _, d, _ in SL.manifest() for f in args.feats for s in SEEDS]
    print(f"MLP half: {len(plan)} runs planned", flush=True)
    t0, n_new = time.time(), 0
    for i, (d, f, s) in enumerate(plan, 1):
        if (d, f, s) in done:
            continue
        t1 = time.time()
        res, hist = run_bag_mlp(d, f, s, tok=tok, debug=args.debug)
        res["head_type"] = "MLP"
        res["wall_s"] = round(time.time() - t1, 1)
        rows.append(res)
        n_new += 1
        h = pd.DataFrame(hist)
        h["dataset"], h["feat"], h["seed"] = d, f, s
        hists.append(h)
        print(f"  MLP {d:24s} {f:6s} s{s} dev={res['dev_mcc']:.4f} "
              f"test={res['test_mcc']:.4f} ep={res['best_epoch']}/{res['epochs_run']} "
              f"{res['wall_s']:6.1f}s", flush=True)
        if n_new % 20 == 0:
            pd.DataFrame(rows).to_csv(OUT + ".mlp.tmp", index=False)

    mlp_df = pd.DataFrame(rows)
    print(f"MLP half: {len(mlp_df)} runs  TOTAL {time.time() - t0:.1f}s", flush=True)
    if hists:
        pd.concat(hists, ignore_index=True).to_csv(HIST, index=False)
else:
    mlp_df = pd.DataFrame()

# ---- combine into the one file 06_build_tables.py reads --------------------
# LR contributes its dev-selected row per (dataset, feat); MLP contributes every
# seed, matching how the embedding arms report their two heads.

frames = []
if os.path.exists(LR_GRID):
    g = pd.read_csv(LR_GRID, keep_default_na=False, na_values=[""])
    sel = g[g["selected"]].copy()
    sel["head_type"] = "LR"
    sel["seed"] = pd.NA
    frames.append(sel)
if len(mlp_df):
    frames.append(mlp_df)
if frames:
    out = pd.concat(frames, ignore_index=True).sort_values(
        ["task", "dataset", "feat", "head_type", "seed"], na_position="first")
    out.to_csv(OUT, index=False)
    print(f"\n{len(out)} rows -> {OUT}")
    print(out.groupby(["head_type", "feat"]).dev_mcc.mean().round(4).to_string())
    tmp = OUT + ".mlp.tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
