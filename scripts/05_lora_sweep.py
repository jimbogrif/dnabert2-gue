"""LoRA arm: fine-tune DNABERT-2 with LoRA adapters on the in-scope datasets.

Two modes:

--mode calibrate [--lrs ...]
    3 datasets x lrs x seed 42 x the ``attn`` variant (15 runs at the default
    grid, ~1.6 h expected) -> results/lora_calibration.csv,
    results/lora_calibration_histories.csv.gz and
    figures/lora_calibration.png. Selects the lr with the highest mean dev MCC
    over the three datasets, then checks the decision gate below. Never starts
    the full run.

    Grid rationale: the notebook ran lr=1e-4 at batch 32 and reached dev MCC
    0.6528 on EMP__H3 while still climbing, against a frozen-MLP 0.7215 +/-
    0.0064 on the same dataset -- the one validated setting loses to the frozen
    probe. 1e-4 is also the bottom of the 1e-4..5e-4 band conventional for
    LoRA, and batch is now 4x larger, so the grid spans that band, its
    batch-scaled range and two points above it: {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}.
    A first pass capped at 5 epochs put the optimum at the 1e-3 edge, but every
    run there hit the cap, so that ranking measured how far each rate got in a
    fixed budget rather than how good it is; the grid is rerun under patience. The datasets span the size range:
    mouse__3 (1,904 train), prom__prom_core_tata (4,904), EMP__H3K4me3 (29,439).

    If the chosen lr is at a grid edge, extend the grid one point in that
    direction (``--lrs`` with the extra value; completed runs are skipped) and
    recalibrate before the full run, as the LR arm's widened C grid did.

    Decision gate (exit status 2 if any trips -- report, do not run full):
      - the epoch cap bound (patience never fired) on >=2 of the 3 datasets at
        the chosen lr: LORA_MAX_EPOCHS is too few;
      - the chosen lr is at a grid edge;
      - every lr leaves dev MCC below the best frozen dev MCC on all three
        datasets.

--mode full --lr LR [--variant {attn,attn_mlp,both}]
    26 datasets x seeds (42, 43, 44) x variants = 156 runs for ``both``.
    Cost depends on where patience stops each run: ~11 h if runs average 10
    epochs, 44 h worst case if none early-stops at LORA_MAX_EPOCHS=40.
    EMP is 76% of the work (500 bp at 459 seq/s vs 3,831 at 70 bp). Per-run rows -> results/lora_results.csv, per-epoch curves ->
    results/lora_histories.csv.gz (local diagnostics, gitignored). Both
    variants run at the lr calibrated for ``attn``.

Both modes resume: a run is skipped if (dataset, seed, variant, lr) is already
in the output CSV; rows are flushed every 20 runs. Adapter weights are not
saved. Requires a CUDA GPU.

Then run 06_build_tables.py, which picks up lora_results.csv.

Usage: 05_lora_sweep.py --mode calibrate [--lrs 1e-4 3e-4 1e-3 3e-3 1e-2]
       05_lora_sweep.py --mode full --lr 3e-4 [--variant both]
"""

import argparse
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import dnabert2_gue as SL
from dnabert2_gue.lora import RESULT_COLS

matplotlib.use("Agg")   # non-interactive; safe to switch after importing pyplot

CALIB = os.path.join(SL.RESULTS_DIR, "lora_calibration.csv")
CALIB_HIST = os.path.join(SL.RESULTS_DIR, "lora_calibration_histories.csv.gz")
OUT = os.path.join(SL.RESULTS_DIR, "lora_results.csv")
HIST = os.path.join(SL.RESULTS_DIR, "lora_histories.csv.gz")
FIG = os.path.join(SL.FIGURES_DIR, "lora_calibration.png")
CALIB_SEED, CALIB_VARIANT = 42, "attn"

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--mode", choices=["calibrate", "full"], required=True)
ap.add_argument("--lrs", type=float, nargs="+", default=list(SL.LORA_CALIB_LRS),
                help="calibration lr grid (calibrate mode)")
ap.add_argument("--lr", type=float, help="calibrated lr (full mode)")
ap.add_argument("--variant", choices=["attn", "attn_mlp", "both"], default="both")
ap.add_argument("--debug", action="store_true", help="print per-epoch progress")
ap.add_argument("--datasets", nargs="+", default=None,
                help="default: the 26 in-scope datasets. Others (e.g. "
                     "splice__reconstructed) are appended to the same "
                     "results file; the resume key is (dataset, variant, "
                     "seed, lr) so nothing already run is repeated.")
args = ap.parse_args()
if args.mode == "full" and args.lr is None:
    ap.error("--mode full requires --lr (the value chosen by --mode calibrate)")

if not torch.cuda.is_available():
    sys.exit("05_lora_sweep.py needs a CUDA GPU (>=8 GB VRAM free; 6.4 GiB peak at "
             "batch 128). Fine-tuning on CPU would take weeks.")


def key(d, s, v, lr):
    return (d, int(s), v, float(lr))


def run_plan(plan, out, hist_path=None):
    """Run (dataset, seed, variant, lr) tuples, resuming from ``out``."""
    rows, done, hists = [], set(), []
    if os.path.exists(out):
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = {key(*k) for k in zip(prev.dataset, prev.seed, prev.variant, prev.lr)}
        print(f"resuming: {len(done)} runs already present", flush=True)
    if hist_path and os.path.exists(hist_path):
        hists = [pd.read_csv(hist_path)]

    def flush():
        cols = RESULT_COLS + [c for c in pd.DataFrame(rows).columns if c not in RESULT_COLS]
        pd.DataFrame(rows)[cols].to_csv(out, index=False)
        if hist_path and hists:
            pd.concat(hists, ignore_index=True).to_csv(hist_path, index=False)

    print(f"{len(plan)} runs planned", flush=True)
    t_all, n_new = time.time(), 0
    for i, (d, s, v, lr) in enumerate(plan, 1):
        if key(d, s, v, lr) in done:
            continue
        t0 = time.time()
        res, hist = SL.run_lora(d, s, lr, variant=v, debug=args.debug)
        res["wall_s"] = round(time.time() - t0, 1)
        rows.append(res)
        n_new += 1

        h = pd.DataFrame(hist)
        h["dataset"], h["seed"], h["variant"], h["lr"] = d, s, v, lr
        hists.append(h)

        print(f"[{i:3d}/{len(plan)}] {d:24s} {v:8s} s{s} lr={lr:g} "
              f"dev={res['dev_mcc']:.4f} test={res['test_mcc']:.4f} "
              f"ep={res['best_epoch']}/{res['epochs_run']} "
              f"{res['wall_s']:6.1f}s", flush=True)
        if n_new % 20 == 0:
            flush()
    if rows:
        flush()
    print(f"{len(rows)} runs -> {out}   TOTAL {time.time() - t_all:.1f}s", flush=True)
    return pd.read_csv(out)


def calibrate():
    lrs = sorted(set(args.lrs))
    plan = [(d, CALIB_SEED, CALIB_VARIANT, lr)
            for d in SL.LORA_CALIB_DATASETS for lr in lrs]
    cal = run_plan(plan, CALIB, CALIB_HIST)

    # at_grid_edge relative to every lr calibrated so far (the grid may have
    # been extended by an earlier --lrs call).
    cal = cal[(cal.seed == CALIB_SEED) & (cal.variant == CALIB_VARIANT)
              & cal.dataset.isin(SL.LORA_CALIB_DATASETS)].copy()
    grid = sorted(cal.lr.unique())
    cal["at_grid_edge"] = cal.lr.isin([grid[0], grid[-1]])
    cal.to_csv(CALIB, index=False)

    complete = [lr for lr in grid
                if set(cal.loc[cal.lr == lr, "dataset"]) == set(SL.LORA_CALIB_DATASETS)]
    mean_dev = cal[cal.lr.isin(complete)].groupby("lr")["dev_mcc"].mean()
    chosen = float(mean_dev.idxmax())
    at_edge = chosen in (grid[0], grid[-1])

    agg = pd.read_csv(os.path.join(SL.RESULTS_DIR, "history_table_aggregated.csv"))
    frozen = (agg[agg.model_type.isin(["LR", "MLP"])]
              .groupby("dataset")["dev_mcc_mean"].max())

    table = cal.pivot(index="dataset", columns="lr", values="dev_mcc")
    table["best_frozen_dev"] = frozen.reindex(table.index)
    print("\ndev MCC by lr (seed 42, attn):")
    print(table.round(4).to_string())
    print("\nmean dev MCC by lr:")
    print(mean_dev.round(4).to_string())
    print(f"\nchosen lr = {chosen:g}  (highest mean dev MCC)  at_grid_edge = {at_edge}")

    # ---- figure ---------------------------------------------------------------
    ds = list(SL.LORA_CALIB_DATASETS)
    fig, axes = plt.subplots(1, len(ds), figsize=(4 * len(ds), 3.4), sharex=True)
    for ax, d in zip(axes, ds):
        sub = cal[cal.dataset == d].sort_values("lr")
        ax.plot(sub.lr, sub.dev_mcc, "o-", label="LoRA attn (dev)")
        ax.axhline(frozen[d], ls="--", c="grey", label="best frozen (dev)")
        ax.axvline(chosen, ls=":", c="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(d)
        ax.set_xlabel("learning rate")
    axes[0].set_ylabel("dev MCC")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(SL.FIGURES_DIR, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")

    # ---- decision gate --------------------------------------------------------
    gate = []
    at_chosen = cal[cal.lr == chosen]
    n_capped = int((~at_chosen.early_stopped.astype(bool)).sum())
    if n_capped >= 2:
        gate.append(f"the epoch cap bound on {n_capped}/3 datasets at lr={chosen:g} "
                    f"(patience {SL.LORA_PATIENCE} never fired within "
                    f"LORA_MAX_EPOCHS={SL.LORA_MAX_EPOCHS}). Ask before raising it.")
    if at_edge:
        step = np.sqrt(10)
        nxt = grid[0] / step if chosen == grid[0] else grid[-1] * step
        gate.append(f"chosen lr {chosen:g} is at a grid edge. Extend: --lrs "
                    + " ".join(f"{x:g}" for x in grid + [float(f"{nxt:.0e}")])
                    + " (if already extended once, ask).")
    below = all((cal.loc[cal.dataset == d, "dev_mcc"] < frozen[d]).all() for d in ds)
    if below:
        gate.append("every lr leaves dev MCC below the best frozen dev MCC on all "
                    "three datasets.")
    if gate:
        print("\nGATE TRIPPED -- report before --mode full:")
        for g in gate:
            print("  -", g)
        sys.exit(2)
    print(f"\nGATE CLEAR. Next: 05_lora_sweep.py --mode full --lr {chosen:g}")


def full():
    variants = ["attn", "attn_mlp"] if args.variant == "both" else [args.variant]
    datasets = args.datasets or [d for _, d, _ in SL.manifest()]
    plan = [(d, s, v, args.lr)
            for d in datasets
            for v in variants
            for s in SL.LORA_SEEDS]
    run_plan(plan, OUT, HIST)


if args.mode == "calibrate":
    calibrate()
else:
    full()
