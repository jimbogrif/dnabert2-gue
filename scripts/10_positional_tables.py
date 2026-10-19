"""Summarize the positional-readout arm.

Reads results/positional_lr_sweep.csv (every grid point) and the k-mer floor,
and writes:

  results/positional_summary.csv   best dev-selected cell per (dataset,
                                   readout): layer, C, class_weight, dev and
                                   test MCC, chosen window where applicable,
                                   and the k-mer floor for that dataset
  results/positional_by_layer.csv  the selected cell at every layer, which is
                                   what the per-layer figure plots

Dev selects throughout; test is read from the row dev chose, never maximized.
"""

import os

import numpy as np
import pandas as pd

import dnabert2_gue as SL

R = SL.RESULTS_DIR
W = SL.POSITIONAL_N_WINDOWS

grid = pd.read_csv(os.path.join(R, "positional_lr_sweep.csv"),
                   keep_default_na=False, na_values=[""])
sel = grid[grid["selected"]].copy()
assert sel.groupby(["dataset", "readout", "layer"]).size().eq(1).all(), \
    "more than one selected row in some cell"

sel["window"] = sel["window"].astype("Int64")
by_layer = sel[["dataset", "task", "readout", "layer", "window", "C",
                "class_weight", "dev_mcc", "test_mcc", "at_grid_edge"]]
by_layer = by_layer.sort_values(["dataset", "readout", "layer"])
by_layer.to_csv(os.path.join(R, "positional_by_layer.csv"), index=False)

# best layer per (dataset, readout), chosen on dev
best = sel.loc[sel.groupby(["dataset", "readout"])["dev_mcc"].idxmax()].copy()

# k-mer floor: splice has its own file, the other two are in the 26-dataset run
floors = []
for f in ("bag_results.csv", "bag_results_ssp.csv"):
    p = os.path.join(R, f)
    if os.path.exists(p):
        floors.append(pd.read_csv(p, keep_default_na=False, na_values=[""]))
floor_mcc = {}
if floors:
    bag = pd.concat(floors, ignore_index=True)
    bag = bag[bag["feat"].str.startswith("kmer")]
    # dev picks the featurization and the head, exactly as in section 4
    agg = (bag.groupby(["dataset", "head_type", "feat"])
              [["dev_mcc", "test_mcc"]].mean().reset_index())
    pick = agg.loc[agg.groupby("dataset")["dev_mcc"].idxmax()]
    floor_mcc = dict(zip(pick.dataset, pick.test_mcc))

best["kmer_floor_test_mcc"] = best["dataset"].map(floor_mcc)
best["minus_floor"] = best["test_mcc"] - best["kmer_floor_test_mcc"]

mean_by_ds = (best[best.readout == "mean"]
              .set_index("dataset")["test_mcc"].to_dict())
best["minus_mean_pooling"] = best.apply(
    lambda r: r["test_mcc"] - mean_by_ds.get(r["dataset"], np.nan), axis=1)

cols = ["dataset", "task", "readout", "layer", "window", "C", "class_weight",
        "dev_mcc", "test_mcc", "minus_mean_pooling", "kmer_floor_test_mcc",
        "minus_floor", "at_grid_edge"]
best = best[cols].sort_values(["dataset", "readout"])
best.to_csv(os.path.join(R, "positional_summary.csv"), index=False)

print(best.to_string(index=False))
n_edge = int(sel["at_grid_edge"].sum())
print(f"\nselected cells at a C grid edge: {n_edge} / {len(sel)}")
if n_edge:
    print(sel[sel.at_grid_edge].groupby(["readout", "C"]).size().to_string())
