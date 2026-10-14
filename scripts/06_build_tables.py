"""Assemble the combined LR + MLP (+ LoRA) history table and its seed-aggregated companion.

Row granularity: one row per (model_type, dataset, pool, layer, seed, C,
class_weight). LR rows span the full 18-point C x class_weight grid at each of
the three selected layers, with test MCC populated only on the dev-selected
setting; MLP rows carry one row per seed with its training hyperparameters.

LoRA rows are read from results/lora_results.csv only if it exists, so the
pipeline runs before the fine-tuning arm has. Fine-tuning has no readout layer
and no free pooling choice, so LoRA rows carry the sentinels ``layer = 0``
(real layers are 1-12), ``layer_role = "finetuned"`` and
``pool = "cls-trained"``. Each LoRA variant gets its own model_type
("LoRA-attn", "LoRA-attn+mlp") so the seed aggregate, keyed on model_type,
never averages the two variants together. Append new runs to
lora_results.csv, never to history_table.csv -- the tables here are
regenerated wholesale.
"""

import os
import sys

import numpy as np
import pandas as pd

import dnabert2_gue as SL

R = SL.RESULTS_DIR
na = dict(keep_default_na=False, na_values=[""])

selected = pd.read_csv(os.path.join(R, "selected_layers.csv"))
role = selected.set_index(["task", "layer"])["role"].to_dict()
keep = set(zip(selected.task, selected.layer))

grid = pd.read_csv(os.path.join(R, "lr_layer_sweep.csv"), **na)
mlp = pd.read_csv(os.path.join(R, "mlp_results.csv"), **na)
LORA = os.path.join(R, "lora_results.csv")
lora = pd.read_csv(LORA, **na) if os.path.exists(LORA) else None

BAG = os.path.join(R, "bag_results.csv")
bag = pd.read_csv(BAG, **na) if os.path.exists(BAG) else None

# ---- LR rows, restricted to the three selected layers per task --------------
lr = grid[[(t, l) in keep for t, l in zip(grid.task, grid.layer)]].copy()
lr["model_type"] = "LR"
lr["seed"] = np.nan
lr = lr.rename(columns={"selected": "dev_selected"})
lr["hyperparams"] = ("C=" + lr["C"].map(lambda x: f"{x:g}")
                     + ", class_weight=" + lr["class_weight"]
                     + ", max_iter=5000, solver=lbfgs")

# ---- MLP rows ---------------------------------------------------------------
mlp["model_type"] = "MLP"
mlp["C"] = np.nan
mlp["class_weight"] = "n/a"
mlp["n_iter"] = np.nan
mlp["at_grid_edge"] = False
mlp["in_notebook_grid"] = False
mlp["dev_selected"] = True          # every MLP run is reported, none discarded
mlp["hyperparams"] = (
    "dims=" + mlp["hidden_dims"].astype(str) + ", dropout=" + mlp["dropout"].astype(str)
    + ", lr=" + mlp["lr"].astype(str)
    + ", weight_decay=" + mlp["weight_decay"].astype(str)
    + ", batch=" + mlp["batch_size"].astype(str)
    + ", optimizer=AdamW, max_epochs=" + mlp["max_epochs"].astype(str)
    + ", patience=" + mlp["patience"].astype(str))

# ---- LoRA rows (optional third arm) ----------------------------------------
if lora is not None:
    LORA_TYPES = {"attn": "LoRA-attn", "attn_mlp": "LoRA-attn+mlp"}
    lora["model_type"] = lora["variant"].map(LORA_TYPES)
    assert lora["model_type"].notna().all(), \
        f"unknown LoRA variant(s): {set(lora.variant) - set(LORA_TYPES)}"
    lora["pool"] = "cls-trained"
    lora["layer"] = 0                   # sentinel: no readout layer, all 12 update
    lora["C"] = np.nan
    lora["class_weight"] = "n/a"
    lora["n_iter"] = np.nan
    lora["at_grid_edge"] = False
    lora["in_notebook_grid"] = False
    lora["dev_selected"] = True         # every LoRA run is reported, none discarded
    lora["hyperparams"] = (
        "r=" + lora["lora_r"].astype(str) + ", alpha=" + lora["lora_alpha"].astype(str)
        + ", lora_dropout=" + lora["lora_dropout"].astype(str)
        + ", target_modules=" + lora["target_modules"].astype(str)
        + ", modules_to_save=" + lora["modules_to_save"].astype(str)
        + ", lr=" + lora["lr"].astype(str)
        + ", weight_decay=" + lora["weight_decay"].astype(str)
        + ", batch=" + lora["batch_size"].astype(str)
        + ", optimizer=" + lora["optimizer"].astype(str)
        + ", precision=" + lora["precision"].astype(str)
        + ", max_epochs=" + lora["max_epochs"].astype(str)
        + ", patience=" + lora["patience"].astype(str)
        + ", warmup_ratio=" + lora["warmup_ratio"].astype(str)
        + ", eval=" + lora["eval_cadence"].astype(str))

# ---- bag-of-tokens floor (optional fourth arm) -----------------------------
# Two families, kept as separate model_type values so the aggregate's groupby
# does not pool them: 'kmer-*' answers "does DNABERT-2 beat base composition?",
# 'bpe-*' the sharper "does the transformer beat its own tokenizer?" (same 4,096
# vocabulary, same segmentation, no forward pass). Within kmer-*, k is selected
# on dev exactly as C is -- see the per-dataset selection below.
if bag is not None:
    fam = np.where(bag["feat"].astype(str).str.startswith("kmer"), "kmer", "bpe")
    bag["model_type"] = fam + "-" + bag["head_type"].astype(str)
    bag["layer"] = -1                   # sentinel: LoRA already uses 0
    bag["pool"] = "bag"
    bag["dev_selected"] = True          # every reported bag row is a winner
    bag["in_notebook_grid"] = False
    if "class_weight" not in bag:
        bag["class_weight"] = "n/a"
    bag["class_weight"] = bag["class_weight"].fillna("n/a")
    bag["hyperparams"] = (
        "feat=" + bag["feat"].astype(str)
        + ", n_features=" + bag["n_features"].astype(str)
        + ", head=" + bag["head_type"].astype(str))


COLS = ["model_type", "task", "dataset", "pool", "layer", "layer_role",
        "seed", "C", "class_weight", "dev_mcc", "test_mcc", "dev_selected",
        "best_epoch", "epochs_run", "early_stopped", "batch_size",
        "hidden_dims", "dropout", "lr", "weight_decay", "optimizer",
        "max_epochs", "patience", "n_iter", "at_grid_edge",
        "in_notebook_grid", "hyperparams"]
# Parameter budget is the LoRA arm's headline, so it gets first-class columns
# (blank for LR / MLP). Added only once LoRA rows exist, so the frozen-only
# tables stay byte-identical to their committed versions until then.
if lora is not None:
    COLS += ["trainable_params", "pct_trainable"]
# Which featurization won is the bag arm's headline, so it is first-class too.
if bag is not None:
    COLS += ["feat", "n_features"]

arms = [lr, mlp] + ([lora] if lora is not None else []) + ([bag] if bag is not None else [])
hist = pd.concat(arms, ignore_index=True)
hist["layer_role"] = [role.get((t, l), "bag" if l == -1 else "finetuned")
                      for t, l in zip(hist.task, hist.layer)]
for c in COLS:
    if c not in hist:
        hist[c] = np.nan
hist = hist[COLS].sort_values(
    ["task", "dataset", "pool", "layer", "model_type", "C", "seed"],
    na_position="first").reset_index(drop=True)
hist.to_csv(os.path.join(R, "history_table.csv"), index=False)

# ---- seed-aggregated / cell-best companion ---------------------------------
# One row per (model_type, dataset, pool, layer): for LR the dev-selected
# setting; for MLP the mean +/- sd over the three seeds.
lr_best = (lr[lr.dev_selected]
           .assign(n=1)
           [["model_type", "task", "dataset", "pool", "layer", "dev_mcc",
             "test_mcc", "C", "class_weight", "at_grid_edge", "n"]]
           .rename(columns={"dev_mcc": "dev_mcc_mean",
                            "test_mcc": "test_mcc_mean"}))
lr_best["dev_mcc_sd"] = np.nan
lr_best["test_mcc_sd"] = np.nan
lr_best["best_epoch_mean"] = np.nan

def seed_agg(df):
    out = (df.groupby(["model_type", "task", "dataset", "pool", "layer"])
           .agg(dev_mcc_mean=("dev_mcc", "mean"), dev_mcc_sd=("dev_mcc", "std"),
                test_mcc_mean=("test_mcc", "mean"), test_mcc_sd=("test_mcc", "std"),
                best_epoch_mean=("best_epoch", "mean"), n=("seed", "count"))
           .reset_index())
    out["C"] = np.nan
    out["class_weight"] = "n/a"
    out["at_grid_edge"] = False
    return out


# groupby drops NaN keys, which is why LoRA rows use layer = 0, not NaN
mlp_agg = seed_agg(mlp)
lora_agg = [seed_agg(lora)] if lora is not None else []

bag_agg = []
if bag is not None:
    # LR rows carry no seed; MLP rows carry three. Aggregate each the way the
    # embedding arms do, then let dev pick the featurization per dataset -- so
    # one row per (family, head) per dataset, k chosen like any hyperparameter.
    b_lr = bag[bag.head_type == "LR"]
    b_mlp = bag[bag.head_type == "MLP"]
    parts = []
    if len(b_lr):
        parts.append(b_lr.assign(n=1)[["model_type", "task", "dataset", "pool", "layer",
                                       "feat", "dev_mcc", "test_mcc", "C", "class_weight",
                                       "at_grid_edge", "n"]]
                     .rename(columns={"dev_mcc": "dev_mcc_mean", "test_mcc": "test_mcc_mean"})
                     .assign(dev_mcc_sd=np.nan, test_mcc_sd=np.nan, best_epoch_mean=np.nan))
    if len(b_mlp):
        a = (b_mlp.groupby(["model_type", "task", "dataset", "pool", "layer", "feat"])
             .agg(dev_mcc_mean=("dev_mcc", "mean"), dev_mcc_sd=("dev_mcc", "std"),
                  test_mcc_mean=("test_mcc", "mean"), test_mcc_sd=("test_mcc", "std"),
                  best_epoch_mean=("best_epoch", "mean"), n=("seed", "count"))
             .reset_index())
        a["C"] = np.nan; a["class_weight"] = "n/a"; a["at_grid_edge"] = False
        parts.append(a)
    b = pd.concat(parts, ignore_index=True)
    b = b.loc[b.groupby(["model_type", "dataset"])["dev_mcc_mean"].idxmax()]
    bag_agg = [b.drop(columns=["feat"])]
    b.to_csv(os.path.join(R, "bag_best_per_dataset.csv"), index=False)

agg = pd.concat([lr_best, mlp_agg] + lora_agg + bag_agg, ignore_index=True)
agg["layer_role"] = [role.get((t, l), "bag" if l == -1 else "finetuned")
                     for t, l in zip(agg.task, agg.layer)]
agg = agg[["model_type", "task", "dataset", "pool", "layer", "layer_role",
           "n", "dev_mcc_mean", "dev_mcc_sd", "test_mcc_mean", "test_mcc_sd",
           "best_epoch_mean", "C", "class_weight", "at_grid_edge"]]
agg = agg.sort_values(["task", "dataset", "pool", "layer", "model_type"]
                      ).reset_index(drop=True)
agg.to_csv(os.path.join(R, "history_table_aggregated.csv"), index=False)

# ---- per-dataset best frozen configuration, selected on dev ----------------
# The honest "what can frozen DNABERT-2 embeddings do" number: pick the single
# configuration with the highest dev MCC per dataset, report its test MCC.
# Restricted to the frozen arms so this file keeps its meaning (and checksum)
# once LoRA rows exist.
frozen = agg[agg.model_type.isin(["LR", "MLP"])]
best = frozen.loc[frozen.groupby("dataset")["dev_mcc_mean"].idxmax()].copy()
paper = pd.read_csv(os.path.join(R, "paper_table6_dnabert2.csv"))
best = best.merge(paper[["dataset", "paper_mcc_x100"]], on="dataset")
best["frozen_mcc_x100"] = best["test_mcc_mean"] * 100
best["gap_vs_paper"] = best["frozen_mcc_x100"] - best["paper_mcc_x100"]
best["pct_of_paper"] = 100 * best["frozen_mcc_x100"] / best["paper_mcc_x100"]
best = best.sort_values(["task", "dataset"]).reset_index(drop=True)
best.to_csv(os.path.join(R, "best_frozen_vs_paper.csv"), index=False)

# ---- frozen best and LoRA side by side, only once the LoRA arm has run -----
if lora is not None:
    # best LoRA row per dataset across both variants, selected on dev
    ft_all = agg[agg.model_type.str.startswith("LoRA")]
    ft = (ft_all.loc[ft_all.groupby("dataset")["dev_mcc_mean"].idxmax()]
          [["dataset", "model_type", "n", "dev_mcc_mean", "test_mcc_mean",
            "test_mcc_sd"]]
          .rename(columns={"model_type": "best_lora_variant",
                           "n": "lora_n", "dev_mcc_mean": "lora_dev_mcc_mean",
                           "test_mcc_mean": "lora_test_mcc_mean",
                           "test_mcc_sd": "lora_test_mcc_sd"}))
    both = best.rename(columns={"gap_vs_paper": "frozen_gap_vs_paper",
                                "pct_of_paper": "frozen_pct_of_paper"})
    both = both.merge(ft, on="dataset", how="left")
    both["lora_mcc_x100"] = both["lora_test_mcc_mean"] * 100
    both["lora_gap_vs_paper"] = both["lora_mcc_x100"] - both["paper_mcc_x100"]
    both["lora_pct_of_paper"] = 100 * both["lora_mcc_x100"] / both["paper_mcc_x100"]
    both.to_csv(os.path.join(R, "best_vs_paper_with_finetune.csv"), index=False)
    print(f"best_vs_paper_with_finetune.csv {len(both):2d} rows")

print(f"history_table.csv            {len(hist):5d} rows")
print(f"history_table_aggregated.csv {len(agg):5d} rows")
print(f"best_frozen_vs_paper.csv     {len(best):5d} rows")
