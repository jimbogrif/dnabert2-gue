"""End-to-end smoke test: model -> embeddings -> LR probe on one small dataset.

Embeds mouse__3 (1,904 / 239 / 239 rows) at the three TF-M layers named in
results/selected_layers.csv into a temporary directory -- never EMBED_DIR, so
it neither depends on nor pollutes the embedding cache -- then runs
grid_search_lr on one layer/pooling cell and asserts a loose dev-MCC floor.

This is a "the pipeline is wired up" check (install, lock file, weight
download, Triton fix, device auto-detect, lazy imports), not a performance
claim. Runs on CPU too, slowly; target is under two minutes on a GPU.
"""

import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import dnabert2_gue as SL

DATASET, TASK = "mouse__3", "TF-M"
POOL = "mean"
FLOOR = 0.30
BATCH = 64

t0 = time.time()
import torch                                                   # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok, model = SL.load_model()

sel = pd.read_csv(os.path.join(SL.RESULTS_DIR, "selected_layers.csv"))
layers = sorted(sel.loc[sel.task == TASK, "layer"].tolist())
top1 = int(sel.loc[(sel.task == TASK) & (sel.role == "top1"), "layer"].iloc[0])
cat, name = DATASET.split("__", 1)

with tempfile.TemporaryDirectory(prefix="dnabert2_smoke_") as d:
    for split in SL.SPLITS:
        df = pd.read_csv(os.path.join(SL.GUE_ROOT, cat, name, f"{split}.csv"))
        seqs = df["sequence"].tolist()
        means, clss = [], []
        with torch.inference_mode():
            for i in range(0, len(seqs), BATCH):
                m, c, _ = SL.embed_batch_layers(seqs[i:i + BATCH], tok, model)
                means.append(m)
                clss.append(c)
        means, clss = torch.cat(means, dim=1), torch.cat(clss, dim=1)
        np.save(SL.label_path(DATASET, split, d), df["label"].values)
        for L in layers:
            np.save(SL.embed_path(DATASET, split, L, "mean", d), means[L - 1].numpy())
            np.save(SL.embed_path(DATASET, split, L, "cls", d), clss[L - 1].numpy())
        print(f"embedded {DATASET} {split:5s} n={len(seqs)} layers={layers}", flush=True)

    rows = SL.grid_search_lr(DATASET, POOL, top1, d=d)

best = next(r for r in rows if r["selected"])
wall = time.time() - t0
print(f"device={device}  layer=L{top1:02d} pool={POOL}  C={best['C']:g} "
      f"class_weight={best['class_weight']}  dev_mcc={best['dev_mcc']:.4f} "
      f"test_mcc={best['test_mcc']:.4f}  wall={wall:.1f}s", flush=True)

if best["dev_mcc"] <= FLOOR:
    sys.exit(f"FAIL: dev MCC {best['dev_mcc']:.4f} <= {FLOOR}")
print("PASS", flush=True)
