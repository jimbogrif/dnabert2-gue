"""Embed the 26 in-scope GUE datasets at every encoder layer.

Writes layer_embed/{dataset}__{split}__L{01..12}__{mean,cls}.npy (float32) and
{dataset}__{split}__y.npy. Arrays are filled through np.lib.format.open_memmap
so peak RAM stays flat regardless of split size. Resumes on restart.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

import dnabert2_gue as SL

BATCH = 64
MAX_LENGTH = 512

if not torch.cuda.is_available():
    sys.exit("02_embed_all.py needs a CUDA GPU (>=6 GB VRAM): embedding 762,523 "
             "sequences on CPU is a multi-day job. Everything downstream of "
             "layer_embed/ is CPU-only.")

os.makedirs(SL.EMBED_DIR, exist_ok=True)
tok, model = SL.load_model()

tokstats = []
t_all = time.time()
for task, dataset, path in SL.manifest():
    for split in SL.SPLITS:
        targets = [SL.embed_path(dataset, split, L, p)
                   for L in range(1, SL.N_LAYERS + 1) for p in SL.POOLS]
        if all(os.path.exists(f) for f in targets) and \
           os.path.exists(SL.label_path(dataset, split)):
            print(f"skip {dataset} {split}", flush=True)
            continue

        df = pd.read_csv(os.path.join(path, f"{split}.csv"))
        assert list(df.columns) == ["sequence", "label"], \
            f"{dataset}/{split} unexpected columns {list(df.columns)}"
        seqs = df["sequence"].tolist()
        n = len(seqs)
        np.save(SL.label_path(dataset, split), df["label"].values)

        # one memmap per (layer, pool); written batch-slice by batch-slice
        mm = {(L, p): np.lib.format.open_memmap(
                  SL.embed_path(dataset, split, L, p), mode="w+",
                  dtype=np.float32, shape=(n, SL.HIDDEN))
              for L in range(1, SL.N_LAYERS + 1) for p in SL.POOLS}

        t0, ntok = time.time(), []
        with torch.inference_mode():
            for i in range(0, n, BATCH):
                means, clss, nt = SL.embed_batch_layers(
                    seqs[i:i + BATCH], tok, model, max_length=MAX_LENGTH)
                ntok.append(nt)
                j = i + means.shape[1]
                for L in range(1, SL.N_LAYERS + 1):
                    mm[(L, "mean")][i:j] = means[L - 1].numpy()
                    mm[(L, "cls")][i:j] = clss[L - 1].numpy()
        for v in mm.values():
            v.flush()
        del mm

        ntok = np.concatenate(ntok)
        dt = time.time() - t0
        tokstats.append({"task": task, "dataset": dataset, "split": split,
                         "n": n, "tok_med": float(np.median(ntok)),
                         "tok_max": int(ntok.max()),
                         "truncated": int((ntok >= MAX_LENGTH).sum()),
                         "seconds": round(dt, 1),
                         "seq_per_s": round(n / dt, 1)})
        print(f"{dataset:24s} {split:5s} n={n:6d} tok_max={ntok.max():4d} "
              f"{dt:6.1f}s {n / dt:7.1f} seq/s", flush=True)

pd.DataFrame(tokstats).to_csv(
    os.path.join(SL.RESULTS_DIR, "embed_manifest.csv"), index=False)
print(f"TOTAL {time.time() - t_all:.1f}s", flush=True)
