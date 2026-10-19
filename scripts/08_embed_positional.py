"""Embed the positional-readout datasets at every layer, three readouts.

Writes layer_embed/{dataset}__{split}__L{01..12}__{mean,cls,winW}.npy (float32)
and {dataset}__{split}__y.npy, through np.lib.format.open_memmap so peak RAM
stays flat. Resumes on restart.

Only three readouts are stored. The other two in POSITIONAL_READOUTS are
derived from the win cache at probe time: winW_best picks one 768-d window on
dev, winW shuf permutes the W blocks per sequence. Caching those would add a
second 41 GB for no information.

Usage:
    python scripts/08_embed_positional.py                  # all three datasets
    python scripts/08_embed_positional.py --limit 200 --datasets splice__reconstructed
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

import dnabert2_gue as SL

BATCH = 64
MAX_LENGTH = 512

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="*", default=list(SL.POSITIONAL_DATASETS))
ap.add_argument("--windows", type=int, default=SL.POSITIONAL_N_WINDOWS)
ap.add_argument("--limit", type=int, default=None,
                help="first N rows of each split; smoke runs only")
ap.add_argument("--batch", type=int, default=BATCH)
ap.add_argument("--suffix", default="",
                help="appended to the dataset key, so smoke output cannot "
                     "collide with the real cache")
args = ap.parse_args()

WIN = f"win{args.windows}"

# A --limit run writes short arrays; letting those land beside the real
# cache would leave a (limit, d) file where the full split is expected and
# every later run would "skip" it. Send them to their own directory, which
# is also a single thing to delete afterwards.
OUTDIR = os.path.join(SL.EMBED_DIR, "_smoke") if args.limit else SL.EMBED_DIR
if args.limit:
    print(f"--limit {args.limit}: writing to {OUTDIR}", flush=True)

if not torch.cuda.is_available():
    sys.exit("08_embed_positional.py needs a CUDA GPU; everything downstream "
             "of layer_embed/ is CPU-only.")

os.makedirs(OUTDIR, exist_ok=True)
tok, model = SL.load_model()

stats = []
t_all = time.time()
for dataset in args.datasets:
    path = SL.dataset_dir(dataset)
    key = dataset + args.suffix
    for split in SL.SPLITS:
        # Write only the readouts that are missing. prom__prom_300_all and
        # tf__0 already have mean/cls from the 26-dataset run; recomputing
        # them is cheap but overwriting a published cache is not something to
        # do silently, and keeping them lets this run check itself against
        # them (see max|new-cached| below).
        readouts = [p for p in ("mean", "cls", WIN)
                    if not all(os.path.exists(
                        SL.embed_path(key, split, L, p, OUTDIR))
                        for L in range(1, SL.N_LAYERS + 1))]
        if not readouts and os.path.exists(SL.label_path(key, split, OUTDIR)):
            print(f"skip {key} {split}", flush=True)
            continue
        # Where mean/cls are already cached from the 26-dataset run, this run
        # recomputes them anyway (same forward pass) and compares, which is a
        # free end-to-end check that the window-pooling path did not disturb
        # the existing one. Measured exact: max|diff| = 0.0 over 12 layers.
        #
        # Only valid at full length. Batches are formed by position, so a
        # --limit run ends in a ragged batch whose padding width differs from
        # the original run's, and different padding changes the reduction
        # order: that alone moved mean pooling by 2e-4 on 8 rows.
        check = [] if args.limit else [
            p for p in ("mean", "cls")
            if all(os.path.exists(SL.embed_path(dataset, split, L, p))
                   for L in range(1, SL.N_LAYERS + 1))]

        df = pd.read_csv(os.path.join(path, f"{split}.csv"))
        assert list(df.columns) == ["sequence", "label"], \
            f"{dataset}/{split} unexpected columns {list(df.columns)}"
        if args.limit:
            df = df.iloc[:args.limit]
        seqs = df["sequence"].tolist()
        n = len(seqs)
        seq_len = df["sequence"].str.len()
        assert seq_len.nunique() == 1, \
            (f"{dataset}/{split} is variable length ({seq_len.min()}-"
             f"{seq_len.max()} bp); uniform bp windows assume fixed length")
        if not os.path.exists(SL.label_path(key, split, OUTDIR)):
            np.save(SL.label_path(key, split, OUTDIR), df["label"].values)

        dims = {"mean": SL.HIDDEN, "cls": SL.HIDDEN,
                WIN: args.windows * SL.HIDDEN}
        mm = {(L, p): np.lib.format.open_memmap(
                  SL.embed_path(key, split, L, p, OUTDIR), mode="w+",
                  dtype=np.float32, shape=(n, dims[p]))
              for L in range(1, SL.N_LAYERS + 1) for p in readouts}

        t0, ntok, n_empty, drift = time.time(), [], 0, 0.0
        with torch.inference_mode():
            for i in range(0, n, args.batch):
                means, clss, wins, nt, empty = SL.embed_batch_positional(
                    seqs[i:i + args.batch], tok, model, args.windows,
                    max_length=MAX_LENGTH)
                ntok.append(nt)
                n_empty += empty
                j = i + means.shape[1]
                got = {"mean": means, "cls": clss}
                for L in range(1, SL.N_LAYERS + 1):
                    for p in readouts:
                        v = (wins[L - 1].reshape(j - i, -1) if p == WIN
                             else got[p][L - 1])
                        mm[(L, p)][i:j] = v.numpy()
                    # free check against the published 26-dataset cache
                    for p in check:
                        ref = np.load(SL.embed_path(dataset, split, L, p),
                                      mmap_mode="r")[i:j]
                        drift = max(drift, float(np.abs(
                            got[p][L - 1].numpy() - ref).max()))
        for v in mm.values():
            v.flush()
        del mm

        ntok = np.concatenate(ntok)
        dt = time.time() - t0
        assert n_empty == 0, f"{key}/{split}: {n_empty} empty windows"
        if check:
            assert drift < 1e-6, (
                f"{key}/{split}: this run's {check} differs from the cached "
                f"26-dataset embeddings by {drift:.2e}. At matched batching "
                f"these agree exactly, so a non-zero value here means the "
                f"model, tokenizer or batch order changed -- not rounding")
            print(f"    checked {check} against cache: max|diff| {drift:.2e}",
                  flush=True)
        stats.append({"task": SL.DATASET_TASK[dataset], "dataset": key,
                      "split": split, "n": n, "bp": int(seq_len.iloc[0]),
                      "readouts_written": "+".join(readouts),
                      "checked_vs_cache": "+".join(check) or "",
                      "max_abs_drift": drift if check else np.nan,
                      "windows": args.windows,
                      "window_bp": round(seq_len.iloc[0] / args.windows, 2),
                      "tok_med": float(np.median(ntok)),
                      "tok_max": int(ntok.max()),
                      "truncated": int((ntok >= MAX_LENGTH).sum()),
                      "empty_windows": n_empty,
                      "seconds": round(dt, 1), "seq_per_s": round(n / dt, 1)})
        print(f"{key:28s} {split:5s} n={n:6d} tok_max={ntok.max():4d} "
              f"{dt:6.1f}s {n / dt:7.1f} seq/s", flush=True)

if stats:
    out = os.path.join(SL.RESULTS_DIR, "positional_embed_manifest.csv")
    df_new = pd.DataFrame(stats)
    if os.path.exists(out):
        prev = pd.read_csv(out)
        keep = ~prev.set_index(["dataset", "split"]).index.isin(
            df_new.set_index(["dataset", "split"]).index)
        df_new = pd.concat([prev[keep], df_new], ignore_index=True)
    df_new.to_csv(out, index=False)
    print(f"-> {out}", flush=True)
print(f"TOTAL {time.time() - t_all:.1f}s", flush=True)
