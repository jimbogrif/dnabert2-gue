"""Bag-of-tokens baselines: the floor the embedding arms are measured against.

Two featurizations, both position-invariant counts over a fixed vocabulary, both
fed through the *identical* downstream protocol as the frozen-embedding arms so
that only the featurization differs:

  - ``kmer{3,4,5,6}`` -- 4**k k-mer frequencies. Asks "does DNABERT-2 beat base
    composition?", the floor the project's own ground rule calls for.
  - ``bpe`` -- frequencies over DNABERT-2's own 4,096-token BPE vocabulary,
    counted with the tokenizer and no forward pass. Asks the sharper question,
    "does the transformer beat its own tokenizer?" Note 4**6 == 4096 exactly, so
    a 6-mer bag already has the same cardinality as the model's whole vocabulary;
    the BPE bag matches its segmentation too.

Design notes:

  - **Frequencies, not raw counts.** 10 of the 26 datasets (all of EMP) are
    variable length, 290-500 bp, so raw counts would smuggle sequence length in
    as a feature on exactly the task family where the embedding arms are
    weakest. L1-normalising per sequence also aligns with the embedding arm,
    which is mean-pooled and therefore length-normalised in the same sense.
  - **No disk cache.** The embedding arm caches 53 GB because a forward pass
    through 117M parameters is expensive; counting k-mers is not (~0.3 min for
    all 762,523 sequences at k=6). Featurize one dataset, fit, discard. The
    largest single dataset at k=6 is ~1.0 GB dense; all 26 at once would be
    11.6 GB, hence per-dataset.
  - **Dense, not sparse.** At k=6 on 70 bp the matrix is ~98% zeros, but
    ``StandardScaler(with_mean=True)`` requires dense input and switching to
    ``with_mean=False`` would break the identical-protocol claim that makes this
    comparison worth anything.
  - Windows containing a non-ACGT base are dropped and counted; the count is
    reported rather than silently absorbed.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import DATASET_TASK, GUE_ROOT, SPLITS, manifest
from .linear import CLASS_WEIGHTS, CS, fit_lr_grid
from .mlp import (MLP_DIMS, MLP_DROPOUT, MLP_LR, MLP_MAX_EPOCHS, MLP_PATIENCE,
                  MLP_WEIGHT_DECAY, TaperedMLP, evaluate, train_model)

KS = (3, 4, 5, 6)
FEATS = tuple(f"kmer{k}" for k in KS) + ("bpe",)
BPE_VOCAB = 4096          # DNABERT-2's tokenizer; == 4**6

_LUT = np.full(256, -1, np.int8)
for _i, _b in enumerate(b"ACGT"):
    _LUT[_b] = _i


# --------------------------------------------------------------------------
# Featurizers
# --------------------------------------------------------------------------

def kmer_matrix(seqs, k, normalize=True, dtype=np.float32):
    """(n, 4**k) k-mer frequency matrix. Returns (X, n_windows_dropped)."""
    w = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    out = np.zeros((len(seqs), 4 ** k), dtype)
    dropped = 0
    for i, s in enumerate(seqs):
        c = _LUT[np.frombuffer(s.encode(), np.uint8)]
        win = np.lib.stride_tricks.sliding_window_view(c, k)
        ok = (win >= 0).all(1)
        dropped += int((~ok).sum())
        cnt = np.bincount(win[ok].astype(np.int64) @ w, minlength=4 ** k)
        tot = cnt.sum()
        out[i] = cnt / tot if (normalize and tot) else cnt
    return out, dropped


def bpe_matrix(seqs, tok, normalize=True, dtype=np.float32, batch=1024):
    """(n, 4096) frequency matrix over DNABERT-2's BPE vocabulary.

    ``add_special_tokens=False`` so [CLS]/[SEP] -- present in every sequence and
    therefore zero-variance -- do not become features.
    """
    out = np.zeros((len(seqs), BPE_VOCAB), dtype)
    for s0 in range(0, len(seqs), batch):
        enc = tok(seqs[s0:s0 + batch], add_special_tokens=False)["input_ids"]
        for j, ids in enumerate(enc):
            cnt = np.bincount(np.asarray(ids, np.int64), minlength=BPE_VOCAB)
            tot = cnt.sum()
            out[s0 + j] = cnt / tot if (normalize and tot) else cnt
    return out, 0


def load_sequences(dataset, split):
    """(sequences, labels) for one GUE split, located via manifest()."""
    path = {key: p for _, key, p in manifest()}[dataset]
    df = pd.read_csv(os.path.join(path, f"{split}.csv"))
    return df["sequence"].tolist(), df["label"].to_numpy()


def featurize(dataset, feat, tok=None):
    """Build (X, y) for all three splits under one featurization.

    Returns (splits_tuple, n_features, n_dropped) where splits_tuple is
    (X_tr, y_tr, X_dv, y_dv, X_te, y_te), ready for fit_lr_grid.
    """
    if feat == "bpe" and tok is None:
        raise ValueError("feat='bpe' needs a tokenizer")
    parts, dropped = [], 0
    for sp in SPLITS:
        seqs, y = load_sequences(dataset, sp)
        if feat == "bpe":
            X, d = bpe_matrix(seqs, tok)
        else:
            X, d = kmer_matrix(seqs, int(feat[4:]))
        parts += [X, y]
        dropped += d
    return tuple(parts), parts[0].shape[1], dropped


# --------------------------------------------------------------------------
# Heads -- identical protocol to the embedding arms, only the input differs
# --------------------------------------------------------------------------

# Sentinels. LoRA already uses layer=0, so the bag arms take -1; real encoder
# layers are 1-12. 'pool' is meaningless for a bag of counts.
BAG_LAYER = -1
BAG_POOL = "bag"


def grid_search_bag(dataset, feat, tok=None, Cs=CS, class_weights=CLASS_WEIGHTS):
    """The frozen arm's C x class_weight grid, on bag features."""
    splits, n_feat, dropped = featurize(dataset, feat, tok)
    meta = {"dataset": dataset, "task": DATASET_TASK[dataset],
            "pool": BAG_POOL, "layer": BAG_LAYER, "feat": feat,
            "n_features": n_feat, "windows_dropped": dropped}
    return fit_lr_grid(splits, meta, Cs, class_weights)


def run_bag_mlp(dataset, feat, seed, tok=None, batch_size=32, device=None,
                debug=False):
    """The frozen arm's TaperedMLP on bag features; returns (row, history).

    Borrows mlp.train_model / evaluate / TaperedMLP and the MLP_*
    constants unchanged -- run_mlp itself cannot be reused because its
    (dataset, pool, layer) signature loads from the .npy cache, and the bag
    features are deliberately never written to disk.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    (X_tr, y_tr, X_dv, y_dv, X_te, y_te), n_feat, dropped = featurize(dataset, feat, tok)
    scaler = StandardScaler().fit(X_tr)          # train-only, as in prep_data
    ten = lambda X, y: (torch.from_numpy(scaler.transform(X)).float(),
                        torch.from_numpy(y).long())
    X_tr, y_tr = ten(X_tr, y_tr); X_dv, y_dv = ten(X_dv, y_dv); X_te, y_te = ten(X_te, y_te)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    model = TaperedMLP(X_tr.shape[1], int(len(torch.unique(y_tr))),
                       dims=MLP_DIMS, dropout=MLP_DROPOUT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)

    model, best_epoch, history = train_model(
        model, loader, X_dv, y_dv, criterion, optimizer, device,
        max_epochs=MLP_MAX_EPOCHS, patience=MLP_PATIENCE, debug=debug)
    _, dev_mcc = evaluate(model, X_dv, y_dv, criterion, device)
    _, test_mcc = evaluate(model, X_te, y_te, criterion, device)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "dataset": dataset, "task": DATASET_TASK[dataset], "feat": feat,
        "pool": BAG_POOL, "layer": BAG_LAYER, "seed": seed,
        "n_features": n_feat, "windows_dropped": dropped,
        "batch_size": batch_size, "best_epoch": best_epoch,
        "epochs_run": len(history),
        "early_stopped": len(history) < MLP_MAX_EPOCHS,
        "dev_mcc": dev_mcc, "test_mcc": test_mcc,
        "hidden_dims": "x".join(str(x) for x in MLP_DIMS),
        "dropout": MLP_DROPOUT, "lr": MLP_LR,
        "weight_decay": MLP_WEIGHT_DECAY, "optimizer": "AdamW",
        "max_epochs": MLP_MAX_EPOCHS, "patience": MLP_PATIENCE,
    }, history
