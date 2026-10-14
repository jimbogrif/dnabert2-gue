"""On-disk layout of the cached per-layer embeddings."""

import os

import numpy as np

from .config import EMBED_DIR


def embed_path(dataset, split, layer, pool, d=EMBED_DIR):
    return os.path.join(d, f"{dataset}__{split}__L{layer:02d}__{pool}.npy")


def label_path(dataset, split, d=EMBED_DIR):
    return os.path.join(d, f"{dataset}__{split}__y.npy")


def load_split(dataset, split, layer, pool, d=EMBED_DIR):
    X = np.load(embed_path(dataset, split, layer, pool, d))
    y = np.load(label_path(dataset, split, d))
    return X, y
