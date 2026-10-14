"""Logistic-regression probe over the cached embeddings.

The third documented fix relative to the original notebook lives here:

  3. ``class_weight=None`` is persisted as the literal string ``"none"`` so a
     naive ``pd.read_csv`` does not turn it into NaN.
"""

import itertools

import numpy as np

from .config import DATASET_TASK, EMBED_DIR
from .embed import load_split

# --------------------------------------------------------------------------
# Logistic regression grid (notebook's grid_search_lr, widened C range)
# --------------------------------------------------------------------------

CS = np.logspace(-5, 3, 9)            # widened from the notebook's (-4, 2, 7)
CLASS_WEIGHTS = [None, "balanced"]
CS_NOTEBOOK = set(np.logspace(-4, 2, 7).round(10))


def fit_lr_grid(splits, meta, Cs=CS, class_weights=CLASS_WEIGHTS):
    """Fit the C x class_weight grid on supplied arrays; dev selects, test once.

    ``splits`` is (X_tr, y_tr, X_dv, y_dv, X_te, y_te); ``meta`` is the dict of
    identifying columns merged into every row (dataset/task/pool/layer for the
    embedding arm, plus the featurization for the bag arm).

    Extracted from ``grid_search_lr`` unchanged so the bag-of-k-mers arm can run
    the identical grid on in-memory features instead of cached .npy — the point
    of that comparison is that only the featurization differs.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import matthews_corrcoef

    X_tr, y_tr, X_dv, y_dv, X_te, y_te = splits

    rows, best_mcc, best_clf, best_i = [], -np.inf, None, None
    for C, cw in itertools.product(Cs, class_weights):   # ascending C: ties keep simpler model
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, class_weight=cw, max_iter=5000))
        clf.fit(X_tr, y_tr)
        mcc = matthews_corrcoef(y_dv, clf.predict(X_dv))
        rows.append({
            **meta, "C": C,
            # literal "none" -- "None" round-trips through read_csv as NaN
            "class_weight": "none" if cw is None else cw,
            "dev_mcc": mcc, "test_mcc": np.nan,
            "n_iter": int(clf[-1].n_iter_[0]),
            "at_grid_edge": bool(C == Cs[0] or C == Cs[-1]),
            "in_notebook_grid": bool(round(float(C), 10) in CS_NOTEBOOK),
        })
        if mcc > best_mcc:
            best_mcc, best_clf, best_i = mcc, clf, len(rows) - 1

    rows[best_i]["test_mcc"] = matthews_corrcoef(y_te, best_clf.predict(X_te))
    rows[best_i]["selected"] = True
    for r in rows:
        r.setdefault("selected", False)
    return rows


def grid_search_lr(dataset, pool, layer, d=EMBED_DIR, Cs=CS,
                   class_weights=CLASS_WEIGHTS):
    """Fit the C x class_weight grid; dev selects, test is scored once."""
    splits = (*load_split(dataset, "train", layer, pool, d),
              *load_split(dataset, "dev", layer, pool, d),
              *load_split(dataset, "test", layer, pool, d))
    meta = {"dataset": dataset, "task": DATASET_TASK[dataset],
            "pool": pool, "layer": layer}
    return fit_lr_grid(splits, meta, Cs, class_weights)
