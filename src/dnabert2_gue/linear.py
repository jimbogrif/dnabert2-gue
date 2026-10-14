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


def grid_search_lr(dataset, pool, layer, d=EMBED_DIR, Cs=CS,
                   class_weights=CLASS_WEIGHTS):
    """Fit the C x class_weight grid; dev selects, test is scored once."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import matthews_corrcoef

    X_tr, y_tr = load_split(dataset, "train", layer, pool, d)
    X_dv, y_dv = load_split(dataset, "dev", layer, pool, d)
    X_te, y_te = load_split(dataset, "test", layer, pool, d)

    rows, best_mcc, best_clf, best_i = [], -np.inf, None, None
    for C, cw in itertools.product(Cs, class_weights):   # ascending C: ties keep simpler model
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, class_weight=cw, max_iter=5000))
        clf.fit(X_tr, y_tr)
        mcc = matthews_corrcoef(y_dv, clf.predict(X_dv))
        rows.append({
            "dataset": dataset, "task": DATASET_TASK[dataset], "pool": pool,
            "layer": layer, "C": C,
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
    for i, r in enumerate(rows):
        r.setdefault("selected", False)
    return rows
