"""Does a 256-unit MLP head beat logistic regression on frozen embeddings?

Panel a pairs the two heads cell by cell against the identity line; panel b
shows the per-cell difference by task with the per-task median. A "cell" is one
(dataset x pooling x layer) combination, so both arms are compared on exactly
the same frozen features.

Reads results/history_table_aggregated.csv and uses its ``LR`` and ``MLP`` rows,
which are already aggregated over seeds and share the (task, dataset, pool,
layer) key. Every number in the title is recomputed from the join.

Usage:
    python scripts/fig_lr_vs_mlp.py [RESULTS_DIR] [OUTDIR]
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLOGFIG_HOME = os.environ.get("BLOGFIG_HOME", "/home/jim/code/blogfig-brand")
sys.path.insert(0, BLOGFIG_HOME)
import blogfig as bf  # noqa: E402

NAME = "lr_vs_mlp"
TASKS = ["CPD", "PD", "TF-H", "TF-M", "EMP"]
# Colour alone does not separate overlapping point clouds; pair it with shape.
MARKERS = {"CPD": "o", "PD": "s", "TF-H": "^", "TF-M": "D", "EMP": "v"}
KEY = ["task", "dataset", "pool", "layer"]


def load(results_dir):
    h = pd.read_csv(results_dir / "history_table_aggregated.csv")
    lr = h[h["model_type"] == "LR"].set_index(KEY)["test_mcc_mean"].rename("lr")
    mlp = h[h["model_type"] == "MLP"].set_index(KEY)["test_mcc_mean"].rename("mlp")
    j = pd.concat([lr, mlp], axis=1).dropna().reset_index()
    j["delta"] = j["mlp"] - j["lr"]
    j["task"] = pd.Categorical(j["task"], categories=TASKS, ordered=True)
    return j.sort_values("task")


def draw(j):
    bf.use()
    N, REF = bf.NEUTRAL, bf.REFERENCE
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.6))

    # ---------------- a: paired scatter against identity -------------------
    lo = min(j["lr"].min(), j["mlp"].min()) - 0.03
    hi = max(j["lr"].max(), j["mlp"].max()) + 0.03
    axa.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), lw=1.2, color=REF, zorder=1)
    for task, g in j.groupby("task", observed=True, sort=False):
        c, m = bf.CATEGORY_LABELS[task], MARKERS[task]
        for pool, gg in g.groupby("pool"):
            filled = (pool == "mean")
            axa.scatter(gg["lr"], gg["mlp"], s=34, marker=m,
                        facecolor=c if filled else "none",
                        edgecolor=c, lw=1.1, alpha=0.85, zorder=3)
    axa.text(0.03, 0.95, "MLP better", transform=axa.transAxes, ha="left",
             va="top", fontsize=9.5, color=N["muted"])
    axa.text(0.97, 0.06, "LR better", transform=axa.transAxes, ha="right",
             va="bottom", fontsize=9.5, color=N["muted"])
    axa.set_xlim(lo, hi)
    axa.set_ylim(lo, hi)
    axa.set_aspect("equal")
    axa.set_xlabel("logistic regression \u2014 test MCC")
    axa.set_ylabel("MLP \u2014 test MCC")
    axa.set_title("Each point is one dataset $\\times$ pooling $\\times$ layer cell",
                  fontsize=11)

    # ---------------- b: per-cell difference by task -----------------------
    rng = np.random.default_rng(0)          # jitter only; no data is resampled
    axb.axhline(0.0, ls=(0, (4, 3)), lw=1.2, color=REF, zorder=1)
    xt, xl = [], []
    for i, (task, g) in enumerate(j.groupby("task", observed=True, sort=False)):
        c, m = bf.CATEGORY_LABELS[task], MARKERS[task]
        x = i + rng.uniform(-0.22, 0.22, len(g))
        axb.scatter(x, g["delta"], s=30, marker=m, color=c, alpha=0.8,
                    edgecolor="none", zorder=3)
        axb.hlines(g["delta"].median(), i - 0.34, i + 0.34, color=N["ink"],
                   lw=2.4, zorder=4)
        xt.append(i)
        xl.append(f"{task}\n(n={len(g)})")
    axb.set_xticks(xt)
    axb.set_xticklabels(xl)
    axb.set_xlim(-0.6, len(xt) - 0.4)
    axb.set_ylabel("test MCC:  MLP $-$ LR")
    axb.set_title("Black tick is the per-task median", fontsize=11)
    axb.grid(axis="x", visible=False)

    for ax, letter in ((axa, "a"), (axb, "b")):
        ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=14,
                weight="bold", color=N["ink"], ha="left", va="top")

    handles = [plt.Line2D([], [], ls="", marker=MARKERS[t], ms=7,
                          color=bf.CATEGORY_LABELS[t], label=t) for t in TASKS]
    handles += [
        plt.Line2D([], [], ls="", marker="o", ms=7, color=N["ink"],
                   label="mean-pooled (filled)"),
        plt.Line2D([], [], ls="", marker="o", ms=7, markerfacecolor="none",
                   markeredgecolor=N["ink"], color=N["ink"],
                   label="[CLS] token (open)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.045))

    wins = int((j["delta"] > 0).sum())
    med = f"{j['delta'].median():+.3f}".replace("-", "\u2212")   # typographic minus
    fig.suptitle(
        "A 256-unit MLP head gives no reliable gain over logistic regression on "
        f"frozen embeddings\nmedian {med} MCC over "
        f"{len(j)} cells; the MLP wins {wins} of them",
        fontsize=12.5, x=0.5, y=1.0,
    )
    # tight_layout already reserves room for the suptitle; lowering rect's top
    # as well would reserve it twice and open a band above the panels.
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures")
    j = load(results_dir)
    fig = draw(j)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(outdir / f"{NAME}.{ext}")
    print(f"wrote {outdir}/{NAME}.svg and .png  ({len(j)} cells)")


if __name__ == "__main__":
    main()
