"""Per-task probing profile across DNABERT-2's 12 encoder layers.

Reads results/lr_layer_sweep.csv and results/selected_layers.csv and draws one
panel per GUE task family: mean dev MCC against encoder layer for both pooling
strategies, with the per-dataset curves behind them, the two layers carried into
the neural arm shaded, and the final layer marked.

Usage:
    python scripts/fig_layer_profile.py [RESULTS_DIR] [OUTDIR]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BLOGFIG_HOME = Path("/home/jim/code/blogfig-brand")
sys.path.insert(0, str(BLOGFIG_HOME))
import blogfig as bf  # noqa: E402

TASKS = ["CPD", "PD", "TF-H", "TF-M", "EMP"]
POOLS = [("mean", "mean-pooled"), ("cls", "[CLS] token")]
FINAL_LAYER = 12


def load(results_dir):
    sweep = pd.read_csv(results_dir / "lr_layer_sweep.csv")
    sweep = sweep[sweep["selected"]]
    cells = (
        sweep.groupby(["task", "dataset", "pool", "layer"])["dev_mcc"].max().reset_index()
    )
    selected = pd.read_csv(results_dir / "selected_layers.csv")
    bands = {
        t: sorted(g.loc[g["role"].isin(["top1", "top2"]), "layer"])
        for t, g in selected.groupby("task")
    }
    return cells, bands


def draw(cells, bands):
    bf.use()
    colors = {"mean": bf.ACCENT["primary"], "cls": bf.ACCENT["secondary"]}

    fig, axes = plt.subplots(1, len(TASKS), figsize=(11.0, 3.4))
    for ax, task in zip(axes, TASKS):
        sub = cells[cells["task"] == task]
        n_ds = sub["dataset"].nunique()

        for layer in bands[task]:
            ax.axvspan(layer - 0.42, layer + 0.42, color=bf.HIGHLIGHT, alpha=0.6,
                       lw=0, zorder=0)
        ax.axvline(FINAL_LAYER, color=bf.REFERENCE, ls=(0, (4, 3)), lw=1.1, zorder=1)

        for pool, _ in POOLS:
            p = sub[sub["pool"] == pool]
            for _, g in p.groupby("dataset"):
                g = g.sort_values("layer")
                ax.plot(g["layer"], g["dev_mcc"], color=colors[pool], lw=0.7,
                        alpha=0.22, zorder=2)
            mean = p.groupby("layer")["dev_mcc"].mean().sort_index()
            ax.plot(mean.index, mean.values, color=colors[pool], lw=2.0,
                    marker="o", ms=3.6, zorder=3)

        ax.set_title(f"{task}  (n={n_ds})", fontsize=11, pad=6)
        ax.set_xlabel("encoder layer", fontsize=9.5)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.set_xlim(0.3, 12.7)
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel("dev MCC  (logistic regression)", fontsize=9.5)
    axes[0].annotate("↑ higher = better", xy=(0.04, 0.955), xycoords="axes fraction",
                     fontsize=8.5, color=bf.NEUTRAL["muted"], va="top")

    handles = [
        plt.Line2D([], [], color=colors["mean"], lw=2.0, marker="o", ms=3.6,
                   label="mean-pooled"),
        plt.Line2D([], [], color=colors["cls"], lw=2.0, marker="o", ms=3.6,
                   label="[CLS] token"),
        plt.Line2D([], [], color=colors["mean"], lw=0.7, alpha=0.5,
                   label="individual dataset"),
        plt.Rectangle((0, 0), 1, 1, color=bf.HIGHLIGHT, alpha=0.6,
                      label="selected: top-2 layer"),
        plt.Line2D([], [], color=bf.REFERENCE, ls=(0, (4, 3)), lw=1.1,
                   label="layer 12 (final)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Frozen + probe model accuracy peaks in layers 2\u20136; DNABERT-2's final layer "
        "ranks 8th\u201312th of 12 on every task",
        fontsize=12.5, x=0.5, y=1.02,
    )
    fig.tight_layout()
    return fig


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures")
    cells, bands = load(results_dir)
    fig = draw(cells, bands)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(outdir / f"layer_profile.{ext}")
    print(f"wrote {outdir}/layer_profile.svg and .png")


if __name__ == "__main__":
    main()
