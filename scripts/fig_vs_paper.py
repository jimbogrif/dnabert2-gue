"""Frozen embeddings against the paper's fine-tuned scores, per dataset.

A dumbbell per GUE dataset: the filled dot is this work's best frozen config
(chosen on dev, test reported once), the open dot is DNABERT-2's fine-tuned
score from the paper's Table 6. Datasets are grouped by task and ordered by the
paper score within each group, so the shortfalls line up as a ladder.

Reads results/best_frozen_vs_paper.csv, which already carries both scores plus
``gap_vs_paper`` and ``pct_of_paper``; every number in the title and the two
callouts is recomputed from it rather than hard-coded.

Usage:
    python scripts/fig_vs_paper.py [RESULTS_DIR] [OUTDIR]
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BLOGFIG_HOME = os.environ.get("BLOGFIG_HOME", "/home/jim/code/blogfig-brand")
sys.path.insert(0, BLOGFIG_HOME)
import blogfig as bf  # noqa: E402

NAME = "vs_paper"
TASKS = ["CPD", "PD", "TF-H", "TF-M", "EMP", "SSP"]   # display order, top to bottom
# bf.CATEGORY_LABELS covers the five task families the frozen sweep benchmarked
# and the palette is only colour-checked to five, so SSP -- a single dataset
# from a later arm -- takes NEUTRAL ink rather than a sixth, wrapped category
# colour that would silently duplicate one of the five.
SSP_READOUTS = ("mean", "cls")     # the readouts comparable to the other arms

# Spelled out, because this figure sits near the top of the results and is where
# a reader first meets the abbreviations. Wording follows the paper's own.
TASK_LABELS = {
    "CPD":  "Core Promoter Detection (CPD)",
    "PD":   "Promoter Detection (PD)",
    "TF-H": "Transcription Factor Prediction, Human (TF-H)",
    "TF-M": "Transcription Factor Prediction, Mouse (TF-M)",
    "EMP":  "Epigenetic Marks Prediction (EMP)",
    "SSP":  "Splice Site Prediction (SSP)",
}


def add_ssp(d, results_dir):
    """Append the splice row, if both halves of it are on disk.

    The frozen side comes from the positional arm rather than the layer sweep,
    so it is assembled here instead of in ``best_frozen_vs_paper.csv``. Only the
    mean/[CLS] readouts are eligible: the windowed readouts are a 6,144-d
    featurization no other dataset in this figure was given, and putting one on
    the same axis would compare unlike things. Best-on-dev, matching the
    selection rule used for every other row.
    """
    ps_path = results_dir / "positional_summary.csv"
    if not ps_path.exists():
        print("  (no positional_summary.csv -- SSP row skipped)")
        return d
    ps = pd.read_csv(ps_path)
    ps = ps[(ps["task"] == "SSP") & (ps["readout"].isin(SSP_READOUTS))]
    paper = pd.read_csv(results_dir / "paper_table6_dnabert2.csv")
    paper = paper[paper["task"] == "SSP"]
    if ps.empty or paper.empty:
        print("  (no SSP frozen or paper score -- SSP row skipped)")
        return d
    best = ps.loc[ps["dev_mcc"].idxmax()]
    pm = float(paper.iloc[0]["paper_mcc_x100"])
    fm = float(best["test_mcc"]) * 100.0
    row = {"task": "SSP", "dataset": best["dataset"], "pool": best["readout"],
           "layer": best["layer"], "paper_mcc_x100": pm, "frozen_mcc_x100": fm,
           "gap_vs_paper": fm - pm, "pct_of_paper": fm / pm * 100.0}
    return pd.concat([d, pd.DataFrame([row])], ignore_index=True)


def load(results_dir):
    d = pd.read_csv(results_dir / "best_frozen_vs_paper.csv")
    d = add_ssp(d, results_dir)
    # 'prom__prom_core_tata' -> 'prom_core_tata'; 'mouse__3' -> '3'
    d["label"] = d["dataset"].str.split("__", n=1).str[-1]
    d["task"] = pd.Categorical(d["task"], categories=TASKS, ordered=True)
    # y grows upward, so lay the groups out in reverse display order (EMP first,
    # lowest) to put CPD at the top; within a group the lowest paper score goes
    # first so the highest ends up at the group's top edge.
    return d.sort_values(["task", "paper_mcc_x100"], ascending=[False, True])


def draw(d):
    bf.use()
    N = bf.NEUTRAL
    fig, ax = plt.subplots(figsize=(8.8, 9.2))

    # One y slot per dataset, with a blank slot between task groups.
    y, ticks, labels, group_top = 0.0, [], [], {}
    for task, g in d.groupby("task", observed=True, sort=False):
        for _, r in g.iterrows():
            ticks.append(y)
            labels.append(r["label"])
            d.loc[r.name, "_y"] = y
            y += 1.0
        group_top[task] = y - 0.5
        y += 1.0

    for task, g in d.groupby("task", observed=True, sort=False):
        c = bf.CATEGORY_LABELS.get(task, N["ink"])
        ax.hlines(g["_y"], g["frozen_mcc_x100"], g["paper_mcc_x100"],
                  color=N["axis"], lw=1.6, zorder=1)
        ax.scatter(g["frozen_mcc_x100"], g["_y"], s=58, color=c,
                   edgecolor=c, lw=1.2, zorder=3)
        ax.scatter(g["paper_mcc_x100"], g["_y"], s=58, facecolor=N["surface"],
                   edgecolor=N["ink"], lw=1.3, zorder=3)
        ax.text(21.0, group_top[task], TASK_LABELS.get(task, task), ha="left",
                va="center", fontsize=10.5, weight="bold", color=c)
        ax.axhline(group_top[task] + 0.5, color=N["grid"], lw=0.9, zorder=0)

    # --- the two rows worth naming, both located from the data -------------
    worst = d.loc[d["gap_vs_paper"].idxmin()]
    gap = f"{worst['gap_vs_paper']:.1f}".replace("-", "\u2212")   # typographic minus
    # Label outside whichever end of the dumbbell has room: once SSP is the worst
    # row its paper dot sits near 85, so a right-hand label would run off the axis.
    if worst["paper_mcc_x100"] > 78.0:
        xy, ha = (worst["frozen_mcc_x100"] - 1.5, worst["_y"]), "right"
    else:
        xy, ha = (worst["paper_mcc_x100"] + 1.5, worst["_y"]), "left"
    ax.annotate(f"largest shortfall, {gap}", xy=xy, ha=ha, va="center",
                fontsize=9.5, color=N["muted"])
    ahead = d[d["gap_vs_paper"] > 0]
    for _, r in ahead.iterrows():
        ax.annotate("only dataset where frozen beats fine-tuned"
                    if len(ahead) == 1 else "frozen beats fine-tuned",
                    xy=(r["frozen_mcc_x100"] + 1.5, r["_y"]),
                    ha="left", va="center", fontsize=9.5, color=N["muted"])

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(-1.0, y - 0.5)
    ax.set_xlim(20, 100)
    ax.set_xlabel("test MCC $\\times$ 100")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=8, color=N["muted"],
                   label="frozen embeddings, best config (this work)"),
        plt.Line2D([], [], ls="", marker="o", ms=8, markerfacecolor=N["surface"],
                   markeredgecolor=N["ink"], color=N["ink"],
                   label="DNABERT-2 fine-tuned (paper Table 6)"),
    ]
    # Below the axes, not inside: the bottom-right corner now holds the SSP row.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.045),
              ncol=2, frameon=False, fontsize=10)

    # A single mean hides the point once SSP is in: it sits ~20 points below the
    # next-worst dataset, so the title reports it separately rather than letting
    # one outlier be averaged into a reassuring headline.
    pct = d["pct_of_paper"]
    rest = d.loc[d["task"] != "SSP", "pct_of_paper"]
    ssp = d.loc[d["task"] == "SSP", "pct_of_paper"]
    if len(ssp):
        line2 = (f"splice site prediction is {ssp.iloc[0]:.1f}%; the other "
                 f"{len(rest)} datasets span {rest.min():.1f}\u2013{rest.max():.1f}%")
        line1 = (f"Frozen DNABERT-2 embeddings reach {pct.mean():.1f}% of the "
                 "fine-tuned score on average \u2014 but not on splice")
    else:
        line1 = (f"Frozen DNABERT-2 embeddings reach {pct.mean():.1f}% of the "
                 "fine-tuned score on average")
        line2 = (f"range {pct.min():.1f}\u2013{pct.max():.1f}%; best config per "
                 "dataset chosen on dev, test reported once")
    fig.suptitle(f"{line1}\n{line2}", fontsize=12.5, x=0.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures")
    d = load(results_dir)
    fig = draw(d)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(outdir / f"{NAME}.{ext}")
    print(f"wrote {outdir}/{NAME}.svg and .png  ({len(d)} datasets)")


if __name__ == "__main__":
    main()
