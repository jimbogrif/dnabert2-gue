"""The whole ladder on one axis: bag floor, frozen probe, LoRA, paper.

One row per GUE dataset, four marks on it -- the bag-of-tokens floor, the best
frozen-embedding probe, the LoRA fine-tune, and DNABERT-2's published
fine-tuned score. Rows are grouped by task and ordered by the paper score
within a group, matching ``fig_vs_paper.py``.

Colour encodes the rung rather than the task here: the task is already carried
by the grouping and its label, and the comparison the figure exists to make is
between arms. The paper reference keeps the open-circle glyph it has in
``fig_vs_paper.py`` so the two figures read together.

Inputs, all under ``RESULTS_DIR``:
    best_vs_paper_with_finetune.csv   frozen, LoRA and paper for 26 datasets
    bag_best_per_dataset.csv          bag arm, 26 datasets x 4 featurizations
    bag_results_ssp.csv               bag arm for splice (raw, per seed)
    positional_summary.csv            frozen arm for splice
    lora_results.csv                  LoRA arm for splice
    paper_table6_dnabert2.csv         paper score for splice

Usage:
    python scripts/fig_four_arms.py [RESULTS_DIR] [OUTDIR]
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BLOGFIG_HOME = os.environ.get("BLOGFIG_HOME", "/home/jim/code/blogfig-brand")
sys.path.insert(0, BLOGFIG_HOME)
import blogfig as bf  # noqa: E402

NAME = "four_arms"
TASKS = ["CPD", "PD", "TF-H", "TF-M", "EMP", "SSP"]
TASK_LABELS = {
    "CPD":  "Core Promoter Detection (CPD)",
    "PD":   "Promoter Detection (PD)",
    "TF-H": "Transcription Factor Prediction, Human (TF-H)",
    "TF-M": "Transcription Factor Prediction, Mouse (TF-M)",
    "EMP":  "Epigenetic Marks Prediction (EMP)",
    "SSP":  "Splice Site Prediction (SSP)",
}
# Rung order is the figure's argument, so the palette is assigned in that order.
ARMS = [
    ("bag",    "bag of tokens (no gLM)",        bf.CATEGORY[0], "o"),
    ("frozen", "frozen embeddings + probe",     bf.CATEGORY[1], "s"),
    ("lora",   "LoRA fine-tune (0.75% of weights)", bf.CATEGORY[2], "D"),
]
PAPER_LABEL = "DNABERT-2 fine-tuned (paper Table 6)"


def _best_on_dev(df, by, dev, test):
    """Row with the highest dev score per group, reported on test."""
    return df.loc[df.groupby(by)[dev].idxmax()]


def load(results_dir):
    core = pd.read_csv(results_dir / "best_vs_paper_with_finetune.csv")
    d = core[["task", "dataset"]].copy()
    d["frozen"] = core["frozen_mcc_x100"] / 100.0
    d["lora"] = core["lora_mcc_x100"] / 100.0
    d["paper"] = core["paper_mcc_x100"] / 100.0

    bag = pd.read_csv(results_dir / "bag_best_per_dataset.csv")
    bag = _best_on_dev(bag, "dataset", "dev_mcc_mean", "test_mcc_mean")
    d = d.merge(bag[["dataset", "test_mcc_mean"]].rename(
        columns={"test_mcc_mean": "bag"}), on="dataset", how="left")
    assert d["bag"].notna().all(), "a dataset has no bag-arm row"

    d = pd.concat([d, _ssp_row(results_dir)], ignore_index=True)
    d["task"] = pd.Categorical(d["task"], categories=TASKS, ordered=True)
    # y grows upward: lay groups out in reverse so CPD ends up on top, and put
    # the lowest paper score first so the highest sits at each group's top edge.
    return d.sort_values(["task", "paper"], ascending=[False, True])


def _ssp_row(results_dir):
    """Splice, assembled from four separate arms' tables.

    The splice run post-dates the layer sweep, so none of its numbers are in
    ``best_vs_paper_with_finetune.csv``. The bag table here is raw per-seed
    rows, unlike ``bag_best_per_dataset.csv``, so it is aggregated first.
    Frozen is restricted to the mean/[CLS] readouts: the windowed readouts are
    a 6,144-d featurization no other dataset in this figure was given.
    """
    need = ["bag_results_ssp.csv", "positional_summary.csv", "lora_results.csv",
            "paper_table6_dnabert2.csv"]
    if not all((results_dir / f).exists() for f in need):
        print("  (splice inputs incomplete -- SSP row skipped)")
        return pd.DataFrame()

    b = pd.read_csv(results_dir / "bag_results_ssp.csv")
    b = (b.groupby(["feat", "head_type"], as_index=False)
           .agg(dev=("dev_mcc", "mean"), test=("test_mcc", "mean")))
    bag = b.loc[b["dev"].idxmax(), "test"]

    p = pd.read_csv(results_dir / "positional_summary.csv")
    p = p[(p["task"] == "SSP") & (p["readout"].isin(["mean", "cls"]))]
    frozen = p.loc[p["dev_mcc"].idxmax(), "test_mcc"]

    lo = pd.read_csv(results_dir / "lora_results.csv")
    lo = lo[lo["dataset"].str.contains("splice")]

    pt = pd.read_csv(results_dir / "paper_table6_dnabert2.csv")
    pt = pt[pt["task"] == "SSP"]
    if p.empty or lo.empty or pt.empty:
        print("  (no splice frozen/LoRA/paper score -- SSP row skipped)")
        return pd.DataFrame()

    return pd.DataFrame([{
        "task": "SSP", "dataset": "splice__reconstructed", "bag": float(bag),
        "frozen": float(frozen), "lora": float(lo["test_mcc"].mean()),
        "paper": float(pt.iloc[0]["paper_mcc_x100"]) / 100.0,
    }])


def draw(d):
    bf.use()
    N = bf.NEUTRAL
    fig, ax = plt.subplots(figsize=(9.4, 9.6))

    y, ticks, labels, group_top = 0.0, [], [], {}
    for task, g in d.groupby("task", observed=True, sort=False):
        for idx in g.index:
            ticks.append(y)
            labels.append(d.loc[idx, "dataset"].split("__", 1)[-1])
            d.loc[idx, "_y"] = y
            y += 1.0
        group_top[task] = y - 0.5
        y += 1.0

    lo_v = d[["bag", "frozen", "lora", "paper"]].min(axis=1)
    hi_v = d[["bag", "frozen", "lora", "paper"]].max(axis=1)
    ax.hlines(d["_y"], lo_v, hi_v, color=N["axis"], lw=1.4, zorder=1)

    for key, _lab, colour, marker in ARMS:
        ax.scatter(d[key], d["_y"], s=46, marker=marker, color=colour,
                   edgecolor=colour, lw=1.0, zorder=3)
    ax.scatter(d["paper"], d["_y"], s=52, marker="o",
               facecolor=N["surface"], edgecolor=N["ink"], lw=1.3, zorder=4)

    for task in d["task"].cat.categories:
        if task not in group_top:
            continue
        ax.text(0.012, group_top[task], TASK_LABELS.get(task, task), ha="left",
                va="center", fontsize=10.5, weight="bold", color=N["ink"])
        ax.axhline(group_top[task] + 0.5, color=N["grid"], lw=0.9, zorder=0)

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(-1.0, y - 0.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("test MCC")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [plt.Line2D([], [], ls="", marker=mk, ms=8, color=c, label=lab)
               for _k, lab, c, mk in ARMS]
    handles.append(plt.Line2D([], [], ls="", marker="o", ms=8,
                              markerfacecolor=N["surface"],
                              markeredgecolor=N["ink"], color=N["ink"],
                              label=PAPER_LABEL))
    # Clear of the x-label: at two columns the legend is tall enough that the
    # vs_paper offset of -0.045 puts its first row on top of "test MCC".
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.075),
              ncol=2, frameon=False, fontsize=10)

    n = len(d)
    lw_ = int((d["lora"] > d["frozen"]).sum())
    fw = int((d["frozen"] > d["bag"]).sum())
    fig.suptitle(
        "Frozen embeddings do not clear the bag-of-tokens floor; the rung that "
        "touches the encoder does\n"
        f"mean test MCC {d['bag'].mean():.3f} bag / {d['frozen'].mean():.3f} "
        f"frozen / {d['lora'].mean():.3f} LoRA, against the paper's "
        f"{d['paper'].mean():.3f}\n"
        f"LoRA beats frozen on {lw_} of {n} datasets; frozen beats the bag on "
        f"only {fw}",
        fontsize=12.5, x=0.5, y=0.99,
    )
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
    # The joined four-arm table is a deliverable in its own right -- it is the
    # only place all four arms sit side by side -- so it lands beside its inputs.
    table = results_dir / "four_arms.csv"
    d.drop(columns="_y").to_csv(table, index=False)
    print(f"wrote {outdir}/{NAME}.svg and .png, {table}  ({len(d)} datasets)")


if __name__ == "__main__":
    main()
