"""The ladder per GUE category, normalised to full fine-tuning.

One cluster of three bars per GUE category -- the best bag-of-tokens model, the
best frozen-embedding probe and the best LoRA fine-tune, each picked on dev and
reported on test. Every dataset's test MCC is divided by DNABERT-2's published
fully fine-tuned score on that dataset (paper Table 6, = 100), the bar is the
median of those normalised scores over the category, and the individual
datasets are overlaid as points.

Colour: hue identifies the category and lightness identifies the arm (muted =
bag, standard = probe, deep = LoRA). The palette has five hues for six
categories, so splice (a single reconstructed dataset) takes the neutral grey
rather than recycling a hue.

Inputs, all under ``RESULTS_DIR``:
    best_vs_paper_with_finetune.csv   frozen, LoRA and paper for 26 datasets
    bag_best_per_dataset.csv          bag arm, 26 datasets x 4 featurizations
    bag_results_ssp.csv               bag arm for splice (raw, per seed)
    positional_summary.csv            frozen arm for splice
    lora_results.csv                  LoRA arm for splice
    paper_table6_dnabert2.csv         paper score for splice

Usage:
    python scripts/fig_four_arms_norm.py [RESULTS_DIR] [OUTDIR]
"""

import colorsys
import os
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLOGFIG_HOME = os.environ.get("BLOGFIG_HOME", "/home/jim/code/blogfig-brand")
sys.path.insert(0, BLOGFIG_HOME)
import blogfig as bf  # noqa: E402

NAME = "four_arms_norm"
TASKS = ["CPD", "PD", "TF-H", "TF-M", "EMP", "SSP"]
TASK_LABELS = {
    "CPD":  "Core promoter\ndetection",
    "PD":   "Promoter\ndetection",
    "TF-H": "TF prediction,\nhuman",
    "TF-M": "TF prediction,\nmouse",
    "EMP":  "Epigenetic marks\nprediction",
    "SSP":  "Splice site\nprediction",
}
ARMS = [
    ("bag",    "k-mer frequency (no gLM)"),
    ("frozen", "frozen gLM embeddings + probe"),
    ("lora",   "LoRA fine-tuned gLM"),
]
# Per-arm fill opacity and edge style, so the arms differ by more than lightness.
ALPHAS = [0.5, 0.75, 1.0]
EDGES = [(1.4, (0, (3, 2))), (1.4, "-"), (3.0, "-")]  # (linewidth, linestyle)
EDGES = [(1, "-"), (1, "-"), (1, "-")]  # (linewidth, linestyle)

SSP_BASE = "#22231F"  # near-black: splice is the one category without a hue


def get_variants(hex_color):
    """Muted, standard and deep variants of one hue."""
    rgb = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    muted = colorsys.hls_to_rgb(h, min(1.0, l * 1.3), s * 0.6)
    deep = colorsys.hls_to_rgb(h, l * 0.7, min(1.0, s * 1.1))
    return [mcolors.to_hex(muted), mcolors.to_hex(rgb), mcolors.to_hex(deep)]


def _best_on_dev(df, by, dev):
    """Row with the highest dev score per group."""
    return df.loc[df.groupby(by)[dev].idxmax()]


def load(results_dir):
    core = pd.read_csv(results_dir / "best_vs_paper_with_finetune.csv")
    d = core[["task", "dataset"]].copy()
    d["frozen"] = core["frozen_mcc_x100"] / 100.0
    d["lora"] = core["lora_mcc_x100"] / 100.0
    d["paper"] = core["paper_mcc_x100"] / 100.0

    bag = _best_on_dev(pd.read_csv(results_dir / "bag_best_per_dataset.csv"),
                       "dataset", "dev_mcc_mean")
    d = d.merge(bag[["dataset", "test_mcc_mean"]].rename(
        columns={"test_mcc_mean": "bag"}), on="dataset", how="left")
    assert d["bag"].notna().all(), "a dataset has no bag-arm row"

    d = pd.concat([d, _ssp_row(results_dir)], ignore_index=True)
    d["task"] = pd.Categorical(d["task"], categories=TASKS, ordered=True)
    return d.sort_values(["task", "dataset"]).reset_index(drop=True)


def _ssp_row(results_dir):
    """Splice, assembled from four separate arms' tables.

    The bag table is raw per-seed rows, so it is aggregated first. Frozen is
    restricted to the mean/[CLS] readouts: the windowed readouts are a 6,144-d
    featurization no other dataset was given.
    """
    need = ["bag_results_ssp.csv", "positional_summary.csv", "lora_results.csv",
            "paper_table6_dnabert2.csv"]
    if not all((results_dir / f).exists() for f in need):
        print("  (splice inputs incomplete -- SSP skipped)")
        return pd.DataFrame()

    b = pd.read_csv(results_dir / "bag_results_ssp.csv")
    b = (b.groupby(["feat", "head_type"], as_index=False)
           .agg(dev=("dev_mcc", "mean"), test=("test_mcc", "mean")))
    bag = b.loc[b["dev"].idxmax(), "test"]

    p = pd.read_csv(results_dir / "positional_summary.csv")
    p = p[(p["task"] == "SSP") & (p["readout"].isin(["mean", "cls"]))]

    lo = pd.read_csv(results_dir / "lora_results.csv")
    lo = lo[lo["dataset"].str.contains("splice")]

    pt = pd.read_csv(results_dir / "paper_table6_dnabert2.csv")
    pt = pt[pt["task"] == "SSP"]
    if p.empty or lo.empty or pt.empty:
        print("  (no splice frozen/LoRA/paper score -- SSP skipped)")
        return pd.DataFrame()

    return pd.DataFrame([{
        "task": "SSP", "dataset": "splice__reconstructed", "bag": float(bag),
        "frozen": float(p.loc[p["dev_mcc"].idxmax(), "test_mcc"]),
        "lora": float(lo["test_mcc"].mean()),
        "paper": float(pt.iloc[0]["paper_mcc_x100"]) / 100.0,
    }])


def normalise(d):
    for key, _lab in ARMS:
        d[f"{key}_norm"] = 100.0 * d[key] / d["paper"]
    return d


def draw(d):
    bf.use()
    N = bf.NEUTRAL
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    bases = list(bf.CATEGORY) + [SSP_BASE]
    w = 0.26
    med = d.groupby("task", observed=True)[[f"{k}_norm" for k, _ in ARMS]].median()

    for i, task in enumerate(TASKS):
        if task not in med.index:
            continue
        g = d[d["task"] == task]
        for j, ((key, _lab), colour) in enumerate(zip(ARMS, get_variants(bases[i]))):
            x = i + (j - 1) * w
            lw_, ls_ = EDGES[j]
            ax.bar(x, med.loc[task, f"{key}_norm"], width=w * 0.92,
                   color=mcolors.to_rgba(colour, ALPHAS[j]),
                   edgecolor=get_variants(bases[i])[2], linewidth=lw_,
                   linestyle=ls_, zorder=2)
            jit = rng.uniform(-w * 0.28, w * 0.28, len(g))
            ax.scatter(x + jit, g[f"{key}_norm"], s=22, color=N["surface"],
                       edgecolor=N["ink"], lw=0.9, alpha=0.9, zorder=4)

    ax.axhline(100, ls=(0, (4, 3)), lw=1.2, color=bf.REFERENCE, zorder=1)

    n = d.groupby("task", observed=True).size()
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([f"{TASK_LABELS[t]}\n(n={n.get(t, 0)})" for t in TASKS])
    ax.set_xlim(-0.55, len(TASKS) - 0.45)
    ax.set_ylim(0, max(112, d[[f"{k}_norm" for k, _ in ARMS]].max().max() + 6))
    ax.set_ylabel("test MCC, % of full fine-tuning")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)

    greys = get_variants(N["muted"])
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=mcolors.to_rgba(c, a),
                             edgecolor=greys[2], linewidth=lw_, linestyle=ls_,
                             label=lab)
               for (_k, lab), c, a, (lw_, ls_) in zip(ARMS, greys, ALPHAS, EDGES)]
    handles.append(plt.Line2D([], [], ls="", marker="o", ms=5,
                              markerfacecolor=N["surface"],
                              markeredgecolor=N["ink"], color=N["ink"],
                              label="individual dataset"))
    handles.append(plt.Line2D([], [], ls=(0, (4, 3)), lw=1.2, color=bf.REFERENCE,
                              label="full fine-tuning (paper Table 6) = 100"))
    # matplotlib fills legend columns top-down: interleave the arms with the two
    # reference entries (and a blank) so the arms land on row 1 in ladder order.
    arms, extras = handles[:3], handles[3:]
    blank = plt.Line2D([], [], ls="", label=" ")
    handles = [arms[0], extras[0], arms[1], extras[1], arms[2], blank]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              ncol=3, frameon=False, fontsize=10)

    m = med.mean()
    fig.suptitle(
        "On average, k-mer frequency and frozen DNABERT2-based classifiers are 82-85% as accurate as full fine-tuning\n"
        "LoRA & full fine-tuning are nearly identical on GUE benchmarks",
        fontsize=12.5, x=0.5, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 1.0))
    fig.subplots_adjust(top=0.89)  # tight_layout leaves a gap under the suptitle
    return fig, med


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures")
    d = normalise(load(results_dir))
    fig, med = draw(d)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(outdir / f"{NAME}.{ext}")
    table = results_dir / f"{NAME}.csv"
    d.to_csv(table, index=False)
    print(med.round(1).to_string())
    print(f"wrote {outdir}/{NAME}.svg and .png, {table}  ({len(d)} datasets)")


if __name__ == "__main__":
    main()
