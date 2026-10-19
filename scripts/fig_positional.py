"""Figures for the positional-readout arm.

  position_fig1  where each GUE task's label-discriminating signal sits, how
                 concentrated it is once corrected for sequence length, and --
                 underneath -- what that information content actually is, as
                 acceptor and donor sequence logos around the splice junction
  position_fig2  test MCC by layer for four readouts of the same frozen
                 embeddings on three tasks, plus gain against concentration

fig1 reads the GUE training CSVs only (no model). fig2 reads
results/positional_by_layer.csv and results/positional_summary.csv.

Both are styled with blogfig, the canonical copy of which lives outside this
repo; set BLOGFIG_HOME if it is elsewhere.

Usage:
    python scripts/fig_positional.py [OUTDIR]
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dnabert2_gue as SL

BLOGFIG_HOME = os.environ.get("BLOGFIG_HOME", "/home/jim/code/blogfig-brand")
sys.path.insert(0, BLOGFIG_HOME)
import blogfig as bf  # noqa: E402
import logomaker  # noqa: E402

BASES = ["A", "C", "G", "T"]
# Where the two splice classes put their motifs, measured from the training
# split: the exon/intron boundary falls between 0-based 199 and 200 in every
# sequence. This window covers the acceptor polypyrimidine tract and AG, and
# the donor GT plus its GTRAGT extension.
LOGO_FROM, LOGO_TO = 184, 208
IC_FLOOR = 0.15            # bits; positions below this are drawn blank
SSP_CLASS = {0: "acceptor", 1: "donor", 2: "negative"}

PANEL_TASKS = [
    ("splice__reconstructed", "SSP  splice (400 bp)"),
    ("prom__prom_300_all", "PD  prom_300_all (300 bp)"),
    ("prom__prom_core_all", "CPD  prom_core_all (70 bp)"),
    ("tf__0", "TF-H  tf__0 (101 bp)"),
    ("mouse__0", "TF-M  mouse__0 (101 bp)"),
]


# -------------------------------------------------------------------------
# per-position information content
# -------------------------------------------------------------------------

def base_freq(seqs):
    """(4, L) base frequency, one row per base, from equal-length sequences."""
    L = len(seqs[0])
    arr = np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(len(seqs), L)
    return np.stack([(arr == b.encode()).mean(0) for b in BASES])


def info_content(freq):
    """Schneider information content per position: 2 - H(p), in bits."""
    p = np.clip(freq, 1e-9, 1.0)
    return 2.0 + (p * np.log2(p)).sum(0)


def load_fixed(dataset, split="train", nmax=40000):
    df = pd.read_csv(os.path.join(SL.dataset_dir(dataset), f"{split}.csv"))
    s = df["sequence"].str.upper()
    assert s.str.len().nunique() == 1, f"{dataset} is variable length"
    return s.iloc[:nmax].tolist(), df["label"].values[:nmax]


def task_ic(dataset):
    """Per-position IC, maximized over classes -- the label-discriminating part."""
    seqs, y = load_fixed(dataset)
    return np.maximum.reduce(
        [info_content(base_freq([s for s, c in zip(seqs, y) if c == k]))
         for k in np.unique(y)])


# -------------------------------------------------------------------------
# figure 1
# -------------------------------------------------------------------------

def logo_matrix(seqs, lo=None, hi=None):
    """logomaker frame of IC-scaled letter heights over the logo window.

    Letter height is p_i x IC, the standard Schneider scaling, so a column's
    total height is its information content in bits and a position where all
    four bases are equally likely is blank. Columns under IC_FLOOR are zeroed
    rather than drawn tiny: the point of the panel is which positions are
    constrained, and a forest of 0.05-bit letters hides it.
    """
    lo = LOGO_FROM if lo is None else lo
    hi = LOGO_TO if hi is None else hi
    f = base_freq(seqs)[:, lo:hi]
    ic = info_content(base_freq(seqs))[lo:hi]
    ic = np.where(ic < IC_FLOOR, 0.0, ic)
    m = pd.DataFrame((f * ic).T, columns=BASES,
                     index=np.arange(lo, hi))
    return m


def figure1(outdir):
    bf.use()
    ics = {d: task_ic(d) for d, _ in PANEL_TASKS}
    conc = {d: 100 * np.sort(v)[::-1][:20].sum() / v.sum() for d, v in ics.items()}
    enr = {d: conc[d] / (100 * 20 / len(ics[d])) for d in ics}
    cols = dict(zip([d for d, _ in PANEL_TASKS], bf.CATEGORY))

    seqs, y = load_fixed("splice__reconstructed")
    logos = {k: logo_matrix([s for s, c in zip(seqs, y) if c == k]) for k in (0, 1)}

    fig = plt.figure(figsize=(10.5, 6.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 0.85, 0.85],
                          width_ratios=[1.75, 1], hspace=0.62, wspace=0.22)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    for d, lab in PANEL_TASKS:
        v = ics[d]
        axA.plot(np.linspace(0, 1, len(v)), v, color=cols[d], lw=1.3, label=lab)
    axA.set_xlabel("position (fraction of sequence length)")
    axA.set_ylabel("information content (bits)")
    axA.set_title("Where the label-discriminating signal sits", fontsize=11, pad=6)
    axA.legend(frameon=False, fontsize=8.5, loc="upper left")
    axA.set_ylim(0, 2.1)

    order = sorted(ics, key=lambda d: enr[d])
    axB.barh(range(5), [enr[d] for d in order],
             color=[cols[d] for d in order], height=0.62)
    axB.axvline(1.0, color=bf.REFERENCE, ls=(0, (4, 3)), lw=1.1)
    axB.set_yticks(range(5))
    axB.set_yticklabels([dict(PANEL_TASKS)[d].split("  ")[0] for d in order])
    axB.set_xlabel("IC concentration in the top 20\npositions (\u00d7 uniform for that length)",
                   fontsize=9)
    axB.set_title("How positional the task is", fontsize=11, pad=6)
    for i, d in enumerate(order):
        axB.text(enr[d] + 0.35, i, f"{enr[d]:.1f}\u00d7", va="center", fontsize=9,
                 color=bf.NEUTRAL["ink"])
    axB.set_xlim(0, 18.5)

    for row, k in ((1, 0), (2, 1)):
        ax = fig.add_subplot(gs[row, :])
        logomaker.Logo(logos[k], ax=ax, color_scheme="classic",
                       show_spines=False, vpad=0.02)
        ax.axvline(199.5, color=bf.REFERENCE, ls=(0, (4, 3)), lw=1.1, zorder=0)
        ax.set_ylim(0, 2.05)
        ax.set_ylabel("bits", fontsize=9)
        ax.set_xlim(LOGO_FROM - 0.6, LOGO_TO - 0.4)
        ax.set_xticks(range(185, LOGO_TO, 4))
        ax.tick_params(labelsize=8.5)
        ax.set_title(f"SSP splice \u2014 {SSP_CLASS[k]} sequences "
                     f"(dashed line = exon/intron boundary)",
                     fontsize=10, pad=4, loc="left")
        if row == 2:
            ax.set_xlabel("position in the 400 bp window", fontsize=9.5)
        else:
            ax.set_xticklabels([])

    fig.suptitle("Only one GUE task has its signal pinned to a position",
                 fontsize=12.5, y=0.985)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(outdir, f"position_fig1.{ext}"))
    return fig


def _shade(hex_color, f):
    """Lighten (f > 0, toward white) or darken (f < 0, toward ink)."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(hex_color)
    if f >= 0:
        return tuple(c + (1.0 - c) * f for c in (r, g, b))
    ink = mcolors.to_rgb(bf.NEUTRAL["ink"])
    return tuple(c + (i - c) * (-f) for c, i in zip((r, g, b), ink))


# One hue, three lightness steps. The sibling panels in position_fig1b use
# bf.CATEGORY to mean *dataset*, so this panel must not spend categorical
# hues on readouts; lightness carries the series instead, darkest = most
# intervention, and markers keep them apart at print size.
GAIN_SERIES = [
    ("win8_best", "best single window, frozen", 0.52, "s"),
    ("win8", "all 8 windows, frozen", 0.0, "o"),
    ("lora", "LoRA fine-tune\n(0.75% of weights)", -0.55, "^"),
]


def gain_panel(ax, legend=True, conc=None):
    """Gain over the mean-pooled probe, as a slopegraph across three tasks.

    Tasks run left to right in increasing positional concentration, on a
    categorical axis: the concentration values themselves are already an
    axis in the companion bar panel, so here they are only a tick label
    saying why the order is the order. One line per readout, so the reader
    sees that no intervention pays until the signal has a place, and that
    the three never cross.

    conc: optional {dataset: x-uniform enrichment} so the caller can pass
    the value it computed rather than the module's recorded CONC3.
    """
    cc = dict(CONC3) if conc is None else {d: conc[d] for d, _ in DS3}
    summ = pd.read_csv(os.path.join(SL.RESULTS_DIR, "positional_summary.csv"))
    q = summ.set_index(["dataset", "readout"])
    base = {d: q.loc[(d, "mean"), "test_mcc"] for d, _ in DS3}

    lora_p = os.path.join(SL.RESULTS_DIR, "lora_results.csv")
    lora = (pd.read_csv(lora_p).query("variant == 'attn'")
            .groupby("dataset").test_mcc.mean() if os.path.exists(lora_p)
            else pd.Series(dtype=float))

    order = sorted((d for d, _ in DS3), key=lambda d: cc[d])
    xs = np.arange(len(order))
    left = []   # left-end value labels, placed after the loop so they can be
                # fanned apart -- on a low-gain task they land on top of each other

    for key, lab, f, mk in GAIN_SERIES:
        col = _shade(bf.ACCENT["primary"], f)
        if key == "lora":
            ys = [lora[d] - base[d] if d in lora.index else np.nan for d in order]
        else:
            ys = [q.loc[(d, key), "test_mcc"] - base[d] for d in order]
        ax.plot(xs, ys, "-", color=col, lw=2.0, marker=mk, ms=7,
                mfc=col, mec="white", mew=1.0, zorder=3, label=lab)
        if np.isfinite(ys[-1]):
            ax.annotate(f"{ys[-1]:+.2f}", (xs[-1], ys[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color=col, fontweight="bold")
        if np.isfinite(ys[0]):
            left.append((ys[0], col))

    ax.axhline(0, color=bf.NEUTRAL["ink"], lw=1.0, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{dict(DS3)[d].split('  ')[0]}\n{cc[d]:.1f}\u00d7"
                        for d in order], fontsize=10)
    ax.set_xlim(-0.55, len(order) - 0.38)
    ax.set_ylabel("$\\Delta$ MCC", fontsize=10)
    # Left-anchored titles (blogfig default) overrun this narrow right-hand
    # column; nothing is drawn in the gutter, so shift the title into it.
    ax.set_title("Gains over mean pooled probe by task", fontsize=10.5, pad=6,
                 loc="left", x=-0.37)
    ax.tick_params(labelsize=9)
    ax.margins(y=0.16)
    # Headroom so the climb into SSP clears the legend block.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.30 * (hi - lo))
    if legend:
        ax.legend(frameon=False, fontsize=8.5, loc="upper left",
                  handlelength=1.6, borderaxespad=0.2, labelspacing=0.35)

    # Left-end labels last, once the y-scale is final: on a low-gain task the
    # three values sit within a hair of each other, so push each one up until
    # it clears the one below rather than fanning symmetrically (which would
    # drive the lowest off the bottom of the panel).
    placed = -np.inf
    for y, col in sorted(left):
        dy = max(0.0, placed + 11.0 - ax.transData.transform((0, y))[1])
        ax.annotate(f"{y:+.2f}", (xs[0], y), xytext=(-8, dy),
                    textcoords="offset points", va="center", ha="right",
                    fontsize=8.5, color=col)
        placed = ax.transData.transform((0, y))[1] + dy


def figure1_aligned(outdir):
    """position_fig1b: logo tracks share the IC panel's width and extent.

    The variant exists for one reason: with the logos on the same horizontal
    scale as the IC profile, the motif sits directly under the information
    spike and the reader can see they are the same object. The cost is that
    the motif is 6% of a 400 bp window, so at full width its letters are
    ~2 pt -- each track therefore carries a magnified callout in the empty
    right-hand space, which costs nothing because nothing is drawn there.

    logomaker requires an integer index, so the logo axes stay in bp while
    the IC axis is a 0-1 fraction; the two align because both span their
    full range across the same column width.
    """
    bf.use()
    ics = {d: task_ic(d) for d, _ in PANEL_TASKS}
    conc = {d: 100 * np.sort(v)[::-1][:20].sum() / v.sum() for d, v in ics.items()}
    enr = {d: conc[d] / (100 * 20 / len(ics[d])) for d in ics}
    cols = dict(zip([d for d, _ in PANEL_TASKS], bf.CATEGORY))

    seqs, y = load_fixed("splice__reconstructed")
    L = len(seqs[0])
    full = {k: logo_matrix([s for s, c in zip(seqs, y) if c == k], 0, L)
            for k in (0, 1)}
    zoom = {k: logo_matrix([s for s, c in zip(seqs, y) if c == k], 186, 207)
            for k in (0, 1)}
    J = 199.5 / L
    ticks = np.arange(0, 1.01, 0.2)

    fig = plt.figure(figsize=(11.0, 6.2))
    # Concentration bars sit in the top row only, level with the IC profile;
    # the two rows beneath them carry the readout-gain panel, so this one
    # figure runs from "where the signal is" to "what reading it there buys".
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 1.0, 1.0],
                          width_ratios=[2.7, 1], hspace=0.45, wspace=0.28)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axG = fig.add_subplot(gs[1:, 1])

    for d, lab in PANEL_TASKS:
        v = ics[d]
        axA.plot(np.linspace(0, 1, len(v)), v, color=cols[d], lw=1.3, label=lab)
    axA.axvline(J, color=bf.REFERENCE, ls=(0, (4, 3)), lw=1.0, zorder=0)
    axA.set_ylabel("information content (bits)", fontsize=9.5)
    axA.set_title("SSP acceptor / donor sequences are centered on conserved motifs",
                  fontsize=11, pad=6)
    axA.legend(frameon=False, fontsize=8.5, loc="upper left")
    axA.set_ylim(0, 2.1)
    axA.set_xlim(0, 1)
    axA.set_xticks(ticks)
    axA.set_xticklabels([])

    for row, k in ((1, 0), (2, 1)):
        ax = fig.add_subplot(gs[row, 0])
        logomaker.Logo(full[k], ax=ax, color_scheme="classic", show_spines=False)
        ax.axvline(199.5, color=bf.REFERENCE, ls=(0, (4, 3)), lw=1.0, zorder=0)
        ax.set_xlim(0, L)
        ax.set_ylim(0, 2.05)
        ax.set_xticks(ticks * L)
        ax.set_ylabel("bits", fontsize=9)
        ax.tick_params(labelsize=8.5)
        ax.set_title(f"SSP splice \u2014 {SSP_CLASS[k]} sequences",
                     fontsize=10, pad=3, loc="left")
        if row == 2:
            ax.set_xticklabels([f"{t:.1f}" for t in ticks])
            ax.set_xlabel("position (fraction of sequence length)", fontsize=9.5)
        else:
            ax.set_xticklabels([])

        axi = ax.inset_axes([0.60, 0.10, 0.37, 0.84])
        logomaker.Logo(zoom[k], ax=axi, color_scheme="classic", show_spines=False)
        axi.axvline(199.5, color=bf.REFERENCE, ls=(0, (4, 3)), lw=1.0, zorder=0)
        axi.set_xlim(185.4, 206.6)
        axi.set_ylim(0, 2.05)
        axi.set_xticks([190, 200])
        axi.set_yticks([0, 1, 2])
        axi.tick_params(labelsize=7.5)
        axi.set_facecolor("white")
        for sp in axi.spines.values():
            sp.set_visible(True)
            sp.set_color(bf.NEUTRAL["axis"])
            sp.set_linewidth(0.8)
        axi.text(0.03, 0.95, "bp 186\u2013206, magnified", transform=axi.transAxes,
                 fontsize=7.5, color=bf.NEUTRAL["muted"], va="top")
        ax.indicate_inset([186, 0, 21, 2.05], inset_ax=axi,
                          edgecolor=bf.NEUTRAL["muted"], lw=0.8, alpha=0.9)

    order = sorted(ics, key=lambda d: enr[d])
    axB.barh(range(5), [enr[d] for d in order],
             color=[cols[d] for d in order], height=0.6)
    axB.axvline(1.0, color=bf.REFERENCE, ls=(0, (4, 3)), lw=1.1)
    axB.set_yticks(range(5))
    axB.set_yticklabels([dict(PANEL_TASKS)[d].split("  ")[0] for d in order])
    # No x-label here: it would sit on the gain panel's title below.
    axB.set_title("How positional the task is\n(top-20 IC share \u00f7 uniform)",
                  fontsize=10.5, pad=6)
    for i, d in enumerate(order):
        axB.text(enr[d] + 0.4, i, f"{enr[d]:.1f}\u00d7", va="center", fontsize=9,
                 color=bf.NEUTRAL["ink"])
    axB.set_xlim(0, 19)
    axB.tick_params(labelsize=9)

    gain_panel(axG, conc=enr)

    fig.suptitle("Mean pooling dilutes strong positional signal in the splice-site dataset",
                 fontsize=12.5, y=0.98)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(outdir, f"position_fig1b.{ext}"))
    return fig


# -------------------------------------------------------------------------
# figure 2
# -------------------------------------------------------------------------

READOUTS = ["mean", "cls", "win8_best", "win8"]
RLAB = {"mean": "mean pooling (768-d)", "cls": "[CLS] (768-d)",
        "win8_best": "best single window (768-d)",
        "win8": "all 8 windows (6,144-d)"}
DS3 = [("splice__reconstructed", "SSP  splice\n15.5\u00d7 positional"),
       ("prom__prom_300_all", "PD  prom_300_all\n2.8\u00d7"),
       ("tf__0", "TF-H  tf__0\n1.1\u00d7")]
CONC3 = {"splice__reconstructed": 15.5, "prom__prom_300_all": 2.8, "tf__0": 1.1}


def figure2(outdir):
    bf.use()
    R = SL.RESULTS_DIR
    by_layer = pd.read_csv(os.path.join(R, "positional_by_layer.csv"))
    summ = pd.read_csv(os.path.join(R, "positional_summary.csv"))
    q = summ.set_index(["dataset", "readout"])
    lora_p = os.path.join(R, "lora_results.csv")
    lora = (pd.read_csv(lora_p).query("variant == 'attn'")
            .groupby("dataset").test_mcc.mean() if os.path.exists(lora_p)
            else pd.Series(dtype=float))
    rcol = dict(zip(READOUTS, [bf.ACCENT["primary"], bf.NEUTRAL["faint"],
                               bf.CATEGORY[2], bf.ACCENT["secondary"]]))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    for ax, (d, lab) in zip(axes[:3], DS3):
        g = by_layer[by_layer.dataset == d]
        ax.axhline(q.loc[(d, "mean"), "kmer_floor_test_mcc"],
                   color=bf.REFERENCE, ls=(0, (4, 3)), lw=1.1)
        if d in lora.index:
            ax.axhline(lora[d], color=bf.CATEGORY[4], ls=(0, (1, 1.6)), lw=1.6)
        for r in READOUTS:
            s = g[g.readout == r].sort_values("layer")
            ax.plot(s.layer, s.test_mcc, color=rcol[r],
                    lw=1.8 if r in ("mean", "win8") else 1.2,
                    marker="o", ms=3.2)
        ax.plot([q.loc[(d, "win8shuf"), "layer"]], [q.loc[(d, "win8shuf"), "test_mcc"]],
                marker="X", ms=8, color=bf.ACCENT["secondary"], mfc="white",
                mew=1.8, zorder=5)
        ax.set_title(lab, fontsize=10.5, pad=6)
        ax.set_xlabel("encoder layer", fontsize=9.5)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.set_ylim(0.18, 0.95)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("test MCC", fontsize=9.5)
    for ax in axes[1:3]:
        ax.set_yticklabels([])
    d0 = DS3[0][0]
    axes[0].annotate("LoRA fine-tune",
                     xy=(2, lora.get(DS3[0][0], np.nan)),
                     xytext=(1.2, 0.905), fontsize=8.5, color=bf.CATEGORY[4])
    axes[0].annotate("k-mer floor",
                     xy=(7.5, q.loc[(d0, "mean"), "kmer_floor_test_mcc"]),
                     xytext=(6.5, 0.255), fontsize=8.5, color=bf.NEUTRAL["muted"])
    axes[0].annotate("windows shuffled",
                     xy=(q.loc[(d0, "win8shuf"), "layer"],
                         q.loc[(d0, "win8shuf"), "test_mcc"]),
                     xytext=(6.4, 0.50), fontsize=8.5, color=bf.NEUTRAL["muted"],
                     arrowprops=dict(arrowstyle="-", color=bf.NEUTRAL["faint"], lw=0.9))


    handles = [plt.Line2D([], [], color=rcol[r], lw=1.8 if r in ("mean", "win8") else 1.2,
                          marker="o", ms=3.2, label=RLAB[r]) for r in READOUTS]
    handles += [plt.Line2D([], [], color=bf.ACCENT["secondary"], marker="X", ms=8,
                           mfc="white", mew=1.8, ls="none",
                           label="all 8 windows, order shuffled"),
                plt.Line2D([], [], color=bf.CATEGORY[4], ls=(0, (1, 1.6)), lw=1.6,
                           label="LoRA fine-tune (0.75% of weights)"),
                plt.Line2D([], [], color=bf.REFERENCE, ls=(0, (4, 3)), lw=1.1,
                           label="k-mer floor")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("Where you read a frozen embedding matters only when the "
                 "task's signal has a place", fontsize=12.5, y=1.04)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(outdir, f"position_fig2.{ext}"))
    return fig


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else SL.FIGURES_DIR
    Path(out).mkdir(parents=True, exist_ok=True)
    figure1(out)
    figure1_aligned(out)
    figure2(out)
    print(f"wrote position_fig1, position_fig1b and position_fig2 (svg + png) to {out}")
