"""Frozen vs LoRA-adapted position-0 representation, as UMAP layouts.

Two rows (PD = prom__prom_300_all, SSP = splice__reconstructed) x four
representations of the same held-out test sequences:

    6-mer frequency | frozen mean-pooled | frozen [CLS] | LoRA [CLS]

The first three are label-free -- nothing in those pipelines ever saw a label.
The fourth was trained on these labels to be linearly separable by the head
above it, so it separates by construction; the gap and the two spanning
headers say so on the figure rather than in the caption. The point the figure
makes is the *pair*: columns 3 and 4 are the same position in the same
architecture, differing only in whether gradients touched the encoder.

Reads (relative to PROJECT_ROOT):
    layer_embed/<dataset>__test__L{06,05}__{mean,cls}.npy
    results/lora_cls/<dataset>__attn__s42__test__cls.npy  (+ __y.npy)
    data/GUE/<family>/<sub>/test.csv                      (for the k-mer panel)

Layouts are cached to results/umap_cls_coords.npz; pass --refit to rebuild
them (about 70 s for all eight fits).

Usage:
    python scripts/fig_lora_cls_umap.py [PROJECT_ROOT] [OUTDIR] [--refit]

Needs umap-learn, which is NOT in requirements.lock.txt -- add it before
running in the project venv.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

BLOGFIG_HOME = Path("/home/jim/code/blogfig-brand")
sys.path.insert(0, str(BLOGFIG_HOME))
import blogfig as bf  # noqa: E402

# dataset -> (row tag, GUE csv subpath, dev-selected frozen layer)
DATASETS = {
    "prom__prom_300_all":    ("PD",  "prom/prom_300_all",    6),
    "splice__reconstructed": ("SSP", "splice/reconstructed",  5),
}
COLS = [("kmer6", "6-mer frequency"), ("frozen_mean", "frozen mean-pooled"),
        ("frozen_cls", "frozen [CLS]"), ("lora_cls", "LoRA [CLS]")]

# Test MCC under each panel. The three label-free columns use the LR head so
# the heads match across them: kmer6 from bag_results{,_ssp}.csv with
# head_type == "LR", frozen from positional_summary.csv (also the LR grid).
# The LoRA column is read from results/lora_cls/lora_cls_manifest.csv at run
# time, because it belongs to the re-run that produced these arrays and not to
# the published lora_results.csv rows.
MCC = {"PD":  {"kmer6": 0.7585, "frozen_mean": 0.8256, "frozen_cls": 0.7990},
       "SSP": {"kmer6": 0.2938, "frozen_mean": 0.4643, "frozen_cls": 0.3942}}

# class -> (colour, label, z-order). The background class is drawn first, and
# no hue carries two meanings across the rows: PD's balanced pair gets the
# accent pair, SSP's two positive classes get category hues and its 57%
# negative class gets neutral grey.
CLASSES = {
    "PD":  {0: ("primary", "non-promoter", 1), 1: ("secondary", "promoter", 2)},
    "SSP": {2: ("faint", "negative", 1), 0: ("cat2", "acceptor", 2),
            1: ("cat3", "donor", 3)},
}

UMAP_KW = dict(n_components=2, n_neighbors=30, min_dist=0.1, random_state=0)

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

# Semantics copied verbatim from src/dnabert2_gue/bags.py::kmer_matrix so this
# script stays torch-free (importing dnabert2_gue pulls in torch). Keep the two
# in sync: windows containing non-ACGT are dropped, counts are L1-normalised.
_LUT = np.full(256, -1, np.int8)
for _i, _b in enumerate(b"ACGT"):
    _LUT[_b] = _i


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


def embedding_pipeline():
    """768-d embeddings: standardise, denoise to 50 PCs, cosine graph."""
    from umap import UMAP
    return Pipeline([("scale", StandardScaler()),
                     ("pca", PCA(n_components=50, random_state=0)),
                     ("umap", UMAP(metric="cosine", **UMAP_KW))])


def kmer_pipeline():
    """4,096-d frequencies: no StandardScaler (it would promote rare-k-mer
    sampling noise to equal footing, which is a real risk here and not in the
    768-d embeddings), no PCA (it leaves the simplex), and Hellinger, the
    natural distance between probability vectors."""
    from umap import UMAP
    return Pipeline([("l1", Normalizer(norm="l1")),
                     ("umap", UMAP(metric="hellinger", **UMAP_KW))])


def fit_layouts(root):
    """Eight independent UMAP fits -> {f'{tag}__{col}': (n, 2)}."""
    out = {}
    for dataset, (tag, csv_sub, layer) in DATASETS.items():
        seqs = pd.read_csv(root / "data/GUE" / csv_sub / "test.csv")["sequence"]
        X_kmer, dropped = kmer_matrix(seqs.tolist(), 6)
        feats = {
            "kmer6": (X_kmer, kmer_pipeline()),
            "frozen_mean": (np.load(root / "layer_embed" /
                                    f"{dataset}__test__L{layer:02d}__mean.npy"),
                            embedding_pipeline()),
            "frozen_cls": (np.load(root / "layer_embed" /
                                   f"{dataset}__test__L{layer:02d}__cls.npy"),
                           embedding_pipeline()),
            "lora_cls": (np.load(root / "results/lora_cls" /
                                 f"{dataset}__attn__s42__test__cls.npy"),
                         embedding_pipeline()),
        }
        y = np.load(root / "layer_embed" / f"{dataset}__test__y.npy")
        y_lora = np.load(root / "results/lora_cls" /
                         f"{dataset}__attn__s42__test__y.npy")
        assert np.array_equal(y, y_lora), (
            f"{dataset}: frozen and LoRA label vectors disagree -- the panels "
            f"are not row-aligned and the figure would be meaningless")
        out[f"{tag}__y"] = y
        for col, (X, pipe) in feats.items():
            t0 = time.time()
            out[f"{tag}__{col}"] = pipe.fit_transform(X).astype(np.float32)
            print(f"  {tag:4s} {col:12s} {X.shape} -> 2  "
                  f"({time.time() - t0:.1f}s)", flush=True)
        print(f"  {tag}: {dropped} k-mer windows dropped (non-ACGT)", flush=True)
    return out


def layouts(root, refit=False):
    cache = root / "results" / "umap_cls_coords.npz"
    if cache.exists() and not refit:
        print(f"reusing {cache} (--refit to rebuild)")
        return dict(np.load(cache))
    out = fit_layouts(root)
    np.savez_compressed(cache, **out)
    print(f"wrote {cache}")
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def draw(coords, mcc):
    bf.use()
    N = bf.NEUTRAL
    palette = {"primary": bf.ACCENT["primary"], "secondary": bf.ACCENT["secondary"],
               "cat2": bf.CATEGORY[2], "cat3": bf.CATEGORY[3], "faint": N["faint"]}
    small = plt.rcParams["legend.fontsize"]

    fig = plt.figure(figsize=(9.6, 5.3))
    # column 4 is held apart by a spacer column, not by colour alone; the right
    # margin is reserved for the per-row class keys
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.22, 1],
                          hspace=0.26, wspace=0.07,
                          left=0.055, right=0.855, top=0.80, bottom=0.075)

    axes = {}
    for r, (dataset, (tag, _, layer)) in enumerate(DATASETS.items()):
        y = coords[f"{tag}__y"]
        for c, (key, col_label) in enumerate(COLS):
            ax = fig.add_subplot(gs[r, c if c < 3 else 4])
            axes[(tag, key)] = ax
            Z = coords[f"{tag}__{key}"]
            for cls, (swatch, label, z) in sorted(CLASSES[tag].items(),
                                                  key=lambda kv: kv[1][2]):
                m = y == cls
                ax.scatter(Z[m, 0], Z[m, 1], s=1.6, alpha=0.38, linewidths=0,
                           color=palette[swatch], zorder=z, rasterized=True,
                           # labels live on the column the key is drawn beside
                           label=label if key == "lora_cls" else None)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.margins(0.06)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                # the band headers above these are the higher tier: bigger,
                # bolder, darker. Column titles sit one step down.
                ax.set_title(col_label, pad=6, fontsize=small - 0.5,
                             fontweight="normal", color=N["ink"])
            sub = (f"layer {layer} · MCC {mcc[tag][key]:.3f}"
                   if key.startswith("frozen") else f"MCC {mcc[tag][key]:.3f}")
            ax.text(0.5, -0.035, sub, transform=ax.transAxes, ha="center",
                    va="top", color=N["ink"], fontsize=small)
            if c == 0:
                ax.set_ylabel(tag, labelpad=10, color=N["ink"])

    # The band is measured off the axes so it can never cross a column title.
    p_tl = axes[("PD", "kmer6")].get_position()
    p_br = axes[("SSP", "frozen_cls")].get_position()
    p_lora = axes[("PD", "lora_cls")].get_position()
    x0, x1 = p_tl.x0 - 0.012, p_br.x1 + 0.012
    y0, y1 = p_br.y0 - 0.048, p_tl.y1 + 0.006
    fig.patches.append(Rectangle((x0, y0), x1 - x0, y1 - y0, lw=0, zorder=0,
                                 transform=fig.transFigure,
                                 facecolor=N["grid"], alpha=0.35))
    band_kw = dict(ha="center", va="bottom", color=N["ink"],
                   fontsize=small + 1.5, fontweight="bold")
    fig.text((x0 + x1) / 2, y1 + 0.052, "label-free representations", **band_kw)
    fig.text((p_lora.x0 + p_lora.x1) / 2, y1 + 0.052, "trained on this task",
             **band_kw)

    # Class keys sit in the right margin, one per row, clear of every panel.
    for tag, _, _ in DATASETS.values():
        leg = axes[(tag, "lora_cls")].legend(
            frameon=False, markerscale=7, handletextpad=0.3, borderpad=0.0,
            labelspacing=0.5, loc="center left", bbox_to_anchor=(1.04, 0.5),
            fontsize=small)
        for h in leg.legend_handles:
            h.set_alpha(1.0)

    # An embedding scatter has no meaningful units, so the axes carry a corner
    # cue on one panel rather than ticks anywhere.
    a = axes[("SSP", "kmer6")]
    for xy, xytext in [((0.20, 0.035), (0.035, 0.035)),
                       ((0.035, 0.20), (0.035, 0.035))]:
        a.annotate("", xy=xy, xytext=xytext, xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="->", lw=0.9, color=N["ink"]))
    a.text(0.22, 0.035, "UMAP-1", transform=a.transAxes, va="center",
           color=N["ink"], fontsize=small)
    a.text(0.035, 0.22, "UMAP-2", transform=a.transAxes, ha="left",
           va="bottom", color=N["ink"], fontsize=small)

    # The claim is about the procedure, not the representation: one recipe,
    # two tasks, a classifier that is specific to each. Nothing here says an
    # SSP-adapted [CLS] would be useful on PD, and it would not be.
    fig.suptitle("LoRA fine-tuning trains [CLS] into a task-specific but "
                 "effective classifier", y=0.955)
    return fig


def bbox_report(fig):
    """Visible text boxes that overlap another text box. [] means clean."""
    import matplotlib as mpl
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    return [(a.get_text(), b.get_text())
            for i, (a, ba) in enumerate(texts)
            for b, bb in texts[i + 1:] if ba.overlaps(bb)]


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else Path(".")
    rest = [a for a in sys.argv[2:] if not a.startswith("-")]
    outdir = Path(rest[0]) if rest else Path("figures")
    refit = "--refit" in sys.argv

    mcc = {k: dict(v) for k, v in MCC.items()}
    man = pd.read_csv(root / "results/lora_cls/lora_cls_manifest.csv")
    for dataset, (tag, _, _) in DATASETS.items():
        row = man[(man["dataset"] == dataset) & (man["seed"] == 42)].iloc[0]
        mcc[tag]["lora_cls"] = float(row["test_mcc"])

    fig = draw(layouts(root, refit=refit), mcc)
    overlaps = bbox_report(fig)
    if overlaps:
        print(f"WARNING overlapping text: {overlaps}")

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(outdir / f"lora_cls_umap.{ext}")
    print(f"wrote {outdir}/lora_cls_umap.svg and .png")


if __name__ == "__main__":
    main()
