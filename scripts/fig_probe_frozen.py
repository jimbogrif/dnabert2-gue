"""Frozen-probe arm schematic: every layer is a candidate, only the head trains.

Draws the 12 frozen blocks, the hidden-state matrix at the zoomed layer, the two
pooling routes read off it, and the trained probe head. Needs no results files.

Usage:
    python scripts/fig_probe_frozen.py [OUTDIR]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import schem as S  # noqa: E402

NAME = "probe_frozen"
S_LAB, S_DET, S_TITLE = 10.0, 8.5, 12.5


def draw():
    S.use()
    fig, ax = S.canvas(9.0, 4.4)

    S.label(ax, 3.0, 41.0,
            "Frozen embeddings extracted at each layer and a simple trained head",
            size=S_TITLE, ha="left", weight="bold", color=S.INK)

    # ================== left: the 12-block stack ==========================
    sx, sw, bh, gap = 4.0, 15.0, 1.65, 0.45
    y0, hot = 11.5, 6
    for i in range(12):
        yy = y0 + i * (bh + gap)
        hl = (i == hot)
        S.box(ax, sx, yy, sw, bh, "", S.FROZEN_FILL,
              S.HILITE if hl else S.FROZEN_EDGE, r=0.3, lw=1.9 if hl else 1.0)
    S.label(ax, sx + sw / 2, y0 + 12 * (bh + gap) + 1.5, "12 blocks, all frozen",
            size=S_LAB, color=S.FROZEN_TXT, weight="bold")
    hy = y0 + hot * (bh + gap)
    S.label(ax, sx + sw / 2, hy + bh / 2, "layer $\\ell$", size=7.5,
            color=S.HILITE, weight="bold")
    S.wedge(ax, (sx + sw, hy), (sx + sw, hy + bh), (31.0, 33.0), (31.0, 13.0),
            alpha=0.12)

    # ================== right: what comes out of layer l ==================
    PX = 31.0
    S.box(ax, PX, 10.0, 57.0, 24.5, "", "none", S.HILITE, r=0.8, lw=1.2,
          style=(0, (4, 2)), zorder=0)

    # --- hidden-state matrix: columns are token positions -----------------
    mx, my, cw, ch, cg = 34.0, 18.6, 1.45, 1.45, 0.28
    ncol, nrow = 7, 5
    MW = ncol * (cw + cg) - cg
    MH = nrow * (ch + cg) - cg
    for c in range(ncol):
        for r_ in range(nrow):
            first = (c == 0)
            S.box(ax, mx + c * (cw + cg), my + r_ * (ch + cg), cw, ch, "",
                  S.FROZEN_FILL if first else S.DATA_FILL,
                  S.FROZEN_EDGE if first else S.DATA_EDGE,
                  r=0.12, lw=1.5 if first else 0.8)
    S.label(ax, mx + MW / 2, 32.6, "hidden states at layer $\\ell$", size=S_LAB,
            color=S.FROZEN_TXT, weight="bold")
    S.label(ax, mx + MW / 2, my - 2.0, "tokens $\\times$ 768", size=S_DET,
            color=S.INK)

    # --- pooling routes (labels inside the boxes, clear of the arrows) -----
    bx2, bw2, bh2 = 50.0, 13.0, 5.6
    cls_y, mean_y = 30.2, 15.2
    S.box(ax, bx2, cls_y - bh2 / 2, bw2, bh2,
          "[CLS] pooling\nposition 0 $\\cdot$ 768-d",
          S.DATA_FILL, S.DATA_EDGE, txt=S.DATA_TXT, size=S_DET, r=0.5)
    S.box(ax, bx2, mean_y - bh2 / 2, bw2, bh2,
          "mean pooling\nall positions $\\cdot$ 768-d",
          S.DATA_FILL, S.DATA_EDGE, txt=S.DATA_TXT, size=S_DET, r=0.5)
    S.arrow(ax, mx + cw / 2, my + MH + 0.3, bx2, cls_y, color=S.DATA_EDGE,
            lw=1.3, ms=9, conn="angle,angleA=90,angleB=0,rad=2.5")
    S.arrow(ax, mx + MW / 2, my - 3.0, bx2, mean_y, color=S.DATA_EDGE,
            lw=1.3, ms=9, conn="angle,angleA=-90,angleB=0,rad=2.5")

    # --- trained head -----------------------------------------------------
    hx, hy2, hw2, hh2 = 68.0, 12.4, 15.0, 20.4
    S.box(ax, hx, hy2, hw2, hh2, "", S.TRAIN_FILL, S.TRAIN_EDGE, r=0.7)
    S.label(ax, hx + hw2 / 2, hy2 + hh2 - 3.2, "probe head", size=S_LAB,
            color=S.TRAIN_TXT, weight="bold")
    S.label(ax, hx + hw2 / 2, hy2 + hh2 - 6.6,
            "the only trained\nweights in this arm", size=S_DET,
            color=S.TRAIN_TXT)
    S.box(ax, hx + 1.6, hy2 + 5.8, hw2 - 3.2, 3.6, "logistic regression",
          S.TRAIN_FILL, S.TRAIN_EDGE, txt=S.TRAIN_TXT, size=S_DET, r=0.4, lw=1.0)
    S.box(ax, hx + 1.6, hy2 + 1.5, hw2 - 3.2, 3.6, "2-layer MLP",
          S.TRAIN_FILL, S.TRAIN_EDGE, txt=S.TRAIN_TXT, size=S_DET, r=0.4, lw=1.0)
    S.arrow(ax, bx2 + bw2 + 0.4, cls_y, hx - 0.4, cls_y, lw=1.3, ms=9)
    S.arrow(ax, bx2 + bw2 + 0.4, mean_y, hx - 0.4, mean_y, lw=1.3, ms=9)

    # ================== strips ============================================
    S.label(ax, 3.0, 7.0, "Cached:", size=S_LAB, ha="left", weight="bold",
            color=S.INK)
    S.label(ax, 12.5, 7.0,
            "12 layers $\\times$ 2 poolings $\\times$ 26 datasets   |   MLP arm "
            "ran 3 layers per task: top 2 by dev MCC, plus layer 12",
            size=S_DET, ha="left", color=S.INK)
    S.label(ax, 3.0, 4.2, "Heads:", size=S_LAB, ha="left", weight="bold",
            color=S.INK)
    S.label(ax, 12.5, 4.2,
            "logistic regression over $C \\in$ logspace($-5$, 3, 9) $\\times$ "
            "class_weight $\\in$ {none, balanced}, scaler fit on train only",
            size=S_DET, ha="left", color=S.TRAIN_TXT)
    S.label(ax, 12.5, 1.6,
            "MLP 768$\\rightarrow$256$\\rightarrow$2, dropout 0.1, AdamW lr 1e-3, "
            "batch 32, $\\leq$200 epochs with patience 10, 3 seeds",
            size=S_DET, ha="left", color=S.TRAIN_TXT)
    return fig


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "figures"
    fig = draw()
    written = S.save(fig, NAME, outdir)
    print(S.bbox_report(fig))
    print("wrote " + ", ".join(written))


if __name__ == "__main__":
    main()
