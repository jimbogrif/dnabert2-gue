"""LoRA arm schematic: the low-rank detour inside one attention block.

Draws the 12 adapter-carrying blocks, then zooms one to show x splitting across
the frozen fused Wqkv and the trained rank-8 pair before the sum. Parameter
counts in the footer are the measured values from ``dnabert2_gue.lora`` for the
binary case. Needs no results files.

Usage:
    python scripts/fig_lora_adapter.py [OUTDIR]
"""

import os
import sys

from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import schem as S  # noqa: E402

NAME = "lora_v3_adapter"
S_LAB, S_DET, S_TITLE = 10.0, 8.5, 12.5


def draw():
    S.use()
    fig, ax = S.canvas(9.0, 4.4)

    S.label(ax, 3.0, 41.0,
            "Inside a block: a rank-8 detour around the frozen attention "
            "projection",
            size=S_TITLE, ha="left", weight="bold", color=S.INK)

    # ================== left: the 12-block stack ==========================
    sx, sw, bh, gap = 4.0, 15.0, 1.65, 0.45
    y0, hot = 9.5, 6                            # hot = the block we zoom into
    for i in range(12):
        yy = y0 + i * (bh + gap)
        hl = (i == hot)
        S.box(ax, sx, yy, sw, bh, "", S.FROZEN_FILL,
              S.HILITE if hl else S.FROZEN_EDGE, r=0.3, lw=1.9 if hl else 1.0)
        S.box(ax, sx + sw - 4.2, yy + 0.42, 3.0, bh - 0.84, "", S.TRAIN_FILL,
              S.TRAIN_EDGE, r=0.2, lw=0.9, zorder=3)
    S.label(ax, sx + sw / 2, y0 + 12 * (bh + gap) + 1.4, "12 blocks", size=S_LAB,
            color=S.FROZEN_TXT, weight="bold")

    hy = y0 + hot * (bh + gap)
    S.wedge(ax, (sx + sw, hy), (sx + sw, hy + bh), (31.0, 34.5), (31.0, 9.0),
            alpha=0.12)

    # ================== right: inside that block ==========================
    PX = 31.0
    S.box(ax, PX, 8.0, 57.0, 26.5, "", "none", S.HILITE, r=0.8, lw=1.2,
          style=(0, (4, 2)), zorder=0)

    YIN = 22.0
    S.box(ax, PX + 2.0, YIN - 2.2, 9.0, 4.4, "$x$\n768-d", S.DATA_FILL,
          S.DATA_EDGE, txt=S.DATA_TXT, size=S_DET, r=0.5)

    # --- frozen path (upper) ----------------------------------------------
    YF = 29.0
    S.box(ax, PX + 17.0, YF - 2.4, 20.0, 4.8, "$W_{qkv}$   768 $\\times$ 2304",
          S.FROZEN_FILL, S.FROZEN_EDGE, txt=S.FROZEN_TXT, size=S_LAB, r=0.5)
    S.label(ax, PX + 27.0, YF + 3.6, "frozen   1.77M weights per block",
            size=S_DET, color=S.FROZEN_TXT)

    # --- LoRA path (lower) -------------------------------------------------
    YL = 16.3
    S.box(ax, PX + 17.0, YL - 2.2, 8.5, 4.4, "$A$\n768 $\\times$ 8",
          S.TRAIN_FILL, S.TRAIN_EDGE, txt=S.TRAIN_TXT, size=S_DET, r=0.5)
    S.box(ax, PX + 28.5, YL - 2.2, 8.5, 4.4, "$B$\n8 $\\times$ 2304",
          S.TRAIN_FILL, S.TRAIN_EDGE, txt=S.TRAIN_TXT, size=S_DET, r=0.5)
    S.arrow(ax, PX + 25.5, YL, PX + 28.5, YL, color=S.TRAIN_EDGE, lw=1.3, ms=9)
    S.label(ax, PX + 27.0, 12.4, "trained   24,576 weights per block",
            size=S_DET, color=S.TRAIN_TXT)
    S.label(ax, PX + 27.0, 10.0,
            "rank 8, dropout 0.05, scaled by $\\alpha/r = 2$",
            size=S_DET, color=S.TRAIN_TXT)

    # --- split from x -------------------------------------------------------
    S.arrow(ax, PX + 11.4, YIN + 1.0, PX + 27.0, YF - 2.4, lw=1.3, ms=9,
            conn="angle,angleA=0,angleB=90,rad=2")
    S.arrow(ax, PX + 11.4, YIN - 1.0, PX + 21.25, YL + 2.2, lw=1.3, ms=9,
            conn="angle,angleA=0,angleB=-90,rad=2")

    # --- merge --------------------------------------------------------------
    MX = PX + 42.5
    ax.add_patch(Circle((MX, YIN), 1.75, facecolor="white", edgecolor=S.INK,
                        lw=1.3, zorder=4))
    S.label(ax, MX, YIN + 0.05, "+", size=13, color=S.INK, zorder=5)
    S.arrow(ax, PX + 37.0, YF, MX, YIN + 1.9, lw=1.3, ms=9,
            conn="angle,angleA=0,angleB=90,rad=3")
    S.arrow(ax, PX + 37.0, YL, MX, YIN - 1.9, lw=1.3, ms=9,
            conn="angle,angleA=0,angleB=-90,rad=3")

    # --- output -------------------------------------------------------------
    S.box(ax, PX + 47.0, YIN - 2.2, 8.5, 4.4, "$q,k,v$\n2304-d", S.DATA_FILL,
          S.DATA_EDGE, txt=S.DATA_TXT, size=S_DET, r=0.5)
    S.arrow(ax, MX + 1.9, YIN, PX + 47.0, YIN, lw=1.3, ms=9)

    # ================== accounting strip ==================================
    S.label(ax, 3.0, 4.0, "Trainable:", size=S_LAB, ha="left", weight="bold",
            color=S.INK)
    S.label(ax, 12.5, 4.0,
            "12 $\\times$ 24,576 = 294,912 adapter   +   590,592 pooler   +   "
            "1,538 classifier   =   887,042 of 117,957,124   (0.752%)",
            size=S_DET, ha="left", color=S.INK)
    return fig


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "figures"
    fig = draw()
    written = S.save(fig, NAME, outdir)
    print(S.bbox_report(fig))
    print("wrote " + ", ".join(written))


if __name__ == "__main__":
    main()
