"""Baseline arm schematic: two featurizations, no gLM, identical head.

Draws the bag-of-tokens floor -- k-mer counting and DNABERT-2's own BPE
tokenizer, both feeding the same trained head. Needs no results files.

Usage:
    python scripts/fig_baseline_bag.py [OUTDIR]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import schem as S  # noqa: E402

NAME = "baseline_bag"
S_LAB, S_DET, S_TITLE = 10.0, 8.5, 12.5


def draw():
    S.use()
    fig, ax = S.canvas(9.0, 4.4)

    S.label(ax, 3.0, 41.0,
            "The bag-of-tokens floor: position-invariant counts, no gLM",
            size=S_TITLE, ha="left", weight="bold", color=S.INK)

    UP, LO = 32.0, 17.0          # the two featurization routes
    BH = 7.2

    # --- sequence --------------------------------------------------------
    S.box(ax, 3.0, 24.5 - BH / 2, 11.5, BH, "GUE\nsequence",
          S.DATA_FILL, S.DATA_EDGE, txt=S.DATA_TXT, size=S_LAB, r=0.7)

    # --- route A: k-mer counting -----------------------------------------
    S.box(ax, 19.0, UP - BH / 2, 22.0, BH, "", S.FROZEN_FILL, S.FROZEN_EDGE, r=0.7)
    S.label(ax, 30.0, UP + 1.5, "$k$-mer counting", size=S_LAB,
            color=S.FROZEN_TXT, weight="bold")
    S.label(ax, 30.0, UP - 1.8, "$k \\in \\{3, 4, 5, 6\\}$", size=S_DET,
            color=S.FROZEN_TXT)
    S.box(ax, 48.0, UP - BH / 2, 17.0, BH, "", S.DATA_FILL, S.DATA_EDGE, r=0.7)
    S.label(ax, 56.5, UP + 1.6, "$4^k$ frequencies", size=S_DET,
            color=S.DATA_TXT, weight="bold")
    S.label(ax, 56.5, UP - 1.7, "64 / 256 / 1,024 / 4,096-d", size=S_DET,
            color=S.DATA_TXT)

    # --- route B: BPE counting -------------------------------------------
    S.box(ax, 19.0, LO - BH / 2, 22.0, BH, "", S.FROZEN_FILL, S.FROZEN_EDGE, r=0.7)
    S.label(ax, 30.0, LO + 1.5, "DNABERT-2 BPE tokenizer", size=S_DET,
            color=S.FROZEN_TXT, weight="bold")
    S.label(ax, 30.0, LO - 1.8, "4,096-token vocabulary", size=S_DET,
            color=S.FROZEN_TXT)
    S.label(ax, 30.0, LO - BH / 2 - 2.0,
            "tokenizer only $-$ the encoder is never run",
            size=S_DET, color=S.TRAIN_TXT, va="top")
    S.box(ax, 48.0, LO - BH / 2, 17.0, BH, "", S.DATA_FILL, S.DATA_EDGE, r=0.7)
    S.label(ax, 56.5, LO + 1.6, "token frequencies", size=S_DET,
            color=S.DATA_TXT, weight="bold")
    S.label(ax, 56.5, LO - 1.7, "4,096-d", size=S_DET, color=S.DATA_TXT)

    # --- shared head ------------------------------------------------------
    hx, hy, hw, hh = 70.0, 13.4, 16.0, 22.2
    S.box(ax, hx, hy, hw, hh, "", S.TRAIN_FILL, S.TRAIN_EDGE, r=0.7)
    # Lato is narrower and sits higher than the DejaVu fallback these offsets
    # were first tuned against, so the caption block needs more clearance under
    # the bold label than the original -3.4 / -7.0 pair gave it.
    S.label(ax, hx + hw / 2, hy + hh - 3.0, "same head", size=S_LAB,
            color=S.TRAIN_TXT, weight="bold")
    S.label(ax, hx + hw / 2, hy + hh - 7.9,
            "the only trained weights,\nidentical protocol\nto the frozen arms",
            size=S_DET, color=S.TRAIN_TXT)
    S.box(ax, hx + 1.7, hy + 6.3, hw - 3.4, 3.6, "logistic regression",
          S.TRAIN_FILL, S.TRAIN_EDGE, txt=S.TRAIN_TXT, size=S_DET, r=0.4, lw=1.0)
    S.box(ax, hx + 1.7, hy + 1.9, hw - 3.4, 3.6, "2-layer MLP",
          S.TRAIN_FILL, S.TRAIN_EDGE, txt=S.TRAIN_TXT, size=S_DET, r=0.4, lw=1.0)

    # --- flow -------------------------------------------------------------
    S.arrow(ax, 14.9, 25.4, 18.6, UP - 1.2, lw=1.4, ms=10)
    S.arrow(ax, 14.9, 23.6, 18.6, LO + 1.2, lw=1.4, ms=10)
    for y in (UP, LO):
        S.arrow(ax, 41.4, y, 47.6, y, lw=1.4, ms=10)
        S.arrow(ax, 65.4, y, hx - 0.4, y, lw=1.4, ms=10)

    # --- the cardinality coincidence --------------------------------------
    S.label(ax, 3.0, 6.6,
            "$4^6 = 4{,}096$: the 6-mer bag and DNABERT-2's entire vocabulary "
            "have identical cardinality, so the BPE bag tests segmentation, "
            "not size.",
            size=S_DET, ha="left", color=S.INK)
    S.label(ax, 3.0, 3.2,
            "5 feature sets $\\times$ 26 datasets   |   frequencies are "
            "L1-normalized per sequence, so sequence length is not smuggled in "
            "as a feature",
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
