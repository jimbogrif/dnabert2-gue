"""Shared drawing primitives for the DNABERT-2 methods schematics.

Each arm has its own script -- ``fig_baseline_bag.py``, ``fig_probe_frozen.py``,
``fig_lora_adapter.py`` -- so a figure can be added, re-drawn or dropped without
touching the others. This module holds only what all of them share.

Colors come from ``blogfig.SCHEMATIC``, the blog's three-way block-diagram
language, and never from literals:

    frozen   weights that do not update
    trained  the small parts that do
    flow     inputs, outputs and the arrows between blocks

That reading is what makes the arm figures legible side by side, and it only
works if all three use it consistently. ``blogfig`` is the canonical brand
package; it deliberately lives outside this repo so one palette edit keeps every
past figure consistent with every future one. Set ``BLOGFIG_HOME`` if it is not
at /home/jim/code/blogfig-brand.

Canvas units are 0.1 inch with equal aspect, so a 9.0 x 4.4 in figure is a
90 x 44 coordinate box and every size below reads directly as tenths of an inch.
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

BLOGFIG_HOME = os.environ.get("BLOGFIG_HOME", "/home/jim/code/blogfig-brand")
if BLOGFIG_HOME not in sys.path:
    sys.path.insert(0, BLOGFIG_HOME)
try:
    import blogfig as bf
except ImportError as exc:                                   # pragma: no cover
    raise ImportError(
        f"blogfig not importable from {BLOGFIG_HOME!r}. It is the blog's brand "
        "package and lives outside this repo; clone it and set BLOGFIG_HOME."
    ) from exc

_S = bf.SCHEMATIC
_N = bf.NEUTRAL

# --- the three-way block-diagram language --------------------------------
FROZEN_FILL, FROZEN_EDGE, FROZEN_TXT = _S["frozen_fill"], _S["frozen"], _S["frozen"]
TRAIN_FILL, TRAIN_EDGE, TRAIN_TXT = _S["trained_fill"], _S["trained"], _S["trained"]
DATA_FILL, DATA_EDGE = _S["flow_fill"], _S["flow"]

# Text on a flow-colored box takes the darker frozen navy rather than ``flow``
# itself: #2C73CC on #D9E8FB measures 3.81:1, under the 4.5:1 floor for the
# 8.5 pt detail type these figures use, where #003488 on #D9E8FB gives 9.14:1.
# ``flow`` still owns the outlines and arrows, so the three-way reading holds.
DATA_TXT = _S["frozen"]

# ``trained`` text on ``trained_fill`` is fine to use directly: it measures
# 4.94:1 since blogfig moved the pair to #A8450F on #FFE6C7. The earlier
# #E6712A on #F6C8A8 was 2.03:1, so any figure predating that palette change
# needs re-rendering rather than trusting its committed SVG.

# --- furniture (never data) ----------------------------------------------
INK = _N["ink"]            # titles, footer prose
MUTED = _N["muted"]        # secondary annotation
HILITE = _S["flow"]        # the zoomed block's outline and its wedge

use = bf.use               # re-exported: call once before creating a figure


def canvas(w, h, dpi=200):
    """Full-bleed axes in units of 0.1 inch, equal aspect, no decoration."""
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10 * w)
    ax.set_ylim(0, 10 * h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, label, fill, edge, txt=None, size=10, weight="normal",
        r=1.6, lw=1.3, va="center", ha="center", zorder=2, alpha=1.0, style=None):
    """Rounded block. ``label`` centered unless ``ha='left'`` insets it."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fill, edgecolor=edge, linewidth=lw,
                       zorder=zorder, alpha=alpha, linestyle=style or "solid")
    ax.add_patch(p)
    if label:
        tx = x + w / 2 if ha == "center" else x + 1.6
        ax.text(tx, y + h / 2, label, ha=ha, va=va, fontsize=size,
                color=txt or INK, weight=weight, zorder=zorder + 1,
                linespacing=1.45)
    return p


def arrow(ax, x0, y0, x1, y1, color=None, lw=1.4, ms=9, style="-|>", zorder=1,
          conn="arc3,rad=0", ls="solid"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                        mutation_scale=ms, color=color or INK, linewidth=lw,
                        connectionstyle=conn, zorder=zorder, linestyle=ls,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def label(ax, x, y, s, size=9, color=None, ha="center", va="center",
          weight="normal", style="normal", zorder=5):
    return ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color or INK,
                   weight=weight, style=style, zorder=zorder, linespacing=1.45)


def legend_swatch(ax, x, y, items, size=9, sw=2.4, gap=14.0):
    """items: [(label, fill, edge), ...] laid out left to right from x."""
    cx = x
    for text, fill, edge in items:
        ax.add_patch(FancyBboxPatch((cx, y - sw / 2), sw, sw,
                     boxstyle="round,pad=0,rounding_size=0.5",
                     facecolor=fill, edgecolor=edge, linewidth=1.2, zorder=4))
        ax.text(cx + sw + 1.1, y, text, ha="left", va="center",
                fontsize=size, color=INK, zorder=4)
        cx += gap
    return cx


def wedge(ax, p0, p1, p2, p3, color=None, alpha=0.13):
    """Zoom wedge from a block in the stack out to the detail panel."""
    ax.add_patch(Polygon([p0, p1, p2, p3], closed=True,
                         facecolor=color or HILITE, edgecolor="none",
                         alpha=alpha, zorder=0))


def save(fig, name, outdir):
    """Write ``name``.svg and .png into ``outdir``.

    ``pad_inches=0`` is the one savefig argument worth passing here. The style
    sheet sets ``savefig.bbox: tight``, and ``bbox_inches=None`` does NOT turn
    that off -- None is the sentinel that means "use the rcParam". On a
    full-bleed axes the tight box resolves to the whole canvas anyway, so the
    only effect left is the default 0.1 in pad, which silently shipped these as
    9.2 x 4.6 instead of 9.0 x 4.4. Zeroing it restores the declared size.

    An overflowing label would still expand the box; ``bbox_report``'s
    ``outside`` list is what catches that.
    """
    os.makedirs(outdir, exist_ok=True)
    written = []
    for ext in ("svg", "png"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, pad_inches=0)
        written.append(p)
    return written


def _arrow_points(a, n=48):
    """Display-space sample points along an arrow's drawn path."""
    tp = a.get_transform().transform_path(a.get_path())
    v = [tuple(pt) for pt in tp.vertices
         if not (abs(pt[0]) < 1e-9 and abs(pt[1]) < 1e-9)]
    pts = []
    for (x0, y0), (x1, y1) in zip(v, v[1:]):
        for i in range(n):
            f = i / n
            pts.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
    return pts + v[-1:]


def bbox_report(fig, pad=1.5):
    """Geometric check: text-text overlap, clipping, and text-on-arrow hits.

    Complements ``blogfig``'s own check, which sees colors and fonts but not
    crowding. Lato is narrower than the DejaVu fallback at the same point size,
    so every hand-placed label needs re-checking whenever the style changes.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    ov = [(a.get_text()[:26], b.get_text()[:26])
          for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:]
          if ba.overlaps(bb)]
    out = [t.get_text()[:26] for t, bt in texts
           if not (fig.bbox.x0 <= bt.x0 and bt.x1 <= fig.bbox.x1
                   and fig.bbox.y0 <= bt.y0 and bt.y1 <= fig.bbox.y1)]
    hits = set()
    for a in fig.findobj(FancyArrowPatch):
        if not a.get_visible():
            continue
        for (px, py) in _arrow_points(a):
            for t, bt in texts:
                if (bt.x0 - pad <= px <= bt.x1 + pad
                        and bt.y0 - pad <= py <= bt.y1 + pad):
                    hits.add(t.get_text()[:30].replace("\n", " / "))
    return {"n_text": len(texts), "overlaps": ov, "outside": out,
            "text_on_arrow": sorted(hits)}
