"""Dot-strip plot: Monarch projection error of trained weights relative to an i.i.d. Gaussian matrix.

Inputs are the TSVs written by relerr_vs_gaussian.py (columns model, layer, group, mtype, rel_err, gauss_rel_err,
ratio): either Moonlight (Muon) and Moonlight-A (AdamW) at 4 blocks, or Qwen2.5-7B at 2 blocks alone.

Usage:
  python plots/plot_relerr_strip.py --muon relerr_muon.tsv --adamw relerr_adamw.tsv --out relerr_strip
  python plots/plot_relerr_strip.py --qwen relerr_qwen.tsv --out relerr_strip_qwen
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D


COLOR = {"adamw": "#2a78d6", "muon": "#eb6834", "qwen": "#1baf7a"}
MARKER = {"adamw": "o", "muon": "s", "qwen": "^"}
INK, MUTED, RULE, GRID = "#1a1a1a", "#555555", "#bdbdbd", "#ebebeb"

GROUPS = [  # (group key in the TSV, label in the figure, matrix types top to bottom)
    ("attn", "Attention\n(MLA)", ["q", "kv_a", "kv_b", "o"]),
    ("dense FFN", "Layer-0\ndense FFN", ["gate", "up", "down"]),
    ("routed experts", "Routed\nexperts", ["gate", "up", "down"]),
    ("shared experts", "Shared\nexperts", ["gate", "up", "down"]),
]
QWEN_GROUP = ("qwen", "Qwen2.5-7B\n(dense,\n2 blocks)", ["q", "k", "v", "o", "gate", "up", "down"])

ROW, QROW, GAP = 1.0, 0.9, 0.4   # row heights (one per matrix type) and gap between groups, in row units
DODGE = 0.25                       # AdamW above, Muon below the row centre
UNIT_IN = 0.13                     # inches per row unit
TOP_IN, BOT_IN, LEFT_IN, RIGHT_IN = 0.52, 0.36, 0.80, 0.07
XLIM = (0.5, 1.035)  # the left end extends if any value falls below it


def load_tsv(path, opt):
    """{(group, mtype, opt): [ratio per layer]}; Qwen rows (attention and dense FFN alike) go to the group "qwen"."""
    vals = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            assert abs(float(r["ratio"]) - float(r["rel_err"]) / float(r["gauss_rel_err"])) < 1e-4, r
            g = "qwen" if opt == "qwen" else r["group"]
            vals.setdefault((g, r["mtype"], opt), []).append(float(r["ratio"]))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muon", type=Path, help="relerr_vs_gaussian.py TSV of Moonlight (Muon).")
    ap.add_argument("--adamw", type=Path, help="relerr_vs_gaussian.py TSV of Moonlight-A (AdamW).")
    ap.add_argument("--qwen", type=Path, help="relerr_vs_gaussian.py TSV of Qwen2.5-7B (2 blocks); plotted alone.")
    ap.add_argument("--out", default="relerr_strip", help="Output path without extension (.pdf and .png are written).")
    a = ap.parse_args()

    if a.qwen:
        vals, groups = load_tsv(a.qwen, "qwen"), [QWEN_GROUP]
    else:
        vals = {**load_tsv(a.muon, "muon"), **load_tsv(a.adamw, "adamw")}
        groups = list(GROUPS)
    out = a.out

    # Layout in row units, y grows downward.
    rows, spans, y = [], [], 0.0
    for gi, (g, glabel, mtypes) in enumerate(groups):
        if gi:
            y += GAP
        h = QROW if g == "qwen" else (0.75 if g == "dense FFN" else ROW)
        y0 = y
        for m in mtypes:
            rows.append((g, m, y + h / 2, h))
            y += h
        spans.append((glabel, y0, y))
    total = y

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "xtick.major.size": 2.5,
        "xtick.color": MUTED, "xtick.labelcolor": INK, "pdf.fonttype": 42,
    })
    W = 3.25
    H = TOP_IN + total * UNIT_IN + BOT_IN
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([LEFT_IN / W, BOT_IN / H, 1 - (LEFT_IN + RIGHT_IN) / W, total * UNIT_IN / H])
    lo = min(XLIM[0], min(min(v) for v in vals.values()) - 0.03)
    ticks = np.round(np.arange(np.ceil(lo * 10) / 10, 1.001, 0.1), 1)
    ax.set_xlim(lo, XLIM[1])
    ax.set_ylim(total, 0)

    for x in ticks[:-1]:
        ax.axvline(x, color=GRID, lw=0.5, zorder=0)
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)

    rng = np.random.default_rng(1)
    for g, m, yc, h in rows:
        opts = ["qwen"] if g == "qwen" else ["adamw", "muon"]
        for opt in opts:
            v = np.array(vals[(g, m, opt)])
            yo = yc + {"adamw": -DODGE, "muon": DODGE, "qwen": 0.0}[opt] * h
            jit = rng.uniform(-0.09, 0.09, len(v)) * h if len(v) > 1 else 0
            if opt == "muon":  # open squares keep Muon distinct from AdamW in grayscale and in dense clusters
                ax.plot(v, yo + jit, ls="none", marker="s", ms=3.0, mfc="white", mec=COLOR[opt], mew=0.7,
                        alpha=0.9, zorder=2)
            else:
                ax.plot(v, yo + jit, ls="none", marker=MARKER[opt], ms=3.2 if opt != "qwen" else 3.6,
                        mfc=COLOR[opt], mec="white", mew=0.35, alpha=0.8, zorder=2)
            if len(v) > 1:
                ax.plot([np.median(v)] * 2, [yo - 0.26 * h, yo + 0.26 * h], color=INK, lw=1.1,
                        solid_capstyle="butt", zorder=3,
                        path_effects=[patheffects.withStroke(linewidth=2.1, foreground="white")])

    ax.set_yticks([r[2] for r in rows], [r[1] for r in rows])
    ax.tick_params(axis="y", length=0, pad=3, labelcolor=INK)
    ax.set_xticks(ticks, [f"{t:.1f}" for t in ticks])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("Projection error / Gaussian-matrix error", color=INK, labelpad=2)
    # Direct labels above the plot: reading direction on the left, the Gaussian reference on the right.
    ax.annotate("more Monarch structure", xy=(lo, -0.55), xytext=(lo + 0.03, -0.55),
                ha="left", va="center", color=INK, annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=MUTED, lw=0.6,
                                shrinkA=0, shrinkB=0))
    ax.plot([1.0, 1.0], [0, -1.1], color=MUTED, lw=0.8, ls=(0, (3, 2)), clip_on=False)
    ax.text(0.993, -0.95, "i.i.d. Gaussian, same shape", ha="right", va="bottom", color=INK)

    # Group labels in the left column and thin separators between groups.
    for gi, (glabel, y0, y1) in enumerate(spans):
        fig.text(0.02 / W, BOT_IN / H + (total - (y0 + y1) / 2) * UNIT_IN / H, glabel, ha="left", va="center",
                 color=INK, linespacing=1.05)
        if gi:
            yy = BOT_IN / H + (total - (y0 - GAP / 2)) * UNIT_IN / H
            fig.add_artist(Line2D([0.02 / W, 1 - RIGHT_IN / W], [yy, yy], color=RULE, lw=0.5))

    handles = [] if a.qwen else [
        Line2D([], [], ls="none", marker="o", ms=3.6, mfc=COLOR["adamw"], mec="white", mew=0.35, label="AdamW (Moonlight-A)"),
        Line2D([], [], ls="none", marker="s", ms=3.3, mfc="white", mec=COLOR["muon"], mew=0.7, label="Muon (Moonlight)"),
    ]
    if a.qwen:
        handles.append(Line2D([], [], ls="none", marker="^", ms=4, mfc=COLOR["qwen"], mec="white", mew=0.35,
                              label="Qwen2.5-7B (AdamW)"))
    handles.append(Line2D([], [], ls="none", marker="|", ms=5.5, mew=1.1, color=INK, label="median over layers"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
               handletextpad=0.3, columnspacing=1.0, borderaxespad=0.1, labelcolor=INK, handlelength=1.2)

    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=300)
    print(f"{out}: {W:.2f} x {H:.2f} in")


if __name__ == "__main__":
    main()
