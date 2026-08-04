#!/usr/bin/env python3
"""Slide diagram for the statistical (POCV) kernel, drawn to sit beside 00_problem.

Deliberately the same shape as draw_problem.py so the two can be shown back to
back: same rows, same add, same max. The only difference is that every value is
a (mean, sigma) pair instead of a single number -- which is the entire point,
because that is why 8x the arithmetic only bought 2x the intensity.

    python3 draw_problem_ssta.py    # writes figures/08_problem_ssta.png (+ dark)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")

FANIN = 8
CELL_W, CELL_H, GAP = 0.86, 0.92, 0.06

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  cpu="#2a78d6", gpu="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 cpu="#3987e5", gpu="#d95926"),
}


def pair_strip(ax, x0, y, n, color, t, alpha=1.0, labels=True):
    """A row of n boxes, each split into a mean half and a sigma half."""
    for i in range(n):
        x = x0 + i * (CELL_W + GAP)
        ax.add_patch(FancyBboxPatch((x, y), CELL_W, CELL_H,
                                    boxstyle="round,pad=0,rounding_size=0.10",
                                    linewidth=0, facecolor=color, alpha=alpha, zorder=3))
        # hairline in the surface colour splitting mean (top) from sigma (bottom)
        ax.plot([x + 0.06, x + CELL_W - 0.06], [y + CELL_H / 2] * 2,
                color=t["surface"], linewidth=1.6, zorder=4)
        if labels:
            ax.text(x + CELL_W / 2, y + CELL_H * 0.75, r"$\mu$", ha="center", va="center",
                    color=t["surface"], fontsize=11, fontweight="bold", zorder=5)
            ax.text(x + CELL_W / 2, y + CELL_H * 0.25, r"$\sigma$", ha="center", va="center",
                    color=t["surface"], fontsize=11, fontweight="bold", zorder=5)
    return x0, x0 + n * (CELL_W + GAP) - GAP


def draw(theme):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(13.0, 7.8), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.set_xlim(-2.6, 11.4)
    ax.set_ylim(-1.9, 9.2)
    ax.axis("off")

    x0 = 0.35
    y_arr, y_del, y_cand, y_out = 5.55, 3.95, 1.95, 0.05

    def row_label(y, text, sub, color):
        ax.text(x0 - 0.30, y + CELL_H / 2 + 0.13, text, ha="right", va="center",
                color=color, fontsize=13, fontweight="bold")
        ax.text(x0 - 0.30, y + CELL_H / 2 - 0.28, sub, ha="right", va="center",
                color=t["muted"], fontsize=10)

    left, right = pair_strip(ax, x0, y_arr, FANIN, t["cpu"], t)
    row_label(y_arr, "arrival_in", f"{FANIN} fanin edges of node $i$", t["cpu"])
    pair_strip(ax, x0, y_del, FANIN, t["gpu"], t)
    row_label(y_del, "delay", "the cell arc", t["gpu"])

    # combine, column by column
    y_op = (y_arr + y_del + CELL_H) / 2
    for i in range(FANIN):
        cx = x0 + i * (CELL_W + GAP) + CELL_W / 2
        ax.text(cx, y_op, "+", ha="center", va="center", color=t["ink2"], fontsize=16,
                fontweight="bold")
        ax.annotate("", xy=(cx, y_cand + CELL_H + 0.12), xytext=(cx, y_del - 0.08),
                    arrowprops=dict(arrowstyle="-|>", color=t["muted"], linewidth=1.1))
    ax.text(right + 0.32, y_op + 0.16, r"means add:   $\mu = \mu_{in} + \mu_d$",
            ha="left", va="center", color=t["ink2"], fontsize=11.5)
    ax.text(right + 0.32, y_op - 0.34,
            r"sigmas in quadrature:   $\sigma = \sqrt{\sigma_{in}^2 + \sigma_d^2}$",
            ha="left", va="center", color=t["ink2"], fontsize=11.5)

    pair_strip(ax, x0, y_cand, FANIN, t["muted"], t, alpha=0.55)
    row_label(y_cand, "candidates", "one per fanin edge", t["ink2"])

    # the max, now over a ranking key rather than the value itself
    ax.add_patch(FancyArrowPatch((left, y_cand - 0.26), (right, y_cand - 0.26),
                                 arrowstyle="|-|,widthA=3,widthB=3", color=t["ink2"],
                                 linewidth=1.4, shrinkA=0, shrinkB=0))
    mid = (left + right) / 2
    ax.text(mid, y_cand - 0.78, r"keep the $k$ that maximises  $\mu_k + 3\sigma_k$",
            ha="center", va="center", color=t["ink"], fontsize=14, fontweight="bold")
    ax.annotate("", xy=(mid, y_out + CELL_H + 0.12), xytext=(mid, y_cand - 1.08),
                arrowprops=dict(arrowstyle="-|>", color=t["ink2"], linewidth=1.8))

    ox = mid - CELL_W / 2
    pair_strip(ax, ox, y_out, 1, t["cpu"], t)
    ax.text(ox - 0.30, y_out + CELL_H / 2, "arrival", ha="right", va="center",
            color=t["cpu"], fontsize=13, fontweight="bold")
    ax.text(ox + CELL_W + 0.35, y_out + CELL_H / 2,
            "one node's arrival distribution\n= one work item = one GPU thread",
            ha="left", va="center", color=t["ink2"], fontsize=11.5)

    ax.text(-2.5, 9.05,
            r"$\mu[i],\ \sigma[i] \;=\; \mathrm{the}\ (\mu_k,\sigma_k)\ "
            r"\mathrm{that\ maximises}\ \ \mu_k + 3\sigma_k$",
            ha="left", va="top", color=t["ink"], fontsize=19)
    ax.text(-2.5, 8.05,
            "Statistical (POCV) propagation, as in INSTA. Same shape as block-based STA — "
            "an add, then a max over the fanin window —\nbut every value is a "
            "(mean, sigma) pair, and the whole thing runs twice: once for rise, once for "
            "fall. An inverting arc swaps them.",
            ha="left", va="top", color=t["ink2"], fontsize=12)

    ax.text(-2.5, -1.30,
            "Per node: 8 floats + a sense byte per edge = 280 bytes, 126 flops  "
            r"$\approx 0.45$ flops/byte",
            ha="left", va="bottom", color=t["ink"], fontsize=12.5, fontweight="bold")
    ax.text(-2.5, -1.62,
            "8× the arithmetic of block STA, but 4× the data — so the ratio only moved from "
            "0.22 to 0.45. Still memory-bandwidth bound.",
            ha="left", va="top", color=t["ink2"], fontsize=11.5)

    out = FIGS if theme == "light" else os.path.join(FIGS, "dark")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "08_problem_ssta.png")
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    for th in ("light", "dark"):
        draw(th)
