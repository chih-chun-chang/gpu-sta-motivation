#!/usr/bin/env python3
"""Draw the problem-formulation figure: what the STA propagation kernel does.

Standalone from plot.py because it draws a diagram rather than plotting data.
Uses matplotlib's built-in mathtext, so no LaTeX installation is required.

    python3 draw_problem.py     # writes figures/00_problem.png (+ dark)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")

FANIN = 8
CELL_W, CELL_H, GAP = 0.86, 0.62, 0.06

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  cpu="#2a78d6", gpu="#eb6834", faint="#e1e0d9"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 cpu="#3987e5", gpu="#d95926", faint="#2c2c2a"),
}


def strip(ax, x0, y, n, color, t, labels=None, alpha=1.0):
    """A row of n boxes starting at x0, returning (left, right) x extent."""
    for i in range(n):
        x = x0 + i * (CELL_W + GAP)
        ax.add_patch(FancyBboxPatch(
            (x, y), CELL_W, CELL_H,
            boxstyle="round,pad=0,rounding_size=0.10",
            linewidth=0, facecolor=color, alpha=alpha, zorder=3))
        if labels:
            ax.text(x + CELL_W / 2, y + CELL_H / 2, labels[i], ha="center", va="center",
                    color=t["surface"], fontsize=11, fontweight="bold", zorder=4)
    return x0, x0 + n * (CELL_W + GAP) - GAP


def draw(theme):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(13.0, 7.4), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.set_xlim(-2.6, 11.4)
    ax.set_ylim(-2.5, 8.5)
    ax.axis("off")

    x0 = 0.35
    y_arr, y_del, y_sum, y_out = 5.35, 4.05, 2.45, 0.30
    ks = [f"$k_{i}$" for i in range(FANIN)]

    def row_label(y, text, sub, color):
        ax.text(x0 - 0.30, y + CELL_H / 2 + 0.10, text, ha="right", va="center",
                color=color, fontsize=13, fontweight="bold")
        ax.text(x0 - 0.30, y + CELL_H / 2 - 0.32, sub, ha="right", va="center",
                color=t["muted"], fontsize=10)

    # --- the two input vectors, added column by column ---------------------
    left, right = strip(ax, x0, y_arr, FANIN, t["cpu"], t, ks)
    row_label(y_arr, "arrival_in", f"{FANIN} fanin edges of node $i$", t["cpu"])
    strip(ax, x0, y_del, FANIN, t["gpu"], t, ks)
    row_label(y_del, "delay", "one per edge", t["gpu"])

    y_plus = (y_arr + y_del + CELL_H) / 2
    for i in range(FANIN):
        cx = x0 + i * (CELL_W + GAP) + CELL_W / 2
        ax.text(cx, y_plus, "+", ha="center", va="center", color=t["ink2"], fontsize=17,
                fontweight="bold")
        ax.annotate("", xy=(cx, y_sum + CELL_H + 0.12), xytext=(cx, y_del - 0.08),
                    arrowprops=dict(arrowstyle="-|>", color=t["muted"], linewidth=1.1))
    ax.text(right + 0.35, y_plus, "elementwise add", ha="left", va="center",
            color=t["ink2"], fontsize=11.5)

    strip(ax, x0, y_sum, FANIN, t["muted"], t, None, alpha=0.55)
    row_label(y_sum, "candidates", "arrival + delay, per edge", t["ink2"])

    # --- max reduction over the fanin window -------------------------------
    ax.add_patch(FancyArrowPatch((left, y_sum - 0.26), (right, y_sum - 0.26),
                                 arrowstyle="|-|,widthA=3,widthB=3", color=t["ink2"],
                                 linewidth=1.4, shrinkA=0, shrinkB=0))
    mid = (left + right) / 2
    ax.text(mid, y_sum - 0.72, r"$\max$ over the $k$ window", ha="center", va="center",
            color=t["ink"], fontsize=14, fontweight="bold")
    ax.annotate("", xy=(mid, y_out + CELL_H + 0.12), xytext=(mid, y_sum - 1.00),
                arrowprops=dict(arrowstyle="-|>", color=t["ink2"], linewidth=1.8))

    # --- the single output -------------------------------------------------
    ox = mid - CELL_W / 2
    ax.add_patch(FancyBboxPatch((ox, y_out), CELL_W, CELL_H,
                                boxstyle="round,pad=0,rounding_size=0.10", linewidth=0,
                                facecolor=t["cpu"], zorder=3))
    ax.text(ox + CELL_W / 2, y_out + CELL_H / 2, "$i$", ha="center", va="center",
            color=t["surface"], fontsize=12, fontweight="bold", zorder=4)
    ax.text(ox - 0.30, y_out + CELL_H / 2, "arrival", ha="right", va="center",
            color=t["cpu"], fontsize=13, fontweight="bold")
    ax.text(ox + CELL_W + 0.35, y_out + CELL_H / 2,
            "one node's arrival time\n= one work item = one GPU thread", ha="left",
            va="center", color=t["ink2"], fontsize=11.5)

    # --- the recurrence, stated --------------------------------------------
    ax.text(-2.5, 8.35,
            r"$\mathrm{arrival}[i] \;=\; \max_{k\,<\,K}\;"
            r"\left(\mathrm{arrival\_in}[i][k] \;+\; \mathrm{delay}[i][k]\right)$",
            ha="left", va="top", color=t["ink"], fontsize=20)
    ax.text(-2.5, 7.15,
            "Block-based STA arrival propagation. Two large vectors in, an add, "
            "a max over each node's fanin window, one value out.",
            ha="left", va="top", color=t["ink2"], fontsize=12)

    # --- the number that decides everything ---------------------------------
    bytes_per_node = FANIN * 2 * 4 + 4
    ops_per_node = FANIN * 2 - 1  # kFanin adds + (kFanin-1) maxes
    ax.text(-2.5, -0.80,
            f"Per node: {FANIN}×2 loads + 1 store = {bytes_per_node} bytes, "
            f"{ops_per_node} flops  "
            rf"$\approx {ops_per_node / bytes_per_node:.2f}$ flops/byte",
            ha="left", va="bottom", color=t["ink"], fontsize=12.5, fontweight="bold")
    ax.text(-2.5, -1.25,
            "Every byte is read exactly once — there is no reuse to exploit. These "
            "machines need 17–43 flops/byte to break even,\nso this is "
            "memory-bandwidth bound by two orders of magnitude — at any problem size.",
            ha="left", va="top", color=t["ink2"], fontsize=11.5)

    out = FIGS if theme == "light" else os.path.join(FIGS, "dark")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "00_problem.png")
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    for th in ("light", "dark"):
        draw(th)
