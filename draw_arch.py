#!/usr/bin/env python3
"""The two architecture diagrams section 2 needs that no whitepaper provides.

  09_simt_coalescing.png   one instruction -> 32 lanes, and what happens to a
                           load when those lanes are contiguous vs scattered
  10_latency_hiding.png    why a slower memory system delivers more bandwidth:
                           a stalling CPU timeline against overlapping warps,
                           plus the Little's Law arithmetic

Same palette and chrome as the rest of the deck.

    python3 draw_arch.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", cpu="#2a78d6", gpu="#eb6834",
                  good="#1baf7a", pale="#dfe9f7"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", cpu="#3987e5", gpu="#d95926",
                 good="#199e70", pale="#22314a"),
}


def box(ax, x, y, w, h, color, t, label=None, fs=10, alpha=1.0, txt=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=0, facecolor=color, alpha=alpha, zorder=3))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color=txt or t["surface"], fontsize=fs, fontweight="bold", zorder=4)


def save(fig, theme, name):
    out = FIGS if theme == "light" else os.path.join(FIGS, "dark")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")


# ---------------------------------------------------------------------------


def draw_simt(theme):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(13.0, 7.0), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.set_xlim(-0.6, 17.4)
    ax.set_ylim(-2.5, 8.4)
    ax.axis("off")

    N, W, G = 16, 0.86, 0.14      # 16 lanes drawn, labelled as 32
    x0 = 0.9
    span = N * (W + G) - G

    ax.text(-0.5, 8.3, "One instruction, 32 lanes", color=t["ink"], fontsize=17,
            fontweight="bold", va="top")
    ax.text(-0.5, 7.65,
            "You write scalar code for one node. The hardware issues it to all 32 "
            "lanes of a warp at once.",
            color=t["ink2"], fontsize=12, va="top")

    # the single instruction
    box(ax, x0 + span / 2 - 3.0, 6.0, 6.0, 0.78, t["ink2"], t,
        "out[i] = propagate(..., i)", fs=13)
    ax.text(x0 + span / 2 + 3.25, 6.39, "one instruction", color=t["ink2"], fontsize=11,
            va="center", ha="left")

    # fan-out to lanes
    for i in range(N):
        cx = x0 + i * (W + G) + W / 2
        ax.annotate("", xy=(cx, 5.02), xytext=(x0 + span / 2, 5.95),
                    arrowprops=dict(arrowstyle="-", color=t["muted"], linewidth=0.7,
                                    alpha=0.75))
        box(ax, x0 + i * (W + G), 4.2, W, 0.8, t["cpu"], t, str(i), fs=9)
    ax.text(x0 + span + 0.3, 4.6, "…32 lanes", color=t["ink2"], fontsize=11,
            va="center", ha="left")
    ax.text(x0 - 0.25, 4.6, "warp", color=t["cpu"], fontsize=12.5, fontweight="bold",
            va="center", ha="right")

    # ---- what the load looks like in memory --------------------------------
    ax.text(-0.5, 3.35, "…and what that does to a memory load", color=t["ink"],
            fontsize=15, fontweight="bold", va="top")

    M, MW, MG = 40, 0.38, 0.055          # the memory strip: 40 addresses
    mspan = M * (MW + MG) - MG
    mx = x0

    def memory_row(y, hit_idx, colour, label, sublabel):
        for j in range(M):
            on = j in hit_idx
            box(ax, mx + j * (MW + MG), y, MW, 0.52,
                colour if on else t["grid"], t, alpha=1.0 if on else 0.9)
        ax.text(mx - 0.25, y + 0.26, label, color=colour, fontsize=12,
                fontweight="bold", va="center", ha="right")
        ax.text(mx + mspan + 0.3, y + 0.26, sublabel, color=t["ink2"], fontsize=11,
                va="center", ha="left")

    # contiguous: 16 adjacent addresses -> one transaction
    y_ok = 2.05
    hit = set(range(4, 20))
    memory_row(y_ok, hit, t["good"], "contiguous", "full bandwidth")
    ax.add_patch(FancyArrowPatch((mx + 4 * (MW + MG), y_ok - 0.18),
                                 (mx + 20 * (MW + MG) - MG, y_ok - 0.18),
                                 arrowstyle="|-|,widthA=3,widthB=3", color=t["good"],
                                 linewidth=1.6, shrinkA=0, shrinkB=0))
    ax.text(mx + 12 * (MW + MG), y_ok - 0.62, "→  ONE wide transaction",
            color=t["good"], fontsize=12.5, fontweight="bold", ha="center")

    # scattered: 16 addresses spread across the strip -> one transaction each
    y_bad = 0.05
    hit2 = {0, 3, 5, 8, 10, 13, 15, 18, 21, 24, 26, 29, 32, 34, 37, 39}
    memory_row(y_bad, hit2, t["gpu"], "scattered", "~1/32 of the bandwidth")
    for j in sorted(hit2):
        cx = mx + j * (MW + MG) + MW / 2
        ax.annotate("", xy=(cx, y_bad - 0.10), xytext=(cx, y_bad - 0.42),
                    arrowprops=dict(arrowstyle="-", color=t["gpu"], linewidth=1.0))
    ax.text(mx + mspan / 2, y_bad - 0.95, "→  32 SEPARATE transactions",
            color=t["gpu"], fontsize=12.5, fontweight="bold", ha="center")

    ax.text(mx - 0.25, (y_ok + y_bad) / 2 + 0.26, "memory\naddresses →",
            color=t["muted"], fontsize=10, va="center", ha="right")

    ax.text(-0.5, -1.60,
            "This is why data layout is a performance decision on a GPU, not a matter "
            "of taste — and why our benchmark stores\nfanin slot k contiguously across "
            "nodes rather than keeping each node's 8 edges together.",
            color=t["ink"], fontsize=12, va="top", fontweight="bold")

    save(fig, theme, "09_simt_coalescing.png")


# ---------------------------------------------------------------------------


def draw_latency(theme):
    t = THEMES[theme]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.6, 6.0), dpi=150,
                                   gridspec_kw=dict(width_ratios=[1.35, 1.0], wspace=0.18))
    fig.patch.set_facecolor(t["surface"])
    for a in (axl, axr):
        a.set_facecolor(t["surface"])
        a.axis("off")

    # ---- left: the two timelines -------------------------------------------
    axl.set_xlim(0, 12.4)
    axl.set_ylim(-0.4, 8.6)
    axl.text(0, 8.5, "The GPU is SLOWER at any one request", color=t["ink"],
             fontsize=15.5, fontweight="bold", va="top")
    axl.text(0, 7.85, "…and delivers more bandwidth anyway, by overlapping them.",
             color=t["ink2"], fontsize=11.5, va="top")

    # CPU timeline
    axl.text(0, 6.75, "CPU thread", color=t["cpu"], fontsize=12.5, fontweight="bold",
             va="center")
    y = 6.15
    box(axl, 0.0, y, 1.0, 0.5, t["cpu"], t, "work", fs=9)
    box(axl, 1.05, y, 3.6, 0.5, t["muted"], t, "stalled on memory", fs=9, alpha=0.5)
    box(axl, 4.7, y, 1.0, 0.5, t["cpu"], t, "work", fs=9)
    box(axl, 5.75, y, 3.6, 0.5, t["muted"], t, "stalled on memory", fs=9, alpha=0.5)
    box(axl, 9.4, y, 1.0, 0.5, t["cpu"], t, "work", fs=9)
    axl.text(10.6, y + 0.25, "mostly waiting", color=t["ink2"], fontsize=10.5,
             va="center", ha="left")

    # GPU timeline: many warps, staggered
    axl.text(0, 5.15, "SM, many warps", color=t["gpu"], fontsize=12.5,
             fontweight="bold", va="center")
    for w in range(7):
        yy = 4.55 - w * 0.56
        off = w * 0.62
        axl.text(-0.05, yy + 0.22, f"w{w}", color=t["muted"], fontsize=8.5,
                 va="center", ha="right")
        box(axl, off, yy, 0.55, 0.44, t["gpu"], t)
        box(axl, off + 0.6, yy, 3.4, 0.44, t["muted"], t, alpha=0.42)
        box(axl, off + 4.05, yy, 0.55, 0.44, t["gpu"], t)
        box(axl, off + 4.65, yy, 3.4, 0.44, t["muted"], t, alpha=0.42)
        if off + 8.1 < 12.0:
            box(axl, off + 8.1, yy, 0.55, 0.44, t["gpu"], t)
    axl.annotate("", xy=(11.2, 4.99), xytext=(11.2, 0.62),
                 arrowprops=dict(arrowstyle="<->", color=t["ink2"], linewidth=1.4))
    axl.text(11.35, 2.8, "some warp is\nalways ready", color=t["ink"], fontsize=11,
             fontweight="bold", va="center", ha="left")

    axl.text(0, 0.0, "Same latency per request. The stall is hidden, not removed.",
             color=t["ink"], fontsize=11.5, fontweight="bold", va="top")

    # ---- right: Little's Law -----------------------------------------------
    axr.set_xlim(0, 10)
    axr.set_ylim(-0.4, 8.6)
    axr.text(0, 8.5, "How much must be in flight?", color=t["ink"], fontsize=15.5,
             fontweight="bold", va="top")
    axr.text(0, 7.8, r"$\mathrm{bytes\ in\ flight} \;=\; "
                     r"\mathrm{bandwidth} \times \mathrm{latency}$",
             color=t["ink2"], fontsize=14, va="top")

    rows = [("i5-13500\nDDR5", "49 GB/s", "~90 ns", "~4 KB", t["cpu"], 0.06),
            ("RTX A4000\nGDDR6", "448 GB/s", "~450 ns", "~200 KB", t["gpu"], 0.34),
            ("H100\nHBM3", "4000 GB/s", "~550 ns", "~2 MB", t["gpu"], 1.0)]
    y = 6.1
    for name, bw, lat, flight, col, frac in rows:
        axr.text(0.0, y + 0.30, name, color=col, fontsize=11.5, fontweight="bold",
                 va="center")
        axr.text(2.5, y + 0.30, f"{bw}  ×  {lat}", color=t["ink2"], fontsize=11,
                 va="center")
        box(axr, 2.5, y - 0.42, 6.2 * frac, 0.36, col, t)
        axr.text(2.5 + 6.2 * frac + 0.15, y - 0.24, flight, color=t["ink"],
                 fontsize=12, fontweight="bold", va="center")
        y -= 1.55

    axr.text(0, 1.35,
             "A CPU core tracks ~a dozen outstanding misses.\n"
             "14 cores ≈ a few hundred. It cannot hold 2 MB\n"
             "in the air — so it could not use that bandwidth\n"
             "even if you gave it the bus.",
             color=t["ink"], fontsize=11.5, va="top", fontweight="bold")
    axr.text(0, -0.30, "Latencies illustrative; bandwidths measured.",
             color=t["muted"], fontsize=10, va="top")

    save(fig, theme, "10_latency_hiding.png")


if __name__ == "__main__":
    for th in ("light", "dark"):
        draw_simt(th)
        draw_latency(th)
