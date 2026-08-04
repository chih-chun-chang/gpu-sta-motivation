#!/usr/bin/env python3
"""Backup slide: why this kernel is memory bound, for any fanin, on any machine.

Two panels:

  left   arithmetic intensity as a function of fanin K. It rises, and it
         saturates at 0.25 flops/byte -- (2K-1)/(8K+4) -> 1/4. So there is no
         fanin large enough to escape the memory-bound regime.
  right  a roofline for all three machines, with the kernel's whole reachable
         intensity range shaded. The band never reaches any ridge point, and
         the three measured points sit on the memory roofs, which is the
         empirical confirmation.

    python3 draw_roofline.py     # writes figures/07_roofline.png (+ dark)

Peak FLOP/s are vendor figures for the GPUs; the CPU one is an estimate
(14 cores, AVX2 FMA) and is labelled as such on the slide. The conclusion does
not depend on it -- the kernel is ~2 orders of magnitude from every ridge point,
so even a 2x error changes nothing.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")

FANIN = 8

# name, peak GFLOP/s (FP32), achieved GB/s (measured by this repo), colour key
MACHINES = [
    ("i5-13500 CPU", 1370.0, 48.9, "cpu"),
    ("RTX A4000", 19170.0, 429.7, "gpu"),
    ("GH200 (H100)", 67000.0, 2852.0, "alt"),
]
# Bandwidth used to draw the memory roof. The GPUs use their vendor peak. The
# CPU uses its MEASURED streaming bandwidth, because the DIMM spec on this box
# is not readable without root -- so the CPU's measured point necessarily sits
# on its own roof, and the slide says so rather than presenting it as a result.
PEAK_BW = {"i5-13500 CPU": 48.9, "RTX A4000": 448.0, "GH200 (H100)": 4023.0}
BW_KIND = {"i5-13500 CPU": "measured", "RTX A4000": "peak", "GH200 (H100)": "peak"}

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  cpu="#2a78d6", gpu="#eb6834", alt="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 cpu="#3987e5", gpu="#d95926", alt="#199e70"),
}


def intensity(k):
    """(K adds + K-1 maxes) / (K*2 loads + 1 store, in bytes)."""
    return (2.0 * k - 1.0) / (8.0 * k + 4.0)


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t["muted"], labelsize=10.5, length=0)
    ax.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def draw(theme):
    t = THEMES[theme]
    fig, (axk, axr) = plt.subplots(1, 2, figsize=(13.4, 5.6), dpi=150,
                                   gridspec_kw=dict(width_ratios=[1.0, 1.35], wspace=0.26))
    fig.patch.set_facecolor(t["surface"])
    style(axk, t)
    style(axr, t)

    # ---- left: intensity vs fanin -----------------------------------------
    ks = np.logspace(0, 7, 200, base=2)
    axk.plot(ks, intensity(ks), color=t["cpu"], linewidth=2.4, zorder=3)
    axk.axhline(0.25, color=t["muted"], linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
    axk.text(1.15, 0.2575, "upper bound: 0.25 flops/byte, for any fanin",
             color=t["ink"], fontsize=11, fontweight="bold", va="bottom")

    axk.scatter([FANIN], [intensity(FANIN)], s=90, c=t["cpu"], edgecolors=t["surface"],
                linewidths=2, zorder=4)
    axk.annotate(f"K = {FANIN}\n{intensity(FANIN):.3f}", xy=(FANIN, intensity(FANIN)),
                 xytext=(FANIN * 2.6, intensity(FANIN) - 0.055), color=t["ink"],
                 fontsize=11.5, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=t["muted"], linewidth=1.2,
                                 shrinkA=0, shrinkB=8))

    axk.set_xscale("log", base=2)
    axk.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    axk.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axk.set_ylim(0, 0.30)
    axk.set_xlabel("Fanin K (log scale)", color=t["ink2"], fontsize=11.5, labelpad=8)
    axk.set_ylabel("Arithmetic intensity (flops/byte)", color=t["ink2"], fontsize=11.5,
                   labelpad=8)
    axk.set_title("Widening the fanin does not help", color=t["ink"], fontsize=14,
                  fontweight="bold", loc="left", pad=12)

    # ---- right: roofline ---------------------------------------------------
    x = np.logspace(-1.4, 2.4, 400)
    lo, hi = intensity(1), 0.25  # the kernel's entire reachable range

    axr.axvspan(lo, hi, color=t["cpu"], alpha=0.13, zorder=1)
    axr.text((lo * hi) ** 0.5, 1.2e5, "the kernel,\nany fanin", color=t["cpu"],
             fontsize=11.5, fontweight="bold", va="top", ha="center")

    for name, peak, achieved, key in MACHINES:
        bw = PEAK_BW[name]
        axr.plot(x, np.minimum(peak, bw * x), color=t[key], linewidth=2.2, zorder=3,
                 label=f"{name}  ({peak / 1000:.1f} TF, {bw:.0f} GB/s {BW_KIND[name]})")
        ridge = peak / bw
        axr.scatter([ridge], [peak], s=44, c=t[key], edgecolors=t["surface"],
                    linewidths=1.6, zorder=4)
        axr.text(ridge * 1.15, peak * 0.62, f"ridge {ridge:.0f}", color=t[key],
                 fontsize=10, fontweight="bold")
        # Measured: achieved bandwidth x this kernel's intensity.
        axr.scatter([intensity(FANIN)], [achieved * intensity(FANIN)], s=95, marker="D",
                    c=t[key], edgecolors=t["surface"], linewidths=1.8, zorder=5)

    axr.set_xscale("log")
    axr.set_yscale("log")
    axr.set_xlim(x[0], x[-1])
    axr.set_ylim(1, 2e5)
    axr.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axr.get_yaxis().set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / 1000:g} TF" if v >= 1000 else f"{v:g}"))
    axr.set_xlabel("Arithmetic intensity (flops/byte, log scale)", color=t["ink2"],
                   fontsize=11.5, labelpad=8)
    axr.set_ylabel("Attainable performance (log scale)", color=t["ink2"], fontsize=11.5,
                   labelpad=8)
    axr.set_title("Two orders of magnitude from every ridge point", color=t["ink"],
                  fontsize=14, fontweight="bold", loc="left", pad=12)

    leg = axr.legend(loc="lower right", frameon=False, fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    axr.scatter([], [], s=95, marker="D", c=t["muted"], edgecolors=t["surface"],
                linewidths=1.8, label="measured")
    axr.text(0.985, 0.30, "◆ measured — on the memory roof", transform=axr.transAxes,
             color=t["ink2"], fontsize=10, ha="right")

    fig.text(0.5, -0.02,
             "Left: intensity is (2K−1)/(8K+4), which tends to 1/4. Right: the shaded band is "
             "every intensity this kernel can reach; it never meets a ridge point.\n"
             "Diamonds are measured throughput. The two GPU diamonds sit on vendor-peak "
             "memory roofs at 96% and 71% — that is what \"memory bound\" means.\n"
             "The CPU roof is its own measured streaming bandwidth (DIMM spec unreadable "
             "without root), so that point is on its roof by construction. CPU peak FLOP/s "
             "is an estimate.",
             color=t["muted"], fontsize=10.5, ha="center", va="top")

    out = FIGS if theme == "light" else os.path.join(FIGS, "dark")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "07_roofline.png")
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    for th in ("light", "dark"):
        draw(th)
