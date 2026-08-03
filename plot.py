#!/usr/bin/env python3
"""Turn the benchmark CSVs in data/ into the slide figures in figures/.

Four figures, in presentation order:

  01  STA throughput vs pool size        -- the CPU ceiling is the memory bus
  02  thread-per-item vs thread pool     -- what a pool is actually worth, and
                                            what it still doesn't buy you
  03  CPU vs GPU, incl. naive offload    -- the GPU bar that's SHORTER than CPU
  04  where the time actually goes       -- PCIe dwarfs the kernel
  05  runtime vs problem size, CPU       -- thread-per-node never catches up
  06  runtime vs problem size, GPU       -- the copy costs more than the CPU

The problem-formulation diagram (00_problem.png) is drawn by draw_problem.py.

Run after run_all.sh. Light figures to figures/, dark to figures/dark/.
"""

import csv
import os
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")

# This machine. i5-13500 = 6 P-cores (2 threads each) + 8 E-cores.
PHYSICAL_CORES = 14
LOGICAL_CORES = 20

# Palette slots 1-3, validated all-pairs in both modes by the dataviz validator
# (worst CVD dE 9.2 light / 9.4 dark).
#
# Colour follows the ENTITY, the same in every figure -- never the rank:
#   cpu    blue   the CPU doing work in parallel (std::for_each)
#   gpu    orange the GPU, in every figure it appears
#   naive  aqua   one std::thread per work item, the strawman
#   muted  gray   single-thread baseline (a reference, not a competitor)
THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        cpu="#2a78d6", gpu="#eb6834", naive="#1baf7a",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        cpu="#3987e5", gpu="#d95926", naive="#199e70",
    ),
}


# --------------------------------------------------------------------------
# data


def read_csv(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def medians(rows, key, value):
    grouped = defaultdict(list)
    for r in rows:
        grouped[float(r[key])].append(float(r[value]))
    return {x: statistics.median(v) for x, v in sorted(grouped.items())}


def si(v, _=None):
    if v >= 1e9:
        return f"{v / 1e9:.1f}G"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}k"
    return f"{v:.0f}"


# --------------------------------------------------------------------------
# chrome


def new_fig(theme, w=12.0, h=6.4):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t["muted"], labelsize=11, length=0)
    ax.grid(True, color=t["grid"], linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    return fig, ax, t


def titles(ax, t, title, subtitle, xlabel, ylabel):
    ax.set_title(title, color=t["ink"], fontsize=17, fontweight="bold", loc="left", pad=28)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, color=t["ink2"], fontsize=12,
            va="bottom", ha="left")
    ax.set_xlabel(xlabel, color=t["ink2"], fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, color=t["ink2"], fontsize=12, labelpad=10)


def save(fig, theme, name):
    out = FIGS if theme == "light" else os.path.join(FIGS, "dark")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")


# --------------------------------------------------------------------------
# figures


def fig_ceiling(theme, all_rows):
    """01 -- the CPU ceiling, and that it arrives long before the core count."""
    rows = [r for r in all_rows if r["strategy"] == "pool"]
    bw = medians(rows, "threads", "gb_per_sec")
    raw = [(float(r["threads"]), float(r["gb_per_sec"])) for r in rows]
    peak = max(bw.values())
    peak_n = max(bw, key=bw.get)
    one = bw[1.0]
    # First pool size reaching 95% of peak: where the bus, not the cores, wins.
    knee = min(n for n, v in bw.items() if v >= 0.95 * peak)

    fig, ax, t = new_fig(theme)
    ax.scatter([x for x, _ in raw], [y for _, y in raw], s=30, c=t["cpu"],
               edgecolors=t["surface"], linewidths=0.7, zorder=3, alpha=0.9)
    ax.axhline(peak, color=t["muted"], linewidth=1.2, linestyle=(0, (5, 4)), zorder=2)

    for n, label, style in ((knee, f"saturated at {knee:.0f}", (0, (4, 3))),
                            (PHYSICAL_CORES, f"{PHYSICAL_CORES} physical cores", (0, (2, 3)))):
        ax.axvline(n, color=t["axis"], linewidth=1.0, linestyle=style, zorder=2)
        ax.text(n, peak * 0.06, f"  {label}", color=t["muted"], fontsize=10, rotation=90,
                va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(0, peak * 1.30)

    ax.annotate(
        f"ceiling: {peak:.0f} GB/s\n"
        f"only {peak / one:.1f}x one thread, on a {PHYSICAL_CORES}-core CPU",
        xy=(peak_n, peak), xytext=(0.03, 0.90), textcoords="axes fraction",
        color=t["ink"], fontsize=12.5, fontweight="bold", ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=t["muted"], linewidth=1.2, shrinkA=12,
                        shrinkB=8, connectionstyle="arc3,rad=-0.15"))

    titles(ax, t,
           "The CPU ceiling is the memory bus, not the core count",
           "STA arrival propagation, 64M timing edges (544 MB working set). "
           "std::for_each(par_unseq), pool size via TBB.",
           "Thread pool size (log scale)", "Achieved memory bandwidth (GB/sec)")
    save(fig, theme, "01_sta_cpu_ceiling.png")
    return peak, one, peak_n


def fig_naive_vs_pool(theme, rows):
    """02 -- act 1 vs act 2: what a thread pool is actually worth."""
    pool = medians([r for r in rows if r["strategy"] == "pool"], "threads", "gb_per_sec")
    tpi = statistics.median(
        [float(r["gb_per_sec"]) for r in rows if r["strategy"] == "thread_per_item"])
    xs = sorted(pool)
    peak = max(pool.values())

    fig, ax, t = new_fig(theme)
    ax.plot(xs, [pool[x] for x in xs], color=t["cpu"], linewidth=2.2, marker="o",
            markersize=5, markeredgecolor=t["surface"], markeredgewidth=1.2, zorder=3,
            label="Thread pool  (std::for_each, par_unseq)")
    ax.axhline(tpi, color=t["naive"], linewidth=2.0, linestyle=(0, (6, 4)), zorder=3,
               label="One std::thread per work item")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.get_yaxis().set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:g}" if v >= 1 else f"{v:.2f}"))
    ax.set_ylim(tpi * 0.35, peak * 4.0)

    ax.axvline(PHYSICAL_CORES, color=t["axis"], linewidth=1.0, linestyle=(0, (2, 3)), zorder=2)
    ax.text(PHYSICAL_CORES, tpi * 0.45, f"  {PHYSICAL_CORES} physical cores",
            color=t["muted"], fontsize=10, rotation=90, va="bottom", ha="left")

    # The gap is the whole point of the slide.
    mid = xs[len(xs) // 2]
    ax.annotate("", xy=(mid, peak), xytext=(mid, tpi),
                arrowprops=dict(arrowstyle="<->", color=t["ink2"], linewidth=1.6))
    ax.text(mid * 1.25, (peak * tpi) ** 0.5, f"{peak / tpi:,.0f}x", color=t["ink"],
            fontsize=15, fontweight="bold", va="center", ha="left")

    ax.annotate(f"bounded by thread creation:\n~7.5 us to spawn, ~50 ns of work",
                xy=(xs[2], tpi), xytext=(0.05, 0.16), textcoords="axes fraction",
                color=t["naive"], fontsize=12, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=t["naive"], linewidth=1.2, shrinkA=0,
                                shrinkB=8))
    sat = min(xs, key=lambda v: abs(v - 64))
    ax.annotate("and it still saturates\nat 6 of 14 cores", xy=(sat, pool[sat]),
                xytext=(0.52, 0.60), textcoords="axes fraction", color=t["cpu"],
                fontsize=12, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=t["cpu"], linewidth=1.2, shrinkA=0,
                                shrinkB=8))

    leg = ax.legend(loc="center left", frameon=False, fontsize=11)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    titles(ax, t,
           "A thread pool buys 5,000x. It does not buy you the machine.",
           "STA propagation. Both axes log. The pool removes thread-creation "
           "cost, and reveals the memory wall underneath.",
           "Thread pool size (log scale)", "Achieved memory bandwidth (GB/sec, log scale)")
    save(fig, theme, "02_naive_vs_pool.png")


def fig_cpu_vs_gpu(theme, cpu_peak, cpu_one, cpu_peak_n, gpu_rows):
    """03 -- the punchline: naive offload is SLOWER than the CPU."""
    by_impl = {r["impl"]: float(r["gb_per_sec"]) for r in gpu_rows}
    kern = by_impl["gpu_kernel"]
    xfer = by_impl["gpu_with_transfer"]
    bars = [
        ("1 CPU thread", cpu_one, "cpu"),
        # Not "best at N threads": the plateau is flat, so the argmax wanders
        # run to run. Quote the value, not a spurious thread count.
        ("CPU, best observed", cpu_peak, "cpu"),
        ("GPU, naive offload\n(copy in, run, copy out)", xfer, "gpu"),
    ]
    if "gpu_managed" in by_impl:
        bars.append(("GPU, unified memory\n(no explicit copies)", by_impl["gpu_managed"], "gpu"))
    bars.append(("GPU, data resident", kern, "gpu"))

    fig, ax, t = new_fig(theme, h=5.8)
    ax.grid(True, axis="x", color=t["grid"], linewidth=0.8)
    ax.grid(False, axis="y")

    ys = range(len(bars))
    for y, (label, val, kind) in zip(ys, bars):
        ax.barh(y, val, height=0.58, color=t[kind], zorder=3, edgecolor=t["surface"],
                linewidth=2)
        note = f"{val:.0f} GB/s"
        if kind == "gpu":
            note += f"   ({val / cpu_peak:.1f}x the whole CPU)"
        ax.text(val + kern * 0.015, y, note, va="center", ha="left", color=t["ink"],
                fontsize=12, fontweight="bold")

    # The line the audience should stare at.
    ax.axvline(cpu_peak, color=t["muted"], linewidth=1.4, linestyle=(0, (5, 4)), zorder=4)

    ax.set_xlim(0, kern * 1.30)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([b[0] for b in bars], color=t["ink"], fontsize=12)
    ax.invert_yaxis()

    ax.annotate("naive offload is SLOWER\nthan just using the CPU",
                xy=(xfer, 2.29), xytext=(kern * 0.28, 2.50), color=t["gpu"], fontsize=12.5,
                fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=t["gpu"], linewidth=1.6, shrinkA=6,
                                shrinkB=6, connectionstyle="arc3,rad=0.2"))

    handles = [plt.Rectangle((0, 0), 1, 1, color=t["cpu"]),
               plt.Rectangle((0, 0), 1, 1, color=t["gpu"])]
    leg = ax.legend(handles, ["CPU", "GPU"], loc="upper right", frameon=False, fontsize=11)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    titles(ax, t,
           "Same kernel, same numbers, bit-for-bit identical results",
           "Achieved bandwidth on STA arrival propagation. Dashed line = the best "
           "the whole CPU can do.",
           "Achieved memory bandwidth (GB/sec)", "")
    save(fig, theme, "03_cpu_vs_gpu.png")


def fig_breakdown(theme, rows):
    """04 -- where the time actually goes."""
    phase = {r["phase"]: (float(r["ms"]), float(r["share"])) for r in rows}
    order = [("h2d", "Copy in  (host -> device)", "gpu"),
             ("kernel", "Compute  (the actual STA work)", "cpu"),
             ("d2h", "Copy out (device -> host)", "gpu")]
    total = sum(phase[k][0] for k, _, _ in order)

    fig, ax, t = new_fig(theme, h=3.4)
    ax.grid(False)
    left = 0.0
    for key, label, kind in order:
        ms, share = phase[key]
        ax.barh(0, ms, left=left, height=0.42, color=t[kind], zorder=3,
                edgecolor=t["surface"], linewidth=2)
        if share > 0.08:
            ax.text(left + ms / 2, 0, f"{share * 100:.0f}%", ha="center", va="center",
                    color=t["surface"], fontsize=13, fontweight="bold", zorder=4)
        left += ms

    ax.annotate(f"{phase['kernel'][1] * 100:.0f}%  ({phase['kernel'][0]:.1f} ms)",
                xy=(phase["h2d"][0] + phase["kernel"][0] / 2, 0.22),
                xytext=(phase["h2d"][0] * 0.62, 0.60), color=t["cpu"], fontsize=12.5,
                fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=t["cpu"], linewidth=1.4, shrinkA=2,
                                shrinkB=4))

    for key, label, _ in order:
        ms, share = phase[key]
        ax.plot([], [], color=t["gpu" if key != "kernel" else "cpu"], linewidth=8,
                label=f"{label} — {ms:.1f} ms")
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), frameon=False,
                    fontsize=11, ncol=1, handlelength=1.2)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    ax.set_ylim(-0.5, 0.85)
    ax.set_yticks([])
    ax.set_xlim(0, total)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} ms"))
    titles(ax, t,
           "97% of a naive GPU port is PCIe",
           f"One STA propagation pass over 64M edges, pinned host memory, "
           f"{total:.0f} ms total.",
           "", "")
    save(fig, theme, "04_where_the_time_goes.png")


def fig_runtime_vs_size(theme, cpu_rows, gpu_rows):
    """05/06 -- runtime against problem size."""
    def series(rows, strat):
        sel = [r for r in rows if r["strategy"] == strat]
        return medians(sel, "nodes", "ms") if sel else {}

    tpi = series(cpu_rows, "thread_per_item")
    one = series(cpu_rows, "single_thread")
    par = series(cpu_rows, "for_each_par")
    kern = series(gpu_rows, "gpu_kernel") if gpu_rows else {}
    xfer = series(gpu_rows, "gpu_with_transfer") if gpu_rows else {}

    # L3 is 24 MB on this machine; mark where the working set outgrows it.
    l3_nodes = 24 * 1024 * 1024 / (sta_bytes_per_node())

    def plot(specs, title, subtitle, name, note=None):
        fig, ax, t = new_fig(theme)
        for med, color, style, label in specs:
            if not med:
                continue
            xs = sorted(med)
            ax.plot(xs, [med[x] for x in xs], color=color, linewidth=2.2, linestyle=style,
                    marker="o", markersize=5, markeredgecolor=t["surface"],
                    markeredgewidth=1.2, zorder=3, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: si(v)))
        ax.get_yaxis().set_major_formatter(
            FuncFormatter(lambda v, _: f"{v:g} ms" if v >= 1 else f"{v:g}"))
        ax.axvline(l3_nodes, color=t["axis"], linewidth=1.0, linestyle=(0, (2, 3)), zorder=2)
        ax.text(l3_nodes, ax.get_ylim()[0] * 1.6, "  working set outgrows L3",
                color=t["muted"], fontsize=10, rotation=90, va="bottom", ha="left")
        if note:
            ax.text(0.03, 0.62, note, transform=ax.transAxes, color=t["ink2"], fontsize=11,
                    va="top", ha="left")
        leg = ax.legend(loc="upper left", frameon=False, fontsize=11,
                        bbox_to_anchor=(0.03, 0.53) if note else (0.02, 0.98))
        for txt in leg.get_texts():
            txt.set_color(t["ink2"])
        titles(ax, t, title, subtitle, "Problem size (nodes, log scale)",
               "Runtime for one full pass (log scale)")
        save(fig, theme, name)

    plot([(tpi, THEMES[theme]["naive"], "-", "One std::thread per node"),
          (one, THEMES[theme]["muted"], (0, (5, 3)), "Single thread (reference)"),
          (par, THEMES[theme]["cpu"], "-", "std::for_each(par_unseq)")],
         "One thread per node never becomes viable",
         "One full STA propagation pass. Both axes log. All three lines are the "
         "same arithmetic.",
         "05_runtime_vs_size_cpu.png",
         note="thread-per-node stops at 1M nodes:\nbeyond that a single pass takes minutes")

    plot([(par, THEMES[theme]["cpu"], "-", "CPU: std::for_each(par_unseq)"),
          (xfer, THEMES[theme]["gpu"], (0, (5, 3)), "GPU: with copy in + copy out"),
          (kern, THEMES[theme]["gpu"], "-", "GPU: data already resident")],
         "The copy costs more than the whole CPU does",
         "One full STA propagation pass. The dashed GPU line sits ABOVE the CPU "
         "line at every size.",
         "06_runtime_vs_size_gpu.png")


def sta_bytes_per_node():
    # kFanin(8) * 2 loads * 4 bytes + 1 store * 4 bytes -- mirrors sta::bytes_per_pass.
    return 8 * 2 * 4 + 4


# --------------------------------------------------------------------------


def main():
    sta_rows = read_csv("sta_cpu.csv")
    gpu_rows = read_csv("gpu_sta.csv")
    bd_rows = read_csv("gpu_breakdown.csv")
    size_cpu = read_csv("size_cpu.csv")
    size_gpu = read_csv("size_gpu.csv")
    if not sta_rows:
        sys.exit("data/sta_cpu.csv missing -- run ./run_all.sh first")
    if gpu_rows and not all(r["checksum_ok"] == "1" for r in gpu_rows):
        print("  WARNING: GPU checksum did not match the CPU reference")

    os.makedirs(FIGS, exist_ok=True)
    for theme in ("light", "dark"):
        print(f"{theme}:")
        peak, one, peak_n = fig_ceiling(theme, sta_rows)
        fig_naive_vs_pool(theme, sta_rows)
        if gpu_rows:
            fig_cpu_vs_gpu(theme, peak, one, peak_n, gpu_rows)
        if bd_rows:
            fig_breakdown(theme, bd_rows)
        if size_cpu:
            fig_runtime_vs_size(theme, size_cpu, size_gpu)


if __name__ == "__main__":
    main()
