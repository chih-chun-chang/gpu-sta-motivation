# Why GPU-accelerated STA — the opening experiment

A toy benchmark for the opening of a talk on GPU-accelerated static timing
analysis. It builds the case in three acts: one thread per work item, then a
thread pool, then the GPU — measuring at each step what is actually limiting
throughput. The figure layout is loosely modelled on the throughput-vs-threads
plots in Conor Spilsbury's CppCon 2025 *Threads vs. Coroutines*, but the kernel,
the workload and the conclusion are all different.

Talking points and anticipated audience questions: **[SPEAKER_NOTES.md](SPEAKER_NOTES.md)**.

## Status

**Built and measured** on i5-13500 + RTX A4000: the STA kernel (shared verbatim by
CPU and GPU), a CPU parallelism sweep, a GPU run with transfer breakdown, four
figures in light and dark, and speaker notes. All numbers below are reproducible
with `./run_all.sh && python3 plot.py`.

**The intended three-act narrative**, with measured numbers for each act:

| act | approach | result | what's actually limiting it |
|---|---|---:|---|
| 1 | one `std::thread` per work item | 0.009 GB/s | thread **creation** (~7.5 µs each vs ~50 ns of work) |
| 2 | thread pool / `std::for_each(par_unseq)` | 48.9 GB/s | **memory bandwidth** — saturates at 5–6 of 14 cores |
| 3 | GPU, data resident | 429.7 GB/s | bandwidth again, but 8.8× more of it |
| 3b | GPU, naive copy-in/copy-out | 11.9 GB/s | **PCIe** — 91% of the time is H2D |

Act 2's bottleneck is thread *creation*, **not context switching**. That was
measured, and the data is unambiguous: 512 compute-bound threads cause ~9,900
context switches/sec against a ~6,400/sec idle baseline, with throughput
unchanged. A blocking-I/O workload at 4096 threads causes 765,000/sec. Saying "the
pool fixes context switching" would not survive a question from the audience;
"the pool removes creation overhead and reveals the memory wall" does.

Act 1 will **not** reproduce the shape of the CppCon slides, and shouldn't: their
work item is a 10 ms blocking sleep, so their curve peaks and collapses. Ours is
50 ns of arithmetic, so it is a flat line rate-limited by the spawner.

**Open decisions**

- A roofline chart is **not** built. Recommended as a backup slide rather than one
  of the main four: it turns the measurement into a prediction (0.25 flops/byte →
  speedup should equal the bandwidth ratio; 448/49 = 9.1 predicted vs 8.8
  measured), which is useful when re-running on new hardware.

## Porting to H100 and GH200

`make` uses `-arch=native`, so the build needs no change on either machine. Three
things do:

1. **Increase `--nodes`.** The kernel is 1.33 ms on an A4000 and would be ~0.2 ms
   on H100 — too short to time cleanly. 80 GB of HBM has room for a much larger
   graph. Scale until the kernel is at least a few ms.
2. **Reinstall TBB** (`conda install -c conda-forge tbb-devel`) or point
   `make TBB_ROOT=...` at it. Without it `std::execution::par` silently runs
   sequentially; `make tbb-check` will tell you.
3. **Expect the CPU baseline to rise.** A server host with 8–12 memory channels
   does 200–400 GB/s, not this desktop's ~49, which narrows the GPU ratio. The
   *shape* of figures 1 and 2 holds regardless; only the ceiling moves.

### The two machines tell different halves of the story

Running on both is worth doing, because they disagree about act 3 — and that
disagreement **is** the argument.

| | this box (PCIe 3/4) | H100 PCIe | GH200 (NVLink-C2C) |
|---|---:|---:|---:|
| host→device link | 12.3 GB/s *(measured)* | ~55 GB/s | ~450 GB/s per direction |
| device memory | 448 GB/s | 3350 GB/s | 3350 GB/s |
| copy 512 MB in | 43.8 ms | ~9 ms | ~1.1 ms |
| kernel over 544 MB | 1.33 ms | ~0.16 ms | ~0.16 ms |
| **transfer share of a naive port** | **97%** | **~98%** | **~88%** |

Two things to take from that row of percentages:

- **On H100 PCIe the problem gets *worse*, not better.** The link roughly
  quadruples, but the kernel gets ~8× faster, so copying dominates even harder.
  A faster bus does not rescue a copy-per-call design.
- **Even C2C's 36× faster link doesn't fully fix it.** GH200 drops the transfer
  share from ~97% to ~88% — a huge improvement, and still transfer-dominated. The
  fix was never a faster copy. It is **not copying**: keeping the graph resident,
  and using coherence so the CPU can touch results without a round trip.

That is what `gpu_managed` in `data/gpu_sta.csv` measures — `cudaMallocManaged`
with no explicit copies at all. On this PCIe box it lands at **42.6 GB/s**: much
better than staged copies (11.9) because the driver migrates only what is touched,
but still just under the CPU, because every page still crosses PCIe on demand. On
GH200 the same code path runs over a coherent link at ~450 GB/s, and that number
should move a great deal. **It is the single most interesting number to re-measure
on the new hardware.**

If the GH200 result lands where the hardware suggests, act 3 stops being "GPUs are
faster" and becomes "the interconnect was the whole problem, and this is the
machine built to admit that."

## The kernel

Block-based STA propagates arrival times forward through the timing graph:

![The STA propagation kernel](figures/00_problem.png)

One work item is one node: take each of its fanin edges, add the edge delay to
the arrival time coming in, keep the max. That is what a single `std::thread`
does in act 1, and what a single GPU thread does in act 3 — the same function,
`sta::propagate`, compiled by both toolchains.

Two large vectors, an elementwise add, a max-reduction over each node's fanin
window. 8M nodes × 8 fanin = **64M timing edges, 544 MB working set** — 22× this
machine's L3, so nothing is hiding in cache.

Per node it moves 68 bytes and does 15 flops: **~0.25 flops/byte**. This kernel is
bound by memory bandwidth, not arithmetic. That is the honest STA regime, and it
is why every result below tracks bandwidth ratios rather than FLOP ratios.

## Results on this machine

i5-13500 (6 P + 8 E = 14 physical / 20 logical) · RTX A4000 (48 SM, 448 GB/s) ·
CUDA 12.6 · gcc 13.3 · TBB 2022

| | achieved GB/s | vs. best CPU |
|---|---:|---:|
| one `std::thread` per work item | 0.009 | 0.0002× |
| 1 CPU thread | 22.2 | 0.46× |
| CPU, saturated plateau (6+ threads) | ~47–48 | 0.97× |
| **CPU, best observed** | **48.9** | **1.0×** |
| GPU, naive offload (copy in, run, copy out) | 11.9 | **0.24×** |
| GPU, unified memory (no explicit copies) | 42.6 | 0.87× |
| GPU, data resident | 429.7 | **8.8×** |

Three findings, in the order the figures present them:

- **The CPU ceiling is the memory bus.** Throughput saturates at **5–6 threads**.
  Fourteen physical cores buy **2.2×** over one thread. Cores 7–14 contribute
  nothing — they are queued on the same memory controller. (Don't quote a "best at
  N threads": above the knee the curve is flat and its argmax wanders between ~9
  and ~24 run to run. Quote the plateau.)
- **The GPU wins by exactly the bandwidth ratio.** 429.7 GB/s is 96% of the card's
  448 GB/s peak; 429.7 / 48.9 = 8.8×. Not a coincidence, and not a FLOP story.
- **A naive port is 4× slower than the CPU.** Copy-in/run/copy-out achieves
  11.9 GB/s, well below the CPU's 48.9. The breakdown says why:

| phase | ms | share |
|---|---:|---:|
| copy in (H2D) | 43.8 | **91%** |
| **compute** | **1.3** | **3%** |
| copy out (D2H) | 2.8 | 6% |

Host staging is **pinned** memory, so that is PCIe's best case (12.3 GB/s measured).
The conclusion is robust to a faster link: at full PCIe 4.0 (~25 GB/s) transfer
would still be ~94% of the time.

CPU and GPU produce **bit-identical results** (checksum `9386759311749429`) —
`-ffp-contract=off` on the host, `--fmad=false` on the device.

### The `std::execution::par` trap

libstdc++ implements C++17 parallel algorithms on Intel TBB. **Without TBB linked,
`par` and `par_unseq` silently run sequentially** — no warning, no error, no link
failure. Measured here before and after installing it:

```
                    without TBB        with TBB
execution::seq      3.05 s / 1 thread  3.05 s / 1 thread
execution::par      3.02 s / 1 thread  0.18 s / 20 threads
```

`make tbb-check` reports which one you have. Install without sudo:
`conda install -c conda-forge tbb-devel`.

### Honest caveats

- **Fanin is fixed at 8 and bucketed by degree.** Real timing graphs are irregular.
  That irregularity hurts the CPU *more* (gather, pointer chasing, branch misses),
  so this is the charitable case for the CPU. `--layout aos` shows what a
  less-friendly layout costs both sides.
- **A many-channel server CPU narrows the gap a lot.** A 12-channel Genoa does
  ~400 GB/s and lands near this GPU. The *shape* of figure 1 holds on any machine;
  only the height of the ceiling moves.
- **This is full re-propagation, not incremental STA.** Incremental work is smaller
  and latency-bound, which makes the per-unit transfer cost worse, not better.
- For contrast, the same GPU gives **72×** on a compute-bound Mandelbrot kernel
  (`make ref`). The difference between 72× and 8.8× is arithmetic intensity —
  worth knowing, even if it doesn't make the slide deck.

## Figures

Every figure below regenerates from `data/*.csv`. Dark versions live in
`figures/dark/`.

### 1. The CPU ceiling is the memory bus, not the core count

![CPU ceiling](figures/01_sta_cpu_ceiling.png)

Throughput saturates at 5–6 threads. The other eight cores are queued on the same
memory controller.

### 2. A thread pool buys 5,000×, and still can't use half the CPU

![Naive vs pool](figures/02_naive_vs_pool.png)

Act 1 against act 2. Deleting the obvious overhead — thread creation — bought four
decades, and the wall underneath did not move.

### 3. Same kernel, same numbers, bit-for-bit identical results

![CPU vs GPU](figures/03_cpu_vs_gpu.png)

The GPU is 8.8× the whole CPU with data resident. Copy in and out per call and it
is 4× *slower* than the CPU. Unified memory recovers most of that and still lands
below the CPU, because every page it touches still crosses PCIe.

### 4. 97% of a naive GPU port is PCIe

![Where the time goes](figures/04_where_the_time_goes.png)

The timing analysis itself is 3% of the wall clock.

### 5. One thread per node never becomes viable

![Runtime vs size, CPU](figures/05_runtime_vs_size_cpu.png)

Runtime against problem size for the three CPU strategies. The thread-per-node
line is measured up to 1M nodes; beyond that a single pass takes minutes. All
three lines compute exactly the same arithmetic.

### 6. The copy costs more than the whole CPU does

![Runtime vs size, GPU](figures/06_runtime_vs_size_gpu.png)

The dashed GPU line sits above the CPU line at *every* problem size. The gap
between the two GPU lines is the copy, and it never closes — it is proportional
to the data, exactly like the work is.

## Running it

```sh
./run_all.sh            # ~15 min: builds, sweeps, writes data/*.csv
python3 plot.py         # figures 01-06
python3 draw_problem.py # the problem-formulation diagram
```

`run_all.sh` sets `ulimit -s 1024` — the blocking-I/O contrast sweep holds 4096
live threads and glibc sizes thread stacks from `RLIMIT_STACK`.

Needs `nvcc`, TBB (see above), and Python with matplotlib. `make sta` builds only
the STA benchmarks; `make ref` builds the Mandelbrot reference.

## Layout

```
src/sta.hpp             the STA kernel, compiled by BOTH g++ and nvcc
src/bench_sta.cpp       CPU: std::for_each(par_unseq), pool size via TBB
src/bench_sta_gpu.cu    GPU: kernel, naive offload, transfer breakdown
src/workload.hpp        compute-bound Mandelbrot reference (roofline contrast)
src/bench_threads.cpp   throughput vs thread count (--mode io | cpu)
src/bench_gpu.cu        Mandelbrot on the GPU
plot.py                 CSVs -> figures 01-06
draw_problem.py         the problem-formulation diagram (00)
```

There is no hand-written thread pool: the CPU benchmark uses `std::for_each` with
`std::execution::par_unseq`, and sweeps pool size with
`tbb::global_control(max_allowed_parallelism, K)`. With the standard parallel
algorithms, "thread count" and "pool size" are the same knob.
