# Why GPU-accelerated STA — the opening experiment

A toy benchmark for the opening of a talk on GPU-accelerated static timing
analysis. It borrows the *figure layout* from Conor Spilsbury's CppCon 2025
[*Threads vs. Coroutines*](https://github.com/CppCon/CppCon2025/blob/main/Presentations/Threads_vs_Coroutines.pdf)
(slides 5, 6, 23) and swaps in the STA propagation kernel, which turns the same
axes into an argument for parallel hardware instead of coroutines.

Talking points and anticipated audience questions: **[SPEAKER_NOTES.md](SPEAKER_NOTES.md)**.

## The kernel

Block-based STA propagates arrival times forward through the timing graph:

```
arrival[i] = max over fanin k of ( arrival_in[i][k] + delay[i][k] )
```

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
| 1 CPU thread | 22.2 | 0.46× |
| CPU, saturated plateau (6+ threads) | ~47–48 | 0.97× |
| **CPU, best observed** | **48.9** | **1.0×** |
| GPU, naive offload (copy in, run, copy out) | 11.9 | **0.24×** |
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

`figures/` (light) and `figures/dark/`. Present in this order:

| | |
|---|---|
| `01_sta_cpu_ceiling.png` | the CPU ceiling is the memory bus, not the core count |
| `02_io_vs_sta_contrast.png` | why the CppCon conclusion doesn't transfer to compute |
| `03_cpu_vs_gpu.png` | the GPU bar that is **shorter** than the CPU bar |
| `04_where_the_time_goes.png` | 97% of a naive port is PCIe |

## Running it

```sh
./run_all.sh      # ~10 min: builds, sweeps, writes data/*.csv
python3 plot.py   # writes figures/
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
plot.py                 CSVs -> figures
```

There is no hand-written thread pool: the CPU benchmark uses `std::for_each` with
`std::execution::par_unseq`, and sweeps pool size with
`tbb::global_control(max_allowed_parallelism, K)`. With the standard parallel
algorithms, "thread count" and "pool size" are the same knob.
