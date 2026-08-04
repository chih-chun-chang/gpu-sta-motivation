# Opening segment — speaker notes

Four figures, ~6–8 minutes, in three acts: *one thread per work item → a thread
pool → the GPU*, measuring at each step what is actually limiting you. The arc is:
*you already can't use the CPU you have → the GPU is genuinely faster → but the
obvious way to use it is slower than doing nothing → so this talk is about
architecture, not kernels.*

Numbers below are from this machine (i5-13500, 14 cores; RTX A4000). Re-run
`./run_all.sh` and update them if you present on different hardware — every
figure regenerates from the CSVs.

---

## Before you start: the setup line

> "Static timing analysis propagates arrival times forward through the timing
> graph. For every node: take each fanin edge, add the edge delay to the arrival
> time coming in, and keep the max. That's it. That's the kernel. Two big
> vectors, an add, and a max."

Put the recurrence on screen and leave it up. Everything after is this one line.

```
arrival[i] = max over fanin k of ( arrival_in[i][k] + delay[i][k] )
```

**Why this framing works:** every timing person in the room recognises it
instantly, and it is genuinely embarrassingly parallel — so when it *still*
doesn't scale, the surprise lands.

---

## Figure 1 — `01_sta_cpu_ceiling.png`

**The one sentence:** *"I have fourteen cores. This gets me 2.2× out of them."*

Talking points:

- 64M timing edges, 544 MB working set — a real design, not a cache-resident toy.
- The code is `std::for_each(std::execution::par_unseq, …)`. Nothing hand-rolled,
  no thread pool of my own. This is what you'd actually write.
- Walk the curve: 1 thread → 22 GB/s. Then it flattens **at 5 or 6 threads**, at
  ~49 GB/s. Cores 7 through 14 do nothing. (Don't say "it peaks at N threads" —
  above the knee the curve is flat and the argmax moves run to run.)
- The punchline: the flat line is not a scheduling problem. Every thread is
  RUNNING. They are all queued on the same memory bus.

**Pause here.** This is the slide the rest of the talk hangs on.

> "You do not have a parallelism problem. You have a bandwidth problem. And you
> hit it before you'd finished spawning threads."

---

## Figure 2 — `02_naive_vs_pool.png`

**The one sentence:** *"A thread pool buys you five thousand times. It does not buy
you the machine."*

This is act 1 meeting act 2. Both axes are log, because the gap is four decades.

- The orange dashed line is one `std::thread` per work item: **0.009 GB/s**. Flat,
  and it would stay flat however many you spawn — you are rate-limited by the
  thread you're spawning *from*.
- Why: a `clone()` costs about 7.2 µs. The work inside is about 1.4 ns. You are
  measuring the operating system, not timing analysis. **5,200:1 overhead** — and
  that ratio is exactly the gap drawn on the chart.
- The blue curve is the same kernel through `std::for_each(par_unseq)`. 5,000×
  better — and then it stops at 6 of your 14 cores.
- The move that matters: *"we deleted the obvious overhead and got five thousand
  times. And we still can't use half the CPU. So the obvious overhead was never
  the real problem."*

That last line is the hinge of the whole opening. Say it slowly.

**Pre-empt the wrong diagnosis** — someone will suggest context switching, and it
is worth being able to shut this down with data:

> "You'd think this was context switching. It isn't. At 512 compute threads this
> machine does about 10,000 context switches a second — the idle baseline is
> 6,400. Throughput doesn't move at all. A compute thread only switches when its
> time slice expires. It's not the scheduler. It's the memory bus."

---

## Figure 3 — `03_cpu_vs_gpu.png`

**The one sentence:** *"The GPU is 8.8× faster, and the obvious way to use it is
4× slower than not using it at all."*

- Same kernel. Same inputs. Bit-for-bit identical results — the checksums match
  exactly, and that's deliberate: `-ffp-contract=off` on the host, `--fmad=false`
  on the device. Say this; it kills the "are you even computing the same thing"
  question before it's asked.
- GPU with data resident: ~430 GB/s, about 96% of the card's 448 GB/s peak.
  Against the CPU's 49 GB/s that's **8.8×** — and notice that's almost exactly the
  ratio of the two memory bandwidths. Not a coincidence. This kernel moves 8 bytes
  per node per edge and does one add. It is bandwidth, all the way down.
- Then point at the third bar, the one **shorter than the CPU bar**: copy the
  vectors in, run the kernel, copy the answer back — 11.9 GB/s. You have just
  bought a GPU to make your tool 4× slower.
- The fourth bar is unified memory — no explicit copies, let the driver migrate
  pages on demand. 42.6 GB/s: 3.6× better than staged copies, and *still* just
  under the CPU, because every page it touches still crosses PCIe. Worth one
  sentence: **"and no, just not writing the memcpy doesn't save you either."**

Let that sit for a beat before figure 4.

---

## Figure 4 — `04_where_the_time_goes.png`

**The one sentence:** *"97% of that was PCIe. The timing analysis was 3%."*

- 43.8 ms copying in. 1.3 ms computing. 2.8 ms copying back.
- Pre-empt the obvious objection yourself: *"and no, a faster link doesn't save
  you — at full PCIe 4.0 speed this is still about 94% transfer. On an H100 over
  PCIe 5 it gets"* — pause — *"worse. About 98%. Because the link roughly
  quadruples and the kernel gets eight times faster."*
- If you have GH200 numbers by then, this is where they land: NVLink-C2C is ~36×
  this machine's link and *still* leaves you transfer-dominated (~88%). Which is
  the cleanest possible proof that the answer was never a faster copy.
- Land the thesis:

> "So GPU-accelerated STA is not a kernel-porting problem. Every kernel in this
> talk is ten lines. The problem is keeping the graph resident, batching the
> irregularity, and never moving data you don't have to. That's what the rest of
> this talk is about."

---

## Questions this audience will ask

**"Your CPU baseline is unoptimised — what about AVX-512 / hand-tuned SIMD?"**
It's `par_unseq`, so it's threaded *and* vectorised by the compiler. But more to
the point: it's already at 49 GB/s of streaming bandwidth. SIMD makes arithmetic
faster, and arithmetic isn't the bottleneck — you can't vectorise your way past
the memory controller. (If pushed: a perfect SIMD version moves the same bytes.)

**"Why not a server CPU with 8 or 12 memory channels?"**
Fair, and it genuinely narrows the gap — that's the honest answer. A 12-channel
Genoa does ~400 GB/s and lands near this GPU. The argument then becomes cost,
power, and what else the box can do. Don't oversell; the *shape* of figure 1
holds on any machine, only the ceiling height moves.

**"Real timing graphs have irregular fanin — your fanin is fixed at 8."**
Correct, and it's a simplification I'll own. The layout here is bucketed by fanin
degree, which is what GPU STA implementations actually do. Irregular fanin makes
the CPU *worse* (gather, pointer chasing, branch misses), so this is the
charitable case for the CPU. `--layout aos` in the repo shows what a
less-friendly layout costs both sides.

**"What about incremental STA? You rarely re-propagate the whole graph."**
Genuinely the strongest objection, and it's a good place to say "hold that
thought — it's [section N]." Incremental changes the arithmetic completely:
small dirty regions, latency-bound, and the PCIe cost per unit of work goes *up*.
That's an argument for residency, not against the GPU.

**"Is float enough? We need picosecond accuracy."**
Float has ~7 decimal digits. At nanosecond magnitudes that's sub-femtosecond
resolution on a single value. The real accuracy question is accumulation over
deep paths, which is a numerics discussion, not a precision-of-storage one — and
FP64 on this card runs at 1/64 rate, which is exactly why the choice matters.

**"You're measuring bandwidth, not timing analysis."**
Yes — deliberately. That IS the finding. If your kernel is 0.22 flops per byte,
the only number that predicts your runtime is bandwidth.

---

## If you're nervous

- The three numbers you must not fumble: **2.2×** (what 14 cores buy you),
  **8.8×** (GPU resident vs whole CPU), **97%** (PCIe share of a naive port).
  Everything else you can read off the slide.
- Every figure has its headline written into the chart title. If you lose your
  place, read the title out loud — it's a complete sentence and it's the point.
- The strongest moment is silence after the third bar in figure 3. Don't rush it.
- If a question derails you: *"That's exactly where this goes next"* is true for
  almost any objection here, because the whole talk is the answer to figure 4.
