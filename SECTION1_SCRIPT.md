# Section 1 — "Why a GPU at all" — slides and script

Five slides, ~8 minutes. Script is written to be **spoken**, not read: short
sentences, one idea each. `[PAUSE]` means stop talking for two seconds — they are
placed where the audience needs to look at the figure.

Numbers in **bold** are the ones you must not fumble. Everything else you can
paraphrase.

> ⚠️ One number needs confirming before you say it out loud: the Grace CPU
> figure on slide 4 (~380 GB/s) is *inferred* from your "3.3× slower" plot, not
> measured directly. Run `./bin/bench_sta --trials 3 --reps 5` on the GH200 and
> use the real peak. If you can't, say "roughly eight times on both machines"
> and don't quote the Grace number.

---

## Slide 1 — Static timing analysis, in one line

**Figure:** `figures/00_problem.png`

**Bullets**
- `arrival[i] = max over fanin k of ( arrival_in[i][k] + delay[i][k] )`
- Two large vectors in, an add, a max over each node's fanin window
- 8M nodes × 8 fanin = **64M timing edges**, 544 MB
- ≈ **0.25 flops per byte**

**Script** *(~75 s)*

> Let me start with the kernel at the heart of block-based timing analysis.
>
> For every node in the timing graph: take each of its fanin edges. Add the edge
> delay to the arrival time coming in. Keep the maximum. That's it. That's the
> whole recurrence.
>
> [PAUSE]
>
> Two large vectors — arrival times and delays. An elementwise add. Then a max
> over each node's fanin window. One value out per node.
>
> For this talk I'm using eight million nodes with a fanin of eight. That's
> sixty-four million timing edges, and about half a gigabyte of data per pass.
>
> Now, the number at the bottom of this slide is the one that decides everything
> that follows. Per node we move sixty-eight bytes and do fifteen floating-point
> operations. That's a quarter of a flop per byte.
>
> This kernel is not limited by arithmetic. It is limited by memory bandwidth.
> Hold on to that, because every result I'm about to show you follows from it.

---

## Slide 2 — Fourteen cores buy you 2.2×

**Figure:** `figures/01_sta_cpu_ceiling.png`

**Bullets**
- `std::for_each(std::execution::par_unseq)` — nothing hand-rolled
- One thread: 22.2 GB/s → best: **48.9 GB/s**
- Saturates at **5–6 threads**, on a 14-core CPU
- Every thread is RUNNING. They're queued on one memory bus.

**Script** *(~85 s)*

> So let's parallelise it. This is a fourteen-core desktop CPU, and the code is
> just `std::for_each` with a parallel execution policy. Nothing hand-rolled, no
> clever thread pool. What you'd actually write.
>
> [PAUSE — let them read the curve]
>
> One thread gets twenty-two gigabytes a second. Then it climbs... and it flattens
> at five or six threads, at about **forty-nine**.
>
> Cores seven through fourteen do nothing at all.
>
> [PAUSE]
>
> Fourteen cores bought me **two point two times**.
>
> And I want to be precise about why, because the obvious explanation is wrong.
> This is not a scheduling problem. Every one of those threads is RUNNING — none
> of them is blocked, none is waiting on I/O. I measured the context switches:
> at five hundred threads this machine does about ten thousand a second, and it
> does six and a half thousand sitting idle. It's noise.
>
> They're not fighting the scheduler. They're all queued on the same memory bus.

---

## Slide 3 — A thread pool buys 5,000×. It doesn't buy you the machine.

**Figure:** `figures/02_naive_vs_pool.png`

**Bullets**
- One `std::thread` per node: **0.009 GB/s**
- ~7.5 µs to spawn a thread, ~50 ns of actual work — **150 : 1**
- Thread pool: 48.9 GB/s — a **5,188×** improvement
- Same ceiling underneath. Still 6 of 14 cores.

**Script** *(~85 s)*

> Now, before someone asks — yes, I did try the naive thing first.
>
> The green line is one `std::thread` per node. Spawn a thread, compute one
> arrival time, exit. It gets nine *thousandths* of a gigabyte per second.
>
> [PAUSE]
>
> The reason is on the slide. Creating a thread costs about seven and a half
> microseconds. The work inside it takes about fifty nanoseconds. You're paying a
> hundred and fifty to one overhead. This benchmark isn't measuring timing
> analysis at all — it's measuring the operating system.
>
> So we move to a thread pool, and we get **five thousand times** faster.
>
> [PAUSE]
>
> And here's the part I want you to take away. We deleted the obvious overhead.
> We got four orders of magnitude. And we *still* can't use more than six of
> fourteen cores.
>
> So the obvious overhead was never the real problem. The wall underneath it
> didn't move at all.

---

## Slide 4 — The GPU is ~8× the whole CPU — on both machines

**Figure:** `figures/03_cpu_vs_gpu.png`

**Bullets**
- Same kernel, same source file, **bit-identical results** (checksum matches)
- Desktop + RTX A4000: **429.7 GB/s** = **8.8×** the whole CPU
- GH200: **2852 GB/s** ≈ **7.5×** the whole CPU
- Naive offload is **slower than the CPU** — on both

**Script** *(~100 s)*

> So we've run out of CPU. Let's use the GPU.
>
> Same kernel — literally the same source file, compiled by both toolchains. And
> the results are bit-identical: the CPU and GPU checksums match exactly. I turned
> off floating-point contraction on both sides specifically so I could make that
> claim. Same arithmetic, same answer.
>
> [PAUSE]
>
> With the data already on the device: **four hundred and thirty** gigabytes a
> second. That's **eight point eight times** the entire CPU.
>
> And notice that number is almost exactly the ratio of the two memory
> bandwidths. That's not a coincidence — it's a bandwidth-bound kernel, so the
> speedup *is* the bandwidth ratio. No FLOPs story here at all.
>
> I ran the same thing on a Grace-Hopper superchip. Roughly **eight times** there
> too.
>
> [PAUSE — then point at the short bar]
>
> Now look at this bar. That's the same GPU, doing the same work — but copying
> the data in, running the kernel, and copying the answer back. Every call.
>
> It is **four times slower than just using the CPU**.
>
> You have bought a GPU to make your tool slower. And before you assume that's a
> cheap-desktop problem — on the Grace-Hopper machine it's three point three times
> slower. Same result.

---

## Slide 5 — 96% of a naive port is data movement

**Figure:** `figures/04_where_the_time_goes.png`

**Bullets**
- Copy in **43.8 ms** · compute **1.3 ms** · copy out **2.8 ms**
- **97%** transfer on the PCIe desktop
- **96%** on GH200 — a 10× faster, cache-coherent link
- The link got 10× faster. The kernel got 6.6× faster. Nothing changed.

**Script** *(~110 s)*

> Here's where that time actually goes.
>
> Forty-four milliseconds copying data in. One point three milliseconds doing the
> timing analysis. Three milliseconds copying the answer back.
>
> [PAUSE]
>
> The timing analysis — the thing we came here to do — is **three percent** of the
> wall clock. Ninety-seven percent is moving data.
>
> Now, the natural reaction is: fine, get a faster link. So let's test that
> properly.
>
> [PAUSE]
>
> Grace-Hopper. NVLink C2C instead of PCIe. Cache-coherent. Ten times the
> bandwidth, measured — not spec-sheet, measured.
>
> Transfer share: **ninety-six percent.**
>
> [PAUSE — this is the moment, let it sit]
>
> It didn't move. And the reason it didn't move is that the link got ten times
> faster and the kernel got six and a half times faster at the same time. The
> interconnect and the compute scale together. You cannot buy your way out of this
> with a faster bus.
>
> I'll add one more, because I tried it: unified memory, no explicit copies at
> all, just let the hardware handle it. On the desktop that was three and a half
> times better than copying. On Grace-Hopper — the machine built for coherent
> access — it was *five times worse*.
>
> Which tells you the problem was never the link. Somebody still has to decide
> where the data lives.

---

## Transition into Section 2

> "So: the GPU is worth about eight times on this problem, and the reason is
> bandwidth, not clock speed. Which means if we want to use it well, we need to
> understand how it actually moves data. So let's look at what's inside one."

---

## Delivery notes

- **Total ≈ 7.5 minutes** at a measured pace. Leaves buffer in an 8-minute slot.
- **Rehearse slides 1 and 2 most.** Anxiety peaks in the first two minutes; once
  you reach slide 3 the numbers carry you.
- **The three big pauses** are on slide 3 ("the wall didn't move"), slide 4 (the
  short bar), and slide 5 ("it didn't move"). Those are the beats people
  remember. Do not talk over them.
- **If you're running long**, cut the unified-memory paragraph at the end of
  slide 5. It's the one thing here that isn't load-bearing.
- **Numbers you must know cold:** 2.2× · 5,000× · 8× · 96%. Everything else can
  be read off the slide.

### Likely questions in this section

**"Your CPU baseline isn't optimised."** It's `par_unseq`, so it's threaded and
vectorised by the compiler. And it's already at 49 GB/s of streaming bandwidth —
SIMD makes arithmetic faster, and arithmetic isn't the bottleneck.

**"A server CPU has more memory channels."** True, and it narrows the gap — Grace
does roughly eight times less than its GPU rather than sixty. The *shape* of the
first two slides holds on any machine; only the height of the ceiling moves.

**"Real fanin is irregular."** Correct, and this is the friendly case. Irregular
fanin hurts the CPU more — gather, pointer chasing, branch misprediction.
