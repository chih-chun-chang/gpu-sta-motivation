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
- ≈ **0.22 flops per byte** — machines need 17–43 to break even

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
> operations. That's about a fifth of a flop per byte.
>
> And the reason that matters: every byte here is read exactly once. There's no
> reuse — each timing edge belongs to exactly one node. Nothing to cache, nothing
> to tile. So that ratio is fixed, no matter how big the design gets.
>
> These machines need somewhere between seventeen and forty flops per byte before
> arithmetic starts to be the limit. We have a fifth of one.
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

### "Why is it memory bound?" — the three answers, shortest first

This is the question a GPU person in the room will ask, and you should be able to
answer it at whatever depth they want.

**1. The intuition (say this first — it is the real reason).**

> "Every byte is read exactly once. There is no reuse to exploit — each timing
> edge is touched by exactly one node, so there is nothing to cache, nothing to
> tile, nothing to block. Compare it with matrix multiply, where you do N³ work on
> N² data: there the intensity *grows* with the problem, so you can tile your way
> into being compute bound. Here the intensity is a constant, and it's tiny. You
> cannot make this compute bound by making the problem bigger."

**2. The arithmetic (if they want the number).**

Per node: 8 arrival loads + 8 delay loads + 1 store = **68 bytes**, against 8 adds
and 7 maxes = **15 flops**. That is **0.22 flops/byte**. (If you only count the
adds as real FLOPs it is 0.12 — even further down.)

Now compare it against *machine balance* — peak FLOP/s ÷ peak bytes/s, the
intensity at which a machine stops being memory limited:

| machine | balance | kernel is below it by |
|---|---:|---:|
| i5-13500 (~1.4 TF FP32, 49 GB/s) | 28 flops/byte | 127× |
| RTX A4000 (19.2 TF, 448 GB/s) | 43 flops/byte | 194× |
| H100 / GH200 (67 TF, 4023 GB/s) | 17 flops/byte | 75× |

> "You need something like seventeen to forty flops per byte before arithmetic
> starts to matter on these machines. We have a fifth of one. We're two orders of
> magnitude into the memory-bound regime — it isn't close."

Note the direction: the **H100 has the *lowest* balance** of the three, because its
bandwidth grew faster than its FP32 throughput. Newer hardware doesn't rescue
you here.

**Backup slide: `figures/07_roofline.png`.** Do not show this unless asked — it
costs 90 seconds and a concept. But if a GPU person pushes, it is the complete
answer on one slide:

> "Left panel: intensity as a function of fanin. It's (2K−1) over (8K+4), which
> tends to one quarter. So there is no fanin — none — that gets this kernel above
> 0.25 flops per byte. Widening the window doesn't help.
>
> Right panel: the shaded band is *every* intensity this kernel can reach, on the
> roofline for all three machines. It never touches a ridge point. And the
> diamonds are the measured throughput — they sit on the sloped memory roofs.
> That's not a prediction, that's where the runs landed."

**If they say "your kernel is too simple — real STA does more than an add and a
max".** This is the strongest form of the objection, and it is answered by
measurement, not argument:

> "Agreed, so we implemented the statistical propagation from INSTA — means add,
> standard deviations combine in quadrature with a square root, both rise and
> fall transitions, sense inversion, ranked by mean plus sigma times sigma. Seven
> times the arithmetic per edge.
>
> The arithmetic intensity went from 0.22 to 0.40. Because every value became a
> mean-and-sigma pair and there are two transitions, so the data grew nearly as
> fast as the maths did.
>
> On the same card that kernel achieves ninety-six percent of peak memory
> bandwidth — and nine tenths of one percent of peak FLOPs. It's in the repo."

The ceiling over *all* fanin is 14/33 = 0.424, still an order of magnitude below
the friendliest ridge point. Details in `ssta/README.md`.

**3. The proof (if someone is still unconvinced — this ends it).**

> "We don't actually have to argue about it. The kernel achieves 430 gigabytes a
> second against a card peak of 448 — ninety-six percent of the memory bandwidth
> of the device. You cannot saturate memory bandwidth *and* be compute bound.
> That measurement is the answer."

### Likely questions in this section

**"Your CPU baseline isn't optimised."** It's `par_unseq`, so it's threaded and
vectorised by the compiler. And it's already at 49 GB/s of streaming bandwidth —
SIMD makes arithmetic faster, and arithmetic isn't the bottleneck.

**"A server CPU has more memory channels."** True, and it narrows the gap — Grace
does roughly eight times less than its GPU rather than sixty. The *shape* of the
first two slides holds on any machine; only the height of the ceiling moves.

**"Real fanin is irregular."** Correct, and this is the friendly case. Irregular
fanin hurts the CPU more — gather, pointer chasing, branch misprediction.
