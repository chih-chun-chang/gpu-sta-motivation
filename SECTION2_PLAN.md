# Section 2 — GPU architecture — 10 minutes

## The job of this section

Section 1 ended with a measured fact: the GPU speedup **equals the ratio of memory
bandwidths**, not of FLOPs. So this section answers exactly one question:

> **Why can a GPU move data ~9× faster than a CPU?**

Keep that question on screen. It stops the section becoming a general GPU tour,
and it makes every slide load-bearing. A timing audience does not need to know
what a tensor core is; they need to know why the bandwidth is there.

**Numbers to pull from the Hopper whitepaper** (marked **[WP]** below) rather than
from me: SM count, warps and registers per SM, cache sizes, HBM3 configuration.
The latency figures I use are illustrative and I have labelled them as such.

---

## Slide 1 — The question, restated

**Bullets**
- Section 1: GPU is 8.8× the CPU, and 448/49 ≈ 9. Not a coincidence.
- Speedup = bandwidth ratio ⇒ the question is *why the bandwidth*
- Two parts: **how wide the pipe is**, and **how you keep it full**

**Script**

> "We measured that the GPU wins by about nine times, and that nine is the ratio
> of the two memory bandwidths — not the ratio of their FLOPs, which is about
> four hundred. So the only question that matters for this workload is: why does
> a GPU have nine times the memory bandwidth?
>
> There are two answers, and the second one is the interesting one. The first is
> that the pipe is physically wider. The second is that a wide pipe is useless
> unless you can keep it full — and *that* is what the whole SM design is for."

---

## Slide 2 — Part one: the pipe is wider

**[WP] diagram:** the memory subsystem / die shot.

**Bullets**
- CPU: dual-channel DDR5, **128-bit** interface
- RTX A4000: GDDR6, **256-bit**
- H100: HBM3, **[WP] ~5120-bit** — stacked memory sitting on the package
- Wider bus + faster memory technology = the raw ~9× and ~80×

**Script**

> "Physically, this is unglamorous. A desktop CPU talks to two channels of DDR5 —
> a 128-bit interface. The A4000 has a 256-bit GDDR6 interface. An H100 has HBM3
> stacked on the package with a bus thousands of bits wide.
>
> That is most of the raw number. Wider bus, faster memory, memory physically
> closer to the chip. There's no trick here — you paid for it."

**Do not linger.** This part is intuitive and the audience will accept it in
thirty seconds. The next slide is the one worth your time.

---

## Slide 3 — Part two: a wide pipe is useless unless you keep it full

**This is the hinge of the section.** Little's Law:

```
bytes in flight = bandwidth × latency
```

To *sustain* a bandwidth you must have that many bytes **outstanding** at all
times — requests issued and not yet returned. Latencies below are illustrative
(±, order-of-magnitude), the bandwidths are measured:

| | bandwidth | latency | bytes that must be in flight |
|---|---:|---:|---:|
| i5-13500, DDR5 | 49 GB/s | ~90 ns | **~4 KB** |
| RTX A4000, GDDR6 | 448 GB/s | ~450 ns | **~200 KB** |
| H100, HBM3 | 4000 GB/s | ~550 ns | **~2 MB** |

**Bullets**
- GPU memory latency is *worse* than CPU — roughly 5× worse
- So to sustain 9× the bandwidth it needs ~50× more requests outstanding
- A CPU core can hold ~10–16 outstanding misses. 14 cores ≈ a few hundred.
- The GPU needs **thousands**. That requirement is what shapes the SM.

**Script**

> "Here's the part people find surprising. GPU memory latency is *worse* than a
> CPU's — a few hundred nanoseconds against about ninety. The GPU is not faster
> at answering any single request. It's slower.
>
> [PAUSE]
>
> What it's good at is having enormous numbers of requests outstanding at once.
> Little's Law: to sustain a bandwidth, the amount of data in flight has to be
> bandwidth times latency. For this CPU that's about four kilobytes. For the
> A4000 it's two hundred kilobytes. For an H100, two megabytes — permanently in
> flight, just to keep the bus busy.
>
> A CPU core can track something like a dozen outstanding cache misses. Fourteen
> cores gets you a few hundred. It is not physically capable of having two
> megabytes in the air. So the CPU cannot use that bandwidth even if you gave it
> the bus.
>
> Everything about the SM follows from needing thousands of requests outstanding."

---

## Slide 4 — That requirement *is* the SM design

**[WP] diagram:** the SM block diagram.

Now the architecture explains itself. Each feature answers "how do we keep
thousands of requests in flight?":

| feature | why it exists |
|---|---|
| **[WP]** many warps resident per SM | each can have loads outstanding |
| **[WP]** enormous register file | a warp keeps its state in registers, so it can be parked mid-load |
| warp scheduler picks a ready warp each cycle | switching costs **zero** cycles — no stack, no OS, no save/restore |
| **[WP]** many SMs | multiply the whole thing |
| small caches per thread | a GPU is not trying to *avoid* the memory access, only to overlap it |

**Script**

> "So look at the SM with that question in mind, and it stops being a list of
> features and becomes one idea.
>
> Why are there so many warps resident? So each can have a load outstanding.
> Why is the register file enormous — bigger than the L1 cache, which never
> happens on a CPU? Because a warp parked waiting on memory keeps all its state
> in registers, so switching to another warp costs *nothing*. No stack to save,
> no kernel involved. On a CPU a context switch costs microseconds; here it costs
> zero cycles.
>
> That's the trade. A CPU spends its transistors on cache and out-of-order
> execution to *avoid* stalling. A GPU accepts the stall and switches to another
> warp. It doesn't avoid latency — it hides it."

---

## Slide 5 — Why *our* kernel got 96% on one card and 71% on the other

This is where the section pays for itself: the theory predicts a number you
already measured in section 1.

**Bullets**
- Our kernel loads scalar floats: 32 lanes × 4 B = **128 B outstanding per warp**
- A4000 needs ~200 KB ÷ 128 B ≈ **1,600 warps**; it holds **[WP] ~2,300** ✓
- H100 needs ~2 MB ÷ 128 B ≈ **17,000 warps**; it holds **[WP] ~8,400** ✗
- ⇒ on H100, thread-level parallelism alone is not enough. You also need wider
  loads per thread (`float4`) so each warp has more bytes in flight.
- **Measured: 96% of peak on the A4000, 71% on the GH200.** That is the gap.

**Script**

> "And this predicts something we already measured. Our kernel loads plain floats
> — thirty-two lanes times four bytes, so one hundred and twenty-eight bytes
> outstanding per warp.
>
> On the A4000 you need about sixteen hundred warps in flight to saturate, and
> the card holds around twenty-three hundred. Comfortable — and we measured
> ninety-six percent of peak bandwidth.
>
> On an H100 you'd need seventeen thousand, and it holds about eight thousand.
> It doesn't fit. You can't get there with more threads — you need each thread to
> have more bytes in the air, which means vector loads.
>
> We measured seventy-one percent on the GH200. That's the gap, and now you know
> what it is."

> ⚠️ Verify the warp-residency numbers against the whitepaper before presenting,
> and re-measure the GH200 percentage if you change the kernel. The *shape* of the
> argument is solid; the exact warp counts should come from **[WP]**.

---

## Transition into Section 3

> "So the bandwidth is there, and the machine is built to keep it busy. But it
> only works if the thirty-two lanes of a warp ask for thirty-two *consecutive*
> addresses. Which brings us to how you actually write this."

That hands straight to coalescing, which is section 3's first point — and it is
also why the benchmark stores fanin slot *k* contiguously across nodes.

---

## If you are running short

Cut in this order:
1. **Slide 2** (the physical bus) — one sentence covers it: "wider bus, faster
   memory, closer to the die."
2. The register-file detail on slide 4 — keep the diagram, drop the numbers.
3. **Never cut slides 3 or 5.** Slide 3 is the idea, slide 5 is the payoff that
   connects it to your own measurement.

## The one sentence for the section

> *A GPU is not faster at memory. It is slower — and it wins by having thousands
> of requests outstanding at once.*
