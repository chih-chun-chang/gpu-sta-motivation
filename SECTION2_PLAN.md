# Section 2 — GPU architecture — 10 minutes

**Audience assumption: they know hardware, but not GPUs.** They are timing and
EDA people — comfortable with datapaths, pipelines, queues and latency, but they
have never written a kernel and do not know what a warp is. So build the
vocabulary before using it. Nothing in this section assumes prior CUDA.

That means the order is: **what it looks like → how it executes → why that gives
bandwidth → back to our number.** Little's Law is the payoff, not the opening.

**[WP]** marks figures and numbers to take from the Hopper whitepaper rather than
from me.

| slide | minutes | lands |
|---|---:|---|
| 1. What's inside the thing that did that | 2.0 | a GPU spends transistors differently |
| 2. The hierarchy, with numbers | 2.0 | thread → warp → SM → GPU |
| 3. SIMT: 32 threads, one instruction | 2.0 | the one genuinely alien idea |
| 4. Why this gives bandwidth | 2.5 | latency hiding, Little's Law |
| 5. Back to our 96% and 71% | 1.5 | theory predicts our measurement |

---

## Slide 1 — What's inside the thing that just did that?

Comes straight after section 1's speedup figure. **[WP]** die shot, or the classic
CPU-vs-GPU transistor-budget picture.

**Bullets**
- CPU: **14 big cores.** Most of the die is cache, branch prediction, out-of-order logic
- GPU: **[WP] 132 small SMs.** Most of the die is arithmetic units and registers
- Same transistor budget, opposite decision
- CPU makes *one* thread fast. GPU runs *many* threads at once.

**Script**

> "So let's look at what actually did that. And I want to start with the one
> design decision everything else follows from.
>
> [PAUSE on the two die pictures]
>
> These have comparable transistor counts. On the CPU, most of that area is not
> arithmetic — it's cache, branch predictors, out-of-order machinery. All of it
> exists to make a *single* instruction stream go as fast as possible.
>
> On the GPU, most of the area is arithmetic units and registers. Almost none of
> it is spent making any one thread fast. Individually, a GPU thread is *slower*
> than a CPU thread — lower clock, no out-of-order, no speculation.
>
> Same budget, opposite decision. The CPU optimises one worker. The GPU gives up
> on that and runs thousands of mediocre ones."

**Do not** mention warps, SMs or CUDA cores yet. One idea per slide.

---

## Slide 2 — The hierarchy, with numbers

**[WP]** the SM block diagram plus the full-chip diagram.

Build the vocabulary bottom-up. Every term gets a number so it stays concrete.

| level | what it is | **[WP]** roughly |
|---|---|---|
| **thread** | one lane of arithmetic — computes one node in our kernel | — |
| **warp** | **32 threads that execute together, always** | fixed at 32 |
| **block** | a group of warps that share fast scratchpad memory | up to 1024 threads |
| **SM** | the actual hardware unit: schedulers, registers, L1/shared | 132 per H100 |
| **grid** | your whole launch, spread across all SMs | millions of threads |

**Script**

> "Four words, and then we're done with vocabulary.
>
> A **thread** is one lane of arithmetic. In our kernel, one thread computes one
> node's arrival time — that's it.
>
> A **warp** is thirty-two threads that execute *together*. Not 'can' — always.
> Hold that; it's the next slide.
>
> An **SM**, streaming multiprocessor, is the real hardware unit. It has warp
> schedulers, a register file, and a scratchpad. Think of it as one core, except
> a core that keeps dozens of warps resident at once. An H100 has [WP] a hundred
> and thirty-two of these.
>
> And a **grid** is your whole launch. When we ran eight million nodes, that was
> eight million threads — the hardware just streams them through the SMs."

**The comparison that makes it click for this audience:**

> "A CPU core runs one thread and switches maybe every millisecond, and the switch
> costs microseconds. An SM keeps dozens of warps resident and switches between
> them *every cycle*, for free. That difference is the whole architecture."

---

## Slide 3 — SIMT: 32 threads, one instruction

This is the genuinely alien idea. Spend the time.

**Bullets**
- One instruction is fetched and issued for **all 32 lanes** at once
- You write scalar code for one node; hardware runs 32 nodes in lockstep
- **If lanes disagree on a branch, both sides execute** and the wrong lanes idle
- **If lanes read scattered addresses, one load becomes 32 loads**

**Script**

> "Here's the part that has no CPU equivalent.
>
> The hardware fetches *one* instruction and issues it to all thirty-two lanes of
> a warp simultaneously. You write ordinary scalar code — 'compute node i' — and
> thirty-two copies run in lockstep on thirty-two different nodes.
>
> That's the deal, and it has two consequences you cannot escape.
>
> First: if the thirty-two lanes hit a branch and disagree, the hardware runs
> *both* sides, with the wrong lanes switched off. You pay for both. That's called
> divergence.
>
> Second, and for us this is the important one: if those thirty-two lanes ask for
> thirty-two *consecutive* addresses, the memory system turns that into one wide
> transaction. If they ask for thirty-two scattered addresses, it becomes
> thirty-two separate ones — and you get a thirty-second of the bandwidth.
>
> That's called coalescing, and it is why data layout is a performance decision on
> a GPU rather than a matter of taste."

**Tie to your own code — say this, it is concrete and it is yours:**

> "It's why our benchmark stores fanin slot *k* contiguously across all nodes,
> rather than each node's eight edges together. Adjacent threads then read
> adjacent addresses."

---

## Slide 4 — Why this gives 9× the bandwidth

Now they have the vocabulary, so the real explanation lands.

**The surprise first:**
- **GPU memory latency is *worse* than a CPU's** — hundreds of ns vs ~90
- It is not faster at answering one request. It is slower.
- It wins by having *thousands* of requests outstanding at once

**Little's Law:** `bytes in flight = bandwidth × latency`

| | bandwidth | latency* | must be in flight |
|---|---:|---:|---:|
| i5-13500, DDR5 | 49 GB/s | ~90 ns | ~4 KB |
| RTX A4000, GDDR6 | 448 GB/s | ~450 ns | ~200 KB |
| H100, HBM3 | 4000 GB/s | ~550 ns | **~2 MB** |

\* latencies illustrative, order-of-magnitude; bandwidths measured

**Script**

> "So: why is the bandwidth there? Two reasons, and the second is the interesting
> one.
>
> The boring one: the bus is physically wider. Dual-channel DDR5 is a 128-bit
> interface; the A4000 is 256-bit GDDR6; an H100 has HBM stacked on the package
> with a bus thousands of bits wide. You paid for that.
>
> But a wide pipe is useless unless you keep it full. And here's the surprise —
> [PAUSE] — GPU memory latency is *worse* than a CPU's. Several hundred
> nanoseconds against about ninety. It is slower at answering any single request.
>
> To *sustain* a bandwidth you need that much data permanently in flight —
> bandwidth times latency. For this CPU it's four kilobytes. For an H100 it's two
> megabytes, continuously, just to keep the bus busy.
>
> A CPU core can track about a dozen outstanding cache misses. Fourteen cores gets
> you a few hundred. It physically cannot have two megabytes in the air — so it
> couldn't use that bandwidth even if you gave it the bus.
>
> Now look back at the last two slides. Why does an SM keep dozens of warps
> resident? So each can have loads outstanding. Why is the register file bigger
> than the L1 cache — which never happens on a CPU? So a warp waiting on memory
> can be parked for free and another swapped in the same cycle.
>
> The GPU doesn't avoid memory latency. It hides it. That's the whole machine."

**Analogy if the room looks lost** — one line, then move on:

> "A CPU is a chef who puts one dish in the oven and waits. A GPU is a kitchen with
> sixty dishes in sixty ovens — every oven is slow, but something is always coming
> out."

---

## Slide 5 — Which explains our 96% and 71%

The payoff: the architecture predicts a number already on screen from section 1.

**Bullets**
- Our kernel loads scalar floats: 32 lanes × 4 B = **128 B in flight per warp**
- A4000 needs ~200 KB ÷ 128 B ≈ **1,600 warps**; holds **[WP] ~2,300** ✓
- H100 needs ~2 MB ÷ 128 B ≈ **17,000 warps**; holds **[WP] ~8,400** ✗
- **Measured: 96% of peak on the A4000, 71% on the GH200**
- Fix is not more threads — it's more bytes per thread (`float4` loads)

**Script**

> "And this predicts something we already measured, before I knew why.
>
> Our kernel loads plain floats. Thirty-two lanes, four bytes each — a hundred and
> twenty-eight bytes outstanding per warp.
>
> On the A4000 you need about sixteen hundred warps in flight to saturate the bus,
> and the card holds around twenty-three hundred. It fits — and we measured
> ninety-six percent of peak bandwidth.
>
> On the H100 you'd need seventeen thousand, and it holds about eight thousand. It
> does not fit. And more threads won't save you, because the limit is warps
> *resident*. You need each thread to have more bytes in the air — vector loads.
>
> We measured seventy-one percent on the GH200. That's the gap, and that's what it
> is."

> ⚠️ Check warp-residency against **[WP]** before presenting. The argument is
> robust to the exact figures; the conclusion is not sensitive to a 20% error.

---

## Transition into Section 3

> "So that's the machine, and why it has the bandwidth. Which leaves the question
> of how you write something that keeps a hundred and thirty-two SMs busy — and
> what that costs you when your timing graph doesn't cooperate."

---

## If you are running short

1. **Slide 1** can compress to one sentence over the die shot.
2. The block/grid rows on slide 2 — warp and SM are the only two you need.
3. Drop the divergence half of slide 3, keep coalescing (you need it for §3).
4. **Never cut slides 4 or 5.** Slide 4 is the explanation; slide 5 is the proof
   that it explains *your* data.

## The one sentence for the section

> *A GPU is not faster at memory. It is slower — and it wins by having thousands
> of requests outstanding at once.*
