# GPU-accelerated STA — 60 minute talk plan

Status: skeleton. Sections 1–3 are drafted; sections 4–5 need your content
(marked **[YOU]**). See the bottom for exactly what I need from you.

---

## Time budget

Aim for **50 minutes of content and 10 of questions**. If the room is quiet you
can stretch; you can never claw time back.

| time | section | minutes | the one thing it must land |
|---|---|---:|---|
| 0:00 | 1. Why a GPU at all | 8 | You already can't use the CPU you have |
| 0:08 | 2. GPU architecture | 10 | A GPU is a bandwidth machine, not a fast CPU |
| 0:18 | 3. Programming model | 8 | SIMT: the cost model is divergence and coalescing |
| 0:26 | 4. SSTA-X / torch-SSTA | 14 | **[YOU]** your contribution |
| 0:40 | 5. INSTA (DAC'25) | 12 | **[YOU]** your contribution |
| 0:52 | wrap + questions | 8 | |

**Sections 4 and 5 are why you were invited.** Sections 2 and 3 are service to
the audience. If you are running late, cut from 2 and 3 — never from 4 and 5.

### Decide your cuts NOW, not on stage

Write these on a card. Knowing in advance exactly what you will drop is the
single most effective thing against panic mid-talk:

1. **First to go** — figures 5 and 6 from the opening (runtime vs problem size).
   Pure supporting evidence. Saves ~2 min.
2. **Second** — the memory-hierarchy detail in §2 (registers/shared/L2/HBM
   numbers). Keep the picture, drop the table. Saves ~3 min.
3. **Third** — the SIMT divergence example in §3. State the rule, skip the code.
   Saves ~3 min.

That is 8 minutes of slack you can release without touching your own work.

---

## Section 1 — Why a GPU at all (8 min)

Already built. Full per-figure notes in **[SPEAKER_NOTES.md](SPEAKER_NOTES.md)**.

| slide | beat |
|---|---|
| `00_problem.png` | the STA recurrence: two vectors, an add, a max |
| `01_sta_cpu_ceiling.png` | 14 cores buy 2.2×; saturates at 5–6 threads |
| `02_naive_vs_pool.png` | a pool buys 5,000× and still can't use half the CPU |
| `03_cpu_vs_gpu.png` | the GPU is ~8× the whole CPU — **on both machines** |
| `04_where_the_time_goes.png` | ...and 96% of a naive port is data movement |

Three numbers to know cold: **2.2×**, **~8×**, **96%**.

The two-machine result is your strongest card here. Say it explicitly:

> "This is a desktop with PCIe, and this is a Grace-Hopper superchip. Eight times
> on both. And on both, naively offloading is *slower* than just using the CPU."

**Transition into §2** — write this one out and say it as written:

> "So the GPU is worth about eight times, and the reason is bandwidth, not clock
> speed. Which means to use it well you need to know how it actually moves data.
> So let's look at what's inside one."

---

## Section 2 — GPU architecture (10 min)

**Fully drafted in [SECTION2_PLAN.md](SECTION2_PLAN.md).** Summary below.

Source: NVIDIA Hopper (H100) architecture whitepaper.

**The one sentence:** *a GPU is not a fast CPU, it is a bandwidth machine that
hides latency with parallelism.*

Suggested beats — resist adding more:

1. **The die shot / block diagram.** 132 SMs on H100. Point out it is mostly
   repeated identical tiles: that repetition *is* the architecture.
2. **One SM in detail.** Warp schedulers, CUDA cores, tensor cores, register
   file, shared memory/L1. The number that matters: registers are enormous
   compared with a CPU, because context switching between warps must be free.
3. **Latency hiding, not latency avoidance.** A CPU spends transistors on cache
   and out-of-order to *avoid* stalls. A GPU accepts the stall and switches warps.
   That is the whole philosophical difference, and it explains everything else.
4. **The memory hierarchy as a bandwidth ladder** — registers → shared/L1 → L2 →
   HBM → the host link. Tie each rung back to a number from section 1.

**Tie back to your own data — this is what makes it your talk and not a recap:**

> "Section 1 measured 2852 GB/s on this kernel against a 4023 GB/s peak. That's
> 71%. The gap is what the rest of this section is about."

**Transition into §3:**

> "That's the machine. Now, how do you actually express a timing graph in a way
> that keeps 132 of those SMs busy?"

---

## Section 3 — Programming model, SIMT (8 min)

**The one sentence:** *you write scalar code for one node; the hardware runs 32 of
them in lockstep — and the cost model follows from that.*

1. **Thread → warp → block → grid**, mapped onto your own kernel. You have the
   perfect example already: one thread computes one node's arrival time.
   Show `sta::propagate` — it is ten lines and it is real.
2. **The two rules that decide performance:**
   - **Coalescing** — 32 consecutive lanes should touch 32 consecutive addresses.
     This is exactly why the benchmark stores fanin slot *k* contiguously across
     nodes instead of each node's edges together. You have a `--layout` flag that
     measures the difference; quote it if you have the number.
   - **Divergence** — a branch that splits a warp costs you both sides. Relevant
     to STA because real fanin is irregular, which is why implementations bucket
     nodes by fanin degree.
3. **Occupancy in one line.** Enough warps in flight to hide memory latency;
   registers and shared memory are the budget.

Do **not** teach `__syncthreads`, streams, or the memory-fence model. Not in eight
minutes, and not needed for what follows.

**Transition into §4:**

> "So that's the hardware and the model. Everything so far has been about a kernel
> you could write in an afternoon. Real STA is not that — and that's what we
> worked on."

---

## Section 4 — SSTA-X and torch-SSTA (14 min) **[YOU]**

Skeleton to fill. This shape works for almost any systems paper:

1. **The problem, in one slide.** What statistical STA needs that block-based
   doesn't, and why it is expensive.
2. **Why the naive GPU port fails.** Ideally reuse section 1's finding — if data
   movement or irregularity was your obstacle too, the opening has already primed
   the audience and you get this for free.
3. **The key idea.** One slide, one diagram. If you can't draw it, it isn't the
   key idea yet.
4. **Results.** Speedup over what baseline, on what designs, at what accuracy.
   State the baseline explicitly — the first question will be "compared to what?"
5. **torch-SSTA** — why PyTorch as the substrate: autograd? ecosystem?
   differentiability for optimisation? Say the *why*, not just the *what*.

**Watch out:** if the speedups here are much larger than section 1's ~8×, explain
where the extra comes from (algorithmic change? better data reuse? lower
precision?). An unexplained jump from 8× to 100× invites scepticism at exactly
the moment you want belief.

---

## Section 5 — INSTA, DAC 2025 best paper (12 min) **[YOU]**

1. **Lead with the award.** One line, no false modesty: it buys you attention for
   the next eleven minutes. *"This work won best paper at DAC last year."*
2. **What INSTA does** that SSTA-X/torch-SSTA didn't — the delta is the story.
3. **The result that won it.** There is usually one. Lead with it, then explain.
4. **Limitations, in your own words.** Naming them makes you more credible, and it
   defuses the toughest question by asking it yourself.

---

## Closing (2 min)

Return to section 1's figure. The talk's arc in three lines:

> "We started with a kernel that a thread pool couldn't scale past 2.2×, that a
> GPU runs 8× faster, and that a naive port makes 4× slower. Everything since has
> been about closing that gap. The lesson isn't that GPUs are fast — it's that
> where your data lives is the design."

---

## If you're nervous

- **Script the first 90 seconds word for word** and rehearse only that. Anxiety
  peaks at the start; once you are past the opening you will be fine.
- **Write out the four transitions** (marked above). Transitions are where
  under-rehearsed talks stall, because they are the only unstructured moments.
- **Every slide title is a complete sentence.** Lose your place, read the title
  aloud — it is the point of the slide.
- **The cut list above is your safety valve.** You cannot run over if you have
  already decided what to drop.
- **"That's a good question, and it's exactly what section N covers"** is true
  for most questions in the first half.
- Rehearse **§4 and §5 twice as much as §2 and §3.** Background you can improvise;
  your own results you cannot afford to fumble.

---

## What I need from you

To draft sections 4 and 5 properly:

1. The **papers** (PDF, or abstract + results table) for SSTA-X, torch-SSTA, INSTA.
2. **Who is in the room** — EDA/timing engineers, students, GPU people? Changes
   how much of §2 and §3 you need. A timing audience needs more GPU; a GPU
   audience needs more STA.
3. Any **existing slides** to build on.
4. Whether **60 minutes includes Q&A** or is 60 + questions.
