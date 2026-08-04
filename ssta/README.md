# ssta/ — is statistical propagation still memory bound?

Answers the obvious objection to the block-based benchmark in `../src`: *"your
kernel is an add and a max — of course it's memory bound. Real STA does more
arithmetic than that."*

So we take the arithmetic from a real implementation and measure it.

## The kernel

Modelled on INSTA's `custom_ops/topk_arrival_kernel.cu`. Per candidate — one
fanin edge, one parent top-K slot — for **each of the two transitions**:

```
m   = p_mean + arc_mean               // means add
sd  = sqrt(p_std² + arc_std²)         // stds combine in quadrature
arr = m + sigma * sd                  // the ranking key
```

Sense matters: a positive-sense arc takes rise from the parent's rise and fall
from its fall; a negative-sense arc swaps them. Candidates are ranked by `arr`
and the largest kept.

That is 7 flops per edge per transition — an add, two multiplies, an add, a
**square root**, a multiply and an add — so **14 per edge**, against 2 for
block-based STA. Seven times the arithmetic.

INSTA keeps the top **K** with startpoint dedup, in two phases: build the
candidate arrays in global memory, then run K argmax passes over them. We keep
the top 1 and fuse both phases, which is the *arithmetically favourable* case —
materialising candidates and re-reading them K times only moves more data, so
real INSTA is further into the memory-bound regime than what is measured here,
not less.

## The answer: yes, still memory bound

|  | block STA | SSTA (INSTA) |
|---|---:|---:|
| bytes per node (K=8) | 68 | 280 |
| flops per node | 15 | 112 |
| **arithmetic intensity** | **0.221** | **0.400** |
| upper bound over all K | 0.250 | **0.424** |

Seven times the arithmetic bought **1.8× the intensity**, because every value is
now a `(mean, std)` pair and there are two transitions — the data grew almost as
fast as the maths did. Measured on the same RTX A4000:

|  | achieved BW | % of 448 GB/s peak | achieved FLOP/s | % of 19.2 TF peak |
|---|---:|---:|---:|---:|
| block STA | 429.7 GB/s | 95.9% | 95 GFLOP/s | 0.50% |
| **SSTA** | **431.9 GB/s** | **96.4%** | 172.8 GFLOP/s | **0.90%** |

**The statistical kernel uses 96% of the card's memory bandwidth and 0.9% of its
arithmetic.** Machine balance is 9.6–43 flops/byte; this kernel tops out at
0.424. It is not close, and no fanin changes that.

CPU side, same machine as the rest of the repo: best 34.3 GB/s, saturating
around 6 threads — the same shape as the block-based kernel, at a lower ceiling
(the `sqrt` and the extra streams cost some efficiency).

The result is in the roofline backup slide, `../figures/07_roofline.png`.

## Running it

From the repo root:

```sh
make ssta                                   # builds bin/bench_ssta and bin/bench_ssta_gpu

./bin/bench_ssta_gpu --nodes 4194304 --reps 20      # GPU; writes data/ssta_gpu.csv
./bin/bench_ssta     --nodes 4194304 --trials 3     # CPU thread sweep, CSV on stdout

python3 draw_roofline.py                    # regenerates figures/07_roofline.png
```

4.19M nodes × 8 fanin is a 1120 MB working set — larger than the block-based
benchmark at the same node count, because there are nine input arrays instead of
two. Keep it well past L3 or the measurement is meaningless.

Options: `--nodes`, `--reps`, `--trials`, and `-DSSTA_FANIN=n` at build time to
change the fanin (rebuild both binaries — they must agree).

The GPU binary checks its output against a host computation of the same kernel
over the first 65,536 nodes and prints `MATCH` or `MISMATCH`.
