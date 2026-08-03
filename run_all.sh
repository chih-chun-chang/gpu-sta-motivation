#!/usr/bin/env bash
# Run every benchmark and drop CSVs into data/. Takes roughly 10 minutes.
set -euo pipefail
cd "$(dirname "$0")"

# The blocking-I/O contrast sweep holds 4096 live threads. glibc sizes each
# thread's stack from RLIMIT_STACK, so the 8 MB default would reserve 32 GB of
# address space. 1 MB is plenty for these workers.
ulimit -s 1024

mkdir -p data
make all

echo "==> [0/4] confirming std::execution::par is not silently sequential"
make tbb-check

echo "==> [1/4] STA propagation on the CPU, vs pool size  (the bandwidth ceiling)"
./bin/bench_sta --trials 3 --reps 5 > data/sta_cpu.csv

echo "==> [2/4] STA propagation on the GPU, incl. transfer breakdown"
./bin/bench_sta_gpu --reps 20

echo "==> [3/4] blocking I/O contrast, as measured in the CppCon talk"
./bin/bench_threads --mode io --trials 3 > data/threads_io.csv

echo "==> [4/4] compute-bound reference (Mandelbrot), for the roofline contrast"
./bin/bench_threads --mode cpu --trials 3 --measure-ms 500 > data/threads_cpu.csv
./bin/bench_gpu --iters 50 > data/gpu_mandelbrot.csv

echo
echo "CSVs written to data/. Now run: python3 plot.py"
