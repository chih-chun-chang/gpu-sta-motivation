CXX       ?= g++
NVCC      ?= nvcc
# `native` detects the installed GPU (sm_86 on RTX A4000, sm_90 on H100), so
# this builds unchanged when moving machines. Override for cross-compiling.
CUDA_ARCH ?= native

# TBB backs libstdc++'s parallel algorithms. Installed without sudo via
#   conda install -c conda-forge tbb-devel
# WITHOUT it, std::execution::par silently runs SEQUENTIALLY. Override
# TBB_ROOT if yours lives elsewhere.
TBB_ROOT ?= $(HOME)/miniconda3
TBB_FLAGS := -I$(TBB_ROOT)/include -L$(TBB_ROOT)/lib -Wl,-rpath,$(TBB_ROOT)/lib -ltbb

# -ffp-contract=off / --fmad=false keep host and device float rounding identical,
# so the CPU and GPU checksums can be compared exactly.
CXXFLAGS  := -O3 -march=native -std=c++20 -ffp-contract=off -pthread -Isrc
NVCCFLAGS := -O3 -std=c++17 --fmad=false -arch=$(CUDA_ARCH) -Isrc

BIN := bin

# The STA benchmarks are the talk. bench_threads/bench_gpu are the
# compute-bound Mandelbrot reference kept for the roofline contrast.
STA_TARGETS  := $(BIN)/bench_sta $(BIN)/bench_sta_gpu
REF_TARGETS  := $(BIN)/bench_threads $(BIN)/bench_gpu

.PHONY: all sta ref clean tbb-check
all: sta ref
sta: $(STA_TARGETS)
ref: $(REF_TARGETS)

$(BIN):
	mkdir -p $(BIN)

$(BIN)/bench_sta: src/bench_sta.cpp src/sta.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $< $(TBB_FLAGS)

$(BIN)/bench_sta_gpu: src/bench_sta_gpu.cu src/sta.hpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN)/bench_threads: src/bench_threads.cpp src/workload.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/bench_gpu: src/bench_gpu.cu src/workload.hpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Confirms std::execution::par is really parallel and not the silent fallback.
tbb-check: $(BIN)/bench_sta
	./$(BIN)/bench_sta --nodes 1e6 --trials 1 --reps 1 --check-parallel 2>&1 | head -3

clean:
	rm -rf $(BIN)
