CXX       ?= g++
NVCC      ?= nvcc
# `native` detects the installed GPU (sm_86 on RTX A4000, sm_90 on H100), so
# this builds unchanged when moving machines. Override for cross-compiling.
CUDA_ARCH ?= native

# TBB backs libstdc++'s parallel algorithms. Without it std::execution::par
# silently runs SEQUENTIALLY, which would make every CPU number in this repo
# wrong. Install it one of these ways:
#
#   conda install -c conda-forge tbb-devel      then TBB_ROOT=$(CONDA_PREFIX)
#   sudo apt install libtbb-dev                 (lands on the default paths)
#   sudo dnf install tbb-devel                  (lands on the default paths)
#   module load tbb  /  oneAPI                  then TBB_ROOT=/opt/intel/oneapi/tbb/latest
#
# TBB_ROOT is a PREFIX, not an include or a lib directory: the build looks for
# $(TBB_ROOT)/include and then for lib64 / lib/<triplet> / lib beneath it,
# because distributions disagree about which one to use. Override TBB_INC and
# TBB_LIB directly if your install has a different shape. `make tbb-info`
# prints what got resolved.
comma := ,
TBB_ROOT ?= $(HOME)/miniconda3
TBB_INC  ?= $(TBB_ROOT)/include
TBB_LIB  ?= $(firstword $(wildcard $(TBB_ROOT)/lib64 \
                                   $(TBB_ROOT)/lib/x86_64-linux-gnu \
                                   $(TBB_ROOT)/lib/aarch64-linux-gnu \
                                   $(TBB_ROOT)/lib))

TBB_HDR := $(wildcard $(TBB_INC)/oneapi/tbb/global_control.h)
ifeq ($(TBB_HDR),)
  # Not under TBB_ROOT. Assume a system install already on the default search
  # paths; if it isn't there either, the compile fails loudly, which is what
  # we want -- see the `tbb-info` target.
  TBB_FLAGS := -ltbb
else
  TBB_FLAGS := -I$(TBB_INC) \
               $(if $(TBB_LIB),-L$(TBB_LIB) -Wl$(comma)-rpath$(comma)$(TBB_LIB)) -ltbb
endif

# -ffp-contract=off / --fmad=false keep host and device float rounding identical,
# so the CPU and GPU checksums can be compared exactly.
CXXFLAGS  := -O3 -march=native -std=c++20 -ffp-contract=off -pthread -Isrc
NVCCFLAGS := -O3 -std=c++17 --fmad=false -arch=$(CUDA_ARCH) -Isrc

BIN := bin

# The STA benchmarks are the talk. bench_threads/bench_gpu are the
# compute-bound Mandelbrot reference kept for the roofline contrast.
STA_TARGETS  := $(BIN)/bench_sta $(BIN)/bench_sta_gpu
# Statistical (POCV) propagation -- the INSTA arithmetic, for the roofline.
SSTA_TARGETS := $(BIN)/bench_ssta $(BIN)/bench_ssta_gpu
REF_TARGETS  := $(BIN)/bench_threads $(BIN)/bench_gpu

.PHONY: all sta ssta ref clean tbb-check tbb-info
all: sta ssta ref
sta: $(STA_TARGETS)
ssta: $(SSTA_TARGETS)
ref: $(REF_TARGETS)

$(BIN):
	mkdir -p $(BIN)

$(BIN)/bench_sta: src/bench_sta.cpp src/sta.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $< $(TBB_FLAGS)

$(BIN)/bench_sta_gpu: src/bench_sta_gpu.cu src/sta.hpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN)/bench_ssta: ssta/bench_ssta.cpp ssta/ssta.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Issta -o $@ $< $(TBB_FLAGS)

$(BIN)/bench_ssta_gpu: ssta/bench_ssta_gpu.cu ssta/ssta.hpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) -Issta -o $@ $<

$(BIN)/bench_threads: src/bench_threads.cpp src/workload.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/bench_gpu: src/bench_gpu.cu src/workload.hpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Shows which TBB the build resolved. Run this first when moving machines.
tbb-info:
	@echo "TBB_ROOT  = $(TBB_ROOT)"
	@echo "TBB_INC   = $(TBB_INC)"
	@echo "  header  = $(if $(TBB_HDR),FOUND,not found -- falling back to system paths)"
	@echo "TBB_LIB   = $(if $(TBB_LIB),$(TBB_LIB),<none> -- using default linker paths)"
	@echo "TBB_FLAGS = $(TBB_FLAGS)"
	@echo
	@echo "If the header is not found and the build fails, pass a prefix, e.g."
	@echo "  make TBB_ROOT=\$$CONDA_PREFIX"
	@echo "  make TBB_ROOT=/opt/intel/oneapi/tbb/latest"
	@echo "  make TBB_INC=/usr/include TBB_LIB=/usr/lib64"

# Confirms std::execution::par is really parallel and not the silent fallback.
# bench_sta aborts on its own if it is sequential, so this failing is fatal.
tbb-check: $(BIN)/bench_sta
	./$(BIN)/bench_sta --nodes 1e6 --trials 1 --reps 1 --check-parallel 2>&1 | head -3

clean:
	rm -rf $(BIN)
