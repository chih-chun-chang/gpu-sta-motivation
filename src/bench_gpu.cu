// Same CPU-bound work item as bench_threads --mode=cpu, run on the GPU.
//
// This is the punchline for the "why do we need a GPU?" slide. The C++
// concurrency story ends at the core count: once every core is RUNNING, more
// threads buy you concurrency but not throughput. The only way past that line
// is more parallel hardware.
//
// Reports three numbers, all in Mandelbrot rows/sec so they sit on the same
// axis as the CPU sweep:
//
//   cpu_1core            one thread, for scale
//   gpu_compute          kernel time only (cudaEvent), the honest "compute" rate
//   gpu_with_transfer    kernel + device-to-host copy of every result
//
// The last one matters: if you only ever move data to the GPU and back without
// doing enough arithmetic on it, PCIe -- not the SMs -- is your throughput.
//
// Output: CSV on stdout.

#include "workload.hpp"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        const cudaError_t err__ = (expr);                                      \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,\
                         cudaGetErrorString(err__));                           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

using Clock = std::chrono::steady_clock;

__global__ void mandel_kernel(uint16_t* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= wl::kWidth * wl::kHeight) return;
    const int py = idx / wl::kWidth;
    const int px = idx % wl::kWidth;
    out[idx] = static_cast<uint16_t>(wl::mandel_pixel(px, py));
}

namespace {

uint64_t checksum(const std::vector<uint16_t>& img) {
    uint64_t s = 0;
    for (uint16_t v : img) s += v;
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    int iters = 50;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--iters") iters = std::atoi(argv[++i]);
    }

    const size_t pixels = static_cast<size_t>(wl::kWidth) * wl::kHeight;
    const size_t bytes = pixels * sizeof(uint16_t);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // cudaDeviceProp::clockRate was removed in CUDA 13; the device attribute
    // works across 11, 12 and 13. Display only, so 0 is tolerable.
    int clock_khz = 0;
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0) != cudaSuccess) {
        cudaGetLastError();
    }
    std::fprintf(stderr, "gpu=%s sms=%d clock=%.2fGHz iters=%d\n", prop.name,
                 prop.multiProcessorCount, clock_khz / 1e6, iters);

    // ---- CPU reference: one thread, one full image ----------------------
    std::vector<uint16_t> cpu_img(pixels);
    const auto cpu_t0 = Clock::now();
    uint64_t cpu_sum = 0;
    for (int py = 0; py < wl::kHeight; ++py) {
        cpu_sum += wl::mandel_row(py, cpu_img.data() + static_cast<size_t>(py) * wl::kWidth);
    }
    const double cpu_secs = std::chrono::duration<double>(Clock::now() - cpu_t0).count();
    const double cpu_rows_per_s = wl::kHeight / cpu_secs;

    // ---- GPU ------------------------------------------------------------
    uint16_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    std::vector<uint16_t> gpu_img(pixels);

    const int block = 256;
    const int grid = static_cast<int>((pixels + block - 1) / block);

    // Warmup, and verify the GPU reproduces the CPU result bit-for-bit.
    mandel_kernel<<<grid, block>>>(d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gpu_img.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    const uint64_t gpu_sum = checksum(gpu_img);
    const bool match = (gpu_sum == cpu_sum);
    std::fprintf(stderr, "checksum cpu=%llu gpu=%llu %s\n",
                 static_cast<unsigned long long>(cpu_sum),
                 static_cast<unsigned long long>(gpu_sum),
                 match ? "MATCH" : "MISMATCH");

    // Compute only.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) mandel_kernel<<<grid, block>>>(d_out);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    const double gpu_rows_per_s = (static_cast<double>(iters) * wl::kHeight) / (ms / 1e3);

    // Compute plus a device-to-host copy of every result.
    const auto xfer_t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        mandel_kernel<<<grid, block>>>(d_out);
        CUDA_CHECK(cudaMemcpy(gpu_img.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double xfer_secs = std::chrono::duration<double>(Clock::now() - xfer_t0).count();
    const double gpu_xfer_rows_per_s = (static_cast<double>(iters) * wl::kHeight) / xfer_secs;

    std::printf("impl,throughput,checksum_ok\n");
    std::printf("cpu_1core,%.1f,1\n", cpu_rows_per_s);
    std::printf("gpu_compute,%.1f,%d\n", gpu_rows_per_s, match ? 1 : 0);
    std::printf("gpu_with_transfer,%.1f,%d\n", gpu_xfer_rows_per_s, match ? 1 : 0);

    CUDA_CHECK(cudaFree(d_out));
    return match ? 0 : 1;
}
