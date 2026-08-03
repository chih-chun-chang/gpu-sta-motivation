// Shared workload definitions, compiled by both g++ (host) and nvcc (device).
//
// Two kinds of "work item" are used across the benchmarks:
//
//   IO  -- a blocking sleep. Stands in for a request that waits on the network
//          or a disk. The thread is BLOCKED, not RUNNING, for the whole item.
//   CPU -- one row of a Mandelbrot image. Pure computation, no waiting, and
//          embarrassingly parallel, so it maps onto a GPU unchanged.
//
// The Mandelbrot math is float, not double: the RTX A4000 (GA104) runs FP64 at
// 1/64 rate, so a double kernel would measure the GPU's crippled FP64 path
// rather than the parallelism we are trying to show. Host code is built with
// -ffp-contract=off and device code with --fmad=false so both sides round
// identically and the checksums can be compared bit-for-bit.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define WL_HD __host__ __device__
#else
#define WL_HD
#endif

namespace wl {

constexpr int kWidth = 1024;
constexpr int kHeight = 1024;
constexpr int kMaxIter = 256;

// Escape-time iteration count for a single pixel.
WL_HD inline int mandel_pixel(int px, int py) {
    const float x0 = -2.5f + 3.5f * (static_cast<float>(px) + 0.5f) / kWidth;
    const float y0 = -1.25f + 2.5f * (static_cast<float>(py) + 0.5f) / kHeight;
    float x = 0.0f, y = 0.0f;
    int i = 0;
    while (x * x + y * y <= 4.0f && i < kMaxIter) {
        const float xt = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xt;
        ++i;
    }
    return i;
}

// One work item: one image row. Returns a checksum, which both keeps the
// compiler from eliding the loop and lets us verify GPU == CPU.
WL_HD inline uint64_t mandel_row(int py, uint16_t* out) {
    uint64_t sum = 0;
    for (int px = 0; px < kWidth; ++px) {
        const int v = mandel_pixel(px, py);
        out[px] = static_cast<uint16_t>(v);
        sum += static_cast<uint64_t>(v);
    }
    return sum;
}

}  // namespace wl
