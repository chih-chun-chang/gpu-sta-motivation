// The STA propagation kernel, compiled by both g++ (host) and nvcc (device).
//
// Block-based static timing analysis propagates arrival times forward through
// the timing graph. For a node with K fanin edges:
//
//     arrival[i] = max over k<K of ( arrival_in[i][k] + delay[i][k] )
//
// That is the whole recurrence: an elementwise add of an arrival vector and a
// delay vector, followed by a max-reduction over each node's fanin window.
//
// LAYOUT. Fanin slot k of every node is stored contiguously (`a[k*N + i]`),
// not each node's K edges together. Same arithmetic either way, but this
// layout lets the CPU auto-vectorise across nodes AND lets the GPU coalesce
// its loads, so neither side is handicapped by the data format. `--layout aos`
// switches to the node-major form to show what the choice costs. Real timing
// graphs have irregular fanin; bucketing nodes by fanin degree into batches
// like this is what GPU STA implementations actually do.
//
// ARITHMETIC INTENSITY. Per node: K*8 bytes read + 4 bytes written, against
// K adds and K-1 maxes. About 0.25 flops/byte -- this kernel is bound by
// memory bandwidth, not by arithmetic. That is the honest STA regime, and it
// is why the GPU advantage here tracks the bandwidth ratio rather than the
// FLOP ratio.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define STA_HD __host__ __device__
#else
#define STA_HD
#endif

namespace sta {

// Fanin per node: the "window" the max is taken over.
constexpr int kFanin = 8;

enum class Layout { Soa, Aos };

// Index of fanin slot k of node i within the edge arrays.
STA_HD inline size_t edge_index(Layout layout, size_t n_nodes, size_t i, int k) {
    return (layout == Layout::Soa) ? (static_cast<size_t>(k) * n_nodes + i)
                                   : (i * kFanin + static_cast<size_t>(k));
}

// One node's propagation. `arrival_in` and `delay` are the two large vectors.
STA_HD inline float propagate(const float* __restrict arrival_in,
                              const float* __restrict delay, Layout layout,
                              size_t n_nodes, size_t i) {
    size_t e = edge_index(layout, n_nodes, i, 0);
    float best = arrival_in[e] + delay[e];
    for (int k = 1; k < kFanin; ++k) {
        e = edge_index(layout, n_nodes, i, k);
        const float cand = arrival_in[e] + delay[e];
        best = cand > best ? cand : best;  // plain compare: same result host & device
    }
    return best;
}

// Deterministic input generation, so CPU and GPU see byte-identical vectors.
// splitmix64 -- cheap, no <random> dependency, identical on both sides.
STA_HD inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Arrival times in [0,100) ns, delays in [0,1) ns -- plausible magnitudes, and
// well inside float's exact-comparison range so the max is unambiguous.
STA_HD inline float gen_arrival(uint64_t e) {
    return static_cast<float>(splitmix64(e) >> 40) * (100.0f / 16777216.0f);
}
STA_HD inline float gen_delay(uint64_t e) {
    return static_cast<float>(splitmix64(e ^ 0xD1B54A32D192ED03ull) >> 40) *
           (1.0f / 16777216.0f);
}

// Order-independent checksum: sums the raw bit patterns as integers, so a
// parallel reduction in any order gives the same answer. Each output element
// is computed independently and identically on both sides, so this compares
// them exactly.
inline uint64_t checksum(const float* out, size_t n) {
    uint64_t s = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        __builtin_memcpy(&bits, &out[i], sizeof(bits));
        s += bits;
    }
    return s;
}

// Bytes of memory traffic one full propagation pass must move, at minimum.
inline size_t bytes_per_pass(size_t n_nodes) {
    return n_nodes * (static_cast<size_t>(kFanin) * 2 * sizeof(float) + sizeof(float));
}

}  // namespace sta
