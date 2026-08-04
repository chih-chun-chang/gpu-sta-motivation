// Statistical (POCV) arrival propagation -- the INSTA kernel, for the roofline.
//
// This exists to answer one objection to the block-based STA benchmark: "your
// kernel is an add and a max, of course it's memory bound; real STA does more
// arithmetic than that." So we take the arithmetic from a real implementation
// and measure it.
//
// Modelled on INSTA's topk_arrival_kernel.cu. Per candidate (one fanin edge,
// one parent top-K slot), for each of the two transitions:
//
//     m   = p_mean + arc_mean                 means add
//     sd  = sqrt(p_std^2 + arc_std^2)         stds combine in quadrature
//     arr = m + sigma * sd                    the ranking key
//
// Sense matters: a positive-sense arc takes rise from the parent's rise and
// fall from its fall; a negative-sense arc swaps them. Candidates are then
// ranked by `arr` and the largest kept. INSTA keeps the top K with startpoint
// dedup; we keep the top 1, which is the arithmetically favourable case -- if
// even this is memory bound, K > 1 (which re-reads the candidate array K times)
// is more so.
//
// Compared with the block-based kernel in ../src/sta.hpp this does 8x the
// arithmetic per edge (16 flops vs 2) but also moves 4x the data, because every
// value is now a (mean, std) pair and there are two transitions. The point of
// the exercise is that the *ratio* barely moves: 0.221 -> 0.450 flops/byte,
// against a machine balance of 9.6 to 43.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define SSTA_HD __host__ __device__
#else
#define SSTA_HD
#endif

namespace ssta {

#ifndef SSTA_FANIN
#define SSTA_FANIN 8
#endif
constexpr int kFanin = SSTA_FANIN;

// Sigma multiplier for the ranking key, as INSTA passes in.
constexpr float kSigma = 3.0f;

// One node's statistical propagation. All arrays are laid out fanin-slot-major
// (`x[k * n_nodes + i]`) so that both the CPU can vectorise across nodes and the
// GPU can coalesce -- the same choice, and for the same reason, as sta.hpp.
struct Result {
    float rise_mean, rise_std, fall_mean, fall_std;
};

SSTA_HD inline Result propagate(const float* __restrict p_rise_mean,
                                const float* __restrict p_rise_std,
                                const float* __restrict p_fall_mean,
                                const float* __restrict p_fall_std,
                                const float* __restrict c_rise_mean,
                                const float* __restrict c_rise_std,
                                const float* __restrict c_fall_mean,
                                const float* __restrict c_fall_std,
                                const uint8_t* __restrict sense,
                                size_t n_nodes, size_t i) {
    float best_rise_arr = -3.0e38f, best_fall_arr = -3.0e38f;
    Result r{0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = 0; k < kFanin; ++k) {
        const size_t e = static_cast<size_t>(k) * n_nodes + i;
        const bool neg = sense[e] != 0;

        // Negative-sense arcs swap which parent transition feeds which output.
        const float prm = neg ? p_fall_mean[e] : p_rise_mean[e];
        const float prs = neg ? p_fall_std[e] : p_rise_std[e];
        const float pfm = neg ? p_rise_mean[e] : p_fall_mean[e];
        const float pfs = neg ? p_rise_std[e] : p_fall_std[e];

        // rise candidate
        const float rm = prm + c_rise_mean[e];
        const float rs = sqrtf(prs * prs + c_rise_std[e] * c_rise_std[e]);
        const float ra = rm + kSigma * rs;
        if (ra > best_rise_arr) {
            best_rise_arr = ra;
            r.rise_mean = rm;
            r.rise_std = rs;
        }

        // fall candidate
        const float fm = pfm + c_fall_mean[e];
        const float fs = sqrtf(pfs * pfs + c_fall_std[e] * c_fall_std[e]);
        const float fa = fm + kSigma * fs;
        if (fa > best_fall_arr) {
            best_fall_arr = fa;
            r.fall_mean = fm;
            r.fall_std = fs;
        }
    }
    return r;
}

// Deterministic inputs, identical on host and device (as in sta.hpp).
SSTA_HD inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
SSTA_HD inline float gen_unit(uint64_t e, uint64_t salt) {
    return static_cast<float>(splitmix64(e ^ salt) >> 40) * (1.0f / 16777216.0f);
}
SSTA_HD inline float gen_mean(uint64_t e, uint64_t salt) { return gen_unit(e, salt) * 100.0f; }
SSTA_HD inline float gen_std(uint64_t e, uint64_t salt) { return gen_unit(e, salt) * 5.0f + 0.5f; }
SSTA_HD inline uint8_t gen_sense(uint64_t e) {
    return static_cast<uint8_t>(splitmix64(e ^ 0xABCD1234ull) & 1ull);
}

// ---------------------------------------------------------------------------
// Traffic and arithmetic, per node. These are what the roofline is built from.

// Per fanin edge we read 8 floats (4 parent stats, 4 arc stats) plus a sense
// byte; per node we write 4 floats.
inline size_t bytes_per_pass(size_t n_nodes) {
    return n_nodes * (static_cast<size_t>(kFanin) * (8 * sizeof(float) + 1) + 4 * sizeof(float));
}

// Counting convention: adds, multiplies, comparisons and sqrt each count as one
// operation. The same convention is used for the block-based kernel in
// ../src/sta.hpp, where the max counts as an operation too -- mixing the two
// would flatter one kernel over the other.
//
// Per edge, per transition: 1 add (mean) + 2 mul + 1 add + 1 sqrt (std) +
// 1 mul + 1 add (rank) = 7, and reducing K candidates costs K-1 comparisons.
// Two transitions: 2 * (7K + K - 1) = 16K - 2.
inline size_t flops_per_pass(size_t n_nodes) {
    return n_nodes * (16 * static_cast<size_t>(kFanin) - 2);
}

inline uint64_t checksum(const float* a, const float* b, const float* c, const float* d,
                         size_t n) {
    uint64_t s = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t x;
        __builtin_memcpy(&x, &a[i], 4); s += x;
        __builtin_memcpy(&x, &b[i], 4); s += x;
        __builtin_memcpy(&x, &c[i], 4); s += x;
        __builtin_memcpy(&x, &d[i], 4); s += x;
    }
    return s;
}

}  // namespace ssta
