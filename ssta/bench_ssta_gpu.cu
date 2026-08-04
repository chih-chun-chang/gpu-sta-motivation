// GPU side of the statistical (POCV) propagation benchmark.
//
// Same kernel as the CPU (both include ssta.hpp), inputs generated on-device by
// the same deterministic functions, checksum compared against a host reference.
//
// Output: CSV written to --out.

#include "ssta.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        const cudaError_t err__ = (expr);                                       \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

struct Dev {
    float *prm, *prs, *pfm, *pfs, *crm, *crs, *cfm, *cfs;
    uint8_t* sense;
    float *orm, *ors, *ofm, *ofs;
};

__global__ void gen_kernel(Dev d, size_t edges) {
    const size_t e = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= edges) return;
    d.prm[e] = ssta::gen_mean(e, 0x11); d.prs[e] = ssta::gen_std(e, 0x12);
    d.pfm[e] = ssta::gen_mean(e, 0x13); d.pfs[e] = ssta::gen_std(e, 0x14);
    d.crm[e] = ssta::gen_mean(e, 0x21); d.crs[e] = ssta::gen_std(e, 0x22);
    d.cfm[e] = ssta::gen_mean(e, 0x23); d.cfs[e] = ssta::gen_std(e, 0x24);
    d.sense[e] = ssta::gen_sense(e);
}

__global__ void propagate_kernel(Dev d, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const ssta::Result r = ssta::propagate(d.prm, d.prs, d.pfm, d.pfs, d.crm, d.crs,
                                           d.cfm, d.cfs, d.sense, n, i);
    d.orm[i] = r.rise_mean; d.ors[i] = r.rise_std;
    d.ofm[i] = r.fall_mean; d.ofs[i] = r.fall_std;
}

int main(int argc, char** argv) {
    size_t nodes = 8u << 20;
    int reps = 20;
    std::string out_path = "data/ssta_gpu.csv";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--nodes") nodes = static_cast<size_t>(std::atof(argv[++i]));
        else if (a == "--reps") reps = std::atoi(argv[++i]);
        else if (a == "--out") out_path = argv[++i];
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }

    const size_t n = nodes, edges = n * ssta::kFanin;
    const size_t eb = edges * sizeof(float), ob = n * sizeof(float);
    const size_t bytes = ssta::bytes_per_pass(n);
    const size_t flops = ssta::flops_per_pass(n);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "gpu=%s sms=%d nodes=%zu working_set=%.0f MB intensity=%.3f\n",
                 prop.name, prop.multiProcessorCount, n, bytes / 1048576.0,
                 static_cast<double>(flops) / bytes);

    Dev d{};
    for (float** p : {&d.prm, &d.prs, &d.pfm, &d.pfs, &d.crm, &d.crs, &d.cfm, &d.cfs})
        CUDA_CHECK(cudaMalloc(p, eb));
    CUDA_CHECK(cudaMalloc(&d.sense, edges));
    for (float** p : {&d.orm, &d.ors, &d.ofm, &d.ofs}) CUDA_CHECK(cudaMalloc(p, ob));

    const int block = 256;
    gen_kernel<<<static_cast<int>((edges + block - 1) / block), block>>>(d, edges);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int grid = static_cast<int>((n + block - 1) / block);
    for (int w = 0; w < 3; ++w) propagate_kernel<<<grid, block>>>(d, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int r = 0; r < reps; ++r) propagate_kernel<<<grid, block>>>(d, n);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    ms /= reps;

    // Host reference on a slice, to confirm both sides agree.
    const size_t check_n = n < (1u << 16) ? n : (1u << 16);
    std::vector<float> h(edges * 8);
    std::vector<uint8_t> hs(edges);
    float* hp[8];
    for (int k = 0; k < 8; ++k) hp[k] = h.data() + k * edges;
    const uint64_t salts[8] = {0x11, 0x12, 0x13, 0x14, 0x21, 0x22, 0x23, 0x24};
    for (size_t e = 0; e < edges; ++e) {
        for (int k = 0; k < 8; ++k)
            hp[k][e] = (k % 2 == 0) ? ssta::gen_mean(e, salts[k]) : ssta::gen_std(e, salts[k]);
        hs[e] = ssta::gen_sense(e);
    }
    std::vector<float> grm(n), grs(n), gfm(n), gfs(n);
    CUDA_CHECK(cudaMemcpy(grm.data(), d.orm, ob, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grs.data(), d.ors, ob, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gfm.data(), d.ofm, ob, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gfs.data(), d.ofs, ob, cudaMemcpyDeviceToHost));

    bool match = true;
    for (size_t i = 0; i < check_n && match; ++i) {
        const ssta::Result r = ssta::propagate(hp[0], hp[1], hp[2], hp[3], hp[4], hp[5],
                                               hp[6], hp[7], hs.data(), n, i);
        match = (r.rise_mean == grm[i] && r.rise_std == grs[i] &&
                 r.fall_mean == gfm[i] && r.fall_std == gfs[i]);
    }
    std::fprintf(stderr, "host/device agreement over %zu nodes: %s\n", check_n,
                 match ? "MATCH" : "MISMATCH");

    const double gbs = static_cast<double>(bytes) / (ms / 1e3) / 1e9;
    const double gflops = static_cast<double>(flops) / (ms / 1e3) / 1e9;
    std::fprintf(stderr, "kernel %.3f ms -> %.1f GB/s, %.1f GFLOP/s\n", ms, gbs, gflops);

    if (FILE* f = std::fopen(out_path.c_str(), "w")) {
        std::fprintf(f, "impl,intensity,gb_per_sec,gflop_per_sec,ms,checksum_ok\n");
        std::fprintf(f, "gpu_kernel,%.5f,%.3f,%.3f,%.4f,%d\n",
                     static_cast<double>(flops) / bytes, gbs, gflops, ms, match);
        std::fclose(f);
        std::fprintf(stderr, "wrote %s\n", out_path.c_str());
    }
    return match ? 0 : 1;
}
