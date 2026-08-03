// GPU side of the STA propagation benchmark.
//
// Same kernel as the CPU (both include sta.hpp), same inputs, checksum compared
// exactly. Reports three things, and the third is the point of the talk:
//
//   gpu_kernel          data already resident on the device. The bandwidth win.
//   gpu_with_transfer   H2D the two vectors, propagate, D2H the result, every
//                       pass -- what a naive "offload this function" port does.
//   breakdown           milliseconds split across H2D / kernel / D2H, so the
//                       audience can see PCIe dwarf the compute.
//
// Host staging buffers are PINNED, which is the best case for PCIe. If transfer
// still dominates with pinned memory, it dominates for everyone.
//
// Output: two CSVs (see --out and --breakdown-out).

#include "sta.hpp"

#include <chrono>
#include <cstdio>
#include <cstdint>
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

__global__ void propagate_kernel(const float* __restrict a, const float* __restrict d,
                                 float* __restrict out, size_t n, sta::Layout layout) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = sta::propagate(a, d, layout, n, i);
}

__global__ void gen_kernel(float* a, float* d, size_t edges) {
    const size_t e = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= edges) return;
    a[e] = sta::gen_arrival(e);
    d[e] = sta::gen_delay(e);
}

int main(int argc, char** argv) {
    size_t nodes = 8u << 20;
    int reps = 20;
    sta::Layout layout = sta::Layout::Soa;
    std::string out_path = "data/gpu_sta.csv";
    std::string breakdown_path = "data/gpu_breakdown.csv";

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--nodes") nodes = static_cast<size_t>(std::atof(argv[++i]));
        else if (a == "--reps") reps = std::atoi(argv[++i]);
        else if (a == "--layout") layout = (std::string(argv[++i]) == "aos") ? sta::Layout::Aos : sta::Layout::Soa;
        else if (a == "--out") out_path = argv[++i];
        else if (a == "--breakdown-out") breakdown_path = argv[++i];
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }

    const size_t n = nodes;
    const size_t edges = n * sta::kFanin;
    const size_t edge_bytes = edges * sizeof(float);
    const size_t out_bytes = n * sizeof(float);
    const size_t pass_bytes = sta::bytes_per_pass(n);
    const char* layout_name = (layout == sta::Layout::Soa) ? "soa" : "aos";

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const double peak_bw =
        2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;
    std::fprintf(stderr, "gpu=%s sms=%d bus=%d-bit peak_bw=%.0f GB/s\n", prop.name,
                 prop.multiProcessorCount, prop.memoryBusWidth, peak_bw);
    std::fprintf(stderr, "nodes=%zu edges=%zu working_set=%.0f MB layout=%s reps=%d\n", n,
                 edges, pass_bytes / 1048576.0, layout_name, reps);

    // ---- device buffers, generated on-device so we never wait on PCIe here --
    float *d_a = nullptr, *d_d = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, edge_bytes));
    CUDA_CHECK(cudaMalloc(&d_d, edge_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

    const int block = 256;
    gen_kernel<<<static_cast<int>((edges + block - 1) / block), block>>>(d_a, d_d, edges);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int grid = static_cast<int>((n + block - 1) / block);

    // ---- correctness: compare against a host computation of the same kernel -
    std::vector<float> h_a(edges), h_d(edges), h_ref(n), h_gpu(n);
    for (size_t e = 0; e < edges; ++e) {
        h_a[e] = sta::gen_arrival(e);
        h_d[e] = sta::gen_delay(e);
    }
    for (size_t i = 0; i < n; ++i)
        h_ref[i] = sta::propagate(h_a.data(), h_d.data(), layout, n, i);

    propagate_kernel<<<grid, block>>>(d_a, d_d, d_out, n, layout);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    const uint64_t ref_sum = sta::checksum(h_ref.data(), n);
    const uint64_t gpu_sum = sta::checksum(h_gpu.data(), n);
    const bool match = (ref_sum == gpu_sum);
    std::fprintf(stderr, "checksum cpu=%llu gpu=%llu %s\n",
                 static_cast<unsigned long long>(ref_sum),
                 static_cast<unsigned long long>(gpu_sum), match ? "MATCH" : "MISMATCH");

    // ---- 1. kernel only, data resident -------------------------------------
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    for (int i = 0; i < 3; ++i) propagate_kernel<<<grid, block>>>(d_a, d_d, d_out, n, layout);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < reps; ++i) propagate_kernel<<<grid, block>>>(d_a, d_d, d_out, n, layout);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, e0, e1));
    kernel_ms /= reps;
    const double kern_eps = edges / (kernel_ms / 1e3);
    const double kern_gbs = pass_bytes / (kernel_ms / 1e3) / 1e9;

    // ---- 2. with transfer, pinned staging (PCIe best case) -----------------
    float *p_a = nullptr, *p_d = nullptr, *p_out = nullptr;
    CUDA_CHECK(cudaHostAlloc(&p_a, edge_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&p_d, edge_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&p_out, out_bytes, cudaHostAllocDefault));
    std::copy(h_a.begin(), h_a.end(), p_a);
    std::copy(h_d.begin(), h_d.end(), p_d);

    double h2d_ms = 0, konly_ms = 0, d2h_ms = 0;
    const int xreps = 5;
    for (int i = 0; i < xreps + 1; ++i) {
        const bool warm = (i == 0);  // discard the first
        auto t0 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpy(d_a, p_a, edge_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_d, p_d, edge_bytes, cudaMemcpyHostToDevice));
        auto t1 = std::chrono::steady_clock::now();
        propagate_kernel<<<grid, block>>>(d_a, d_d, d_out, n, layout);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpy(p_out, d_out, out_bytes, cudaMemcpyDeviceToHost));
        auto t3 = std::chrono::steady_clock::now();
        if (warm) continue;
        h2d_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        konly_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        d2h_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
    }
    h2d_ms /= xreps;
    konly_ms /= xreps;
    d2h_ms /= xreps;
    const double total_ms = h2d_ms + konly_ms + d2h_ms;
    const double xfer_eps = edges / (total_ms / 1e3);
    const double xfer_gbs = pass_bytes / (total_ms / 1e3) / 1e9;

    std::fprintf(stderr, "h2d=%.2fms kernel=%.2fms d2h=%.2fms  (PCIe %.1f GB/s)\n", h2d_ms,
                 konly_ms, d2h_ms, (2.0 * edge_bytes) / (h2d_ms / 1e3) / 1e9);

    if (FILE* f = std::fopen(out_path.c_str(), "w")) {
        std::fprintf(f, "impl,layout,edges_per_sec,gb_per_sec,checksum_ok\n");
        std::fprintf(f, "gpu_kernel,%s,%.0f,%.2f,%d\n", layout_name, kern_eps, kern_gbs, match);
        std::fprintf(f, "gpu_with_transfer,%s,%.0f,%.2f,%d\n", layout_name, xfer_eps, xfer_gbs,
                     match);
        std::fclose(f);
        std::fprintf(stderr, "wrote %s\n", out_path.c_str());
    }
    if (FILE* f = std::fopen(breakdown_path.c_str(), "w")) {
        std::fprintf(f, "phase,ms,share\n");
        std::fprintf(f, "h2d,%.3f,%.4f\n", h2d_ms, h2d_ms / total_ms);
        std::fprintf(f, "kernel,%.3f,%.4f\n", konly_ms, konly_ms / total_ms);
        std::fprintf(f, "d2h,%.3f,%.4f\n", d2h_ms, d2h_ms / total_ms);
        std::fclose(f);
        std::fprintf(stderr, "wrote %s\n", breakdown_path.c_str());
    }

    CUDA_CHECK(cudaFreeHost(p_a));
    CUDA_CHECK(cudaFreeHost(p_d));
    CUDA_CHECK(cudaFreeHost(p_out));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaFree(d_out));
    return match ? 0 : 1;
}
