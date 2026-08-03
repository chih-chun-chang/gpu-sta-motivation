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

// Runtime against problem size: kernel alone, and kernel plus the copies.
// Writes its own CSV and returns.
static int run_size_sweep(int lo, int hi, int trials, sta::Layout layout,
                          const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        return 1;
    }
    std::fprintf(f, "strategy,nodes,edges,trial,ms,gb_per_sec\n");

    for (int lg = lo; lg <= hi; ++lg) {
        const size_t n = size_t(1) << lg;
        const size_t edges = n * sta::kFanin;
        const size_t edge_bytes = edges * sizeof(float);
        const size_t out_bytes = n * sizeof(float);
        const size_t pass_bytes = sta::bytes_per_pass(n);
        const int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);

        float *d_a, *d_d, *d_out, *p_a, *p_d, *p_out;
        CUDA_CHECK(cudaMalloc(&d_a, edge_bytes));
        CUDA_CHECK(cudaMalloc(&d_d, edge_bytes));
        CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
        CUDA_CHECK(cudaHostAlloc(&p_a, edge_bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&p_d, edge_bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&p_out, out_bytes, cudaHostAllocDefault));
        gen_kernel<<<static_cast<int>((edges + block - 1) / block), block>>>(d_a, d_d, edges);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto emit = [&](const char* s, int t, double ms) {
            std::fprintf(f, "%s,%zu,%zu,%d,%.5f,%.3f\n", s, n, edges, t, ms,
                         pass_bytes / (ms / 1e3) / 1e9);
        };

        // Kernel only, data already resident.
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        for (int w = 0; w < 3; ++w) propagate_kernel<<<grid, block>>>(d_a, d_d, d_out, n, layout);
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int t = 0; t < trials; ++t) {
            const int inner = 10;
            CUDA_CHECK(cudaEventRecord(e0));
            for (int r = 0; r < inner; ++r)
                propagate_kernel<<<grid, block>>>(d_a, d_d, d_out, n, layout);
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            emit("gpu_kernel", t, ms / inner);
        }

        // Kernel plus copy in and copy out.
        for (int t = 0; t < trials + 1; ++t) {
            const auto t0 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaMemcpy(d_a, p_a, edge_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_d, p_d, edge_bytes, cudaMemcpyHostToDevice));
            propagate_kernel<<<grid, block>>>(d_a, d_d, d_out, n, layout);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(p_out, d_out, out_bytes, cudaMemcpyDeviceToHost));
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            if (t > 0) emit("gpu_with_transfer", t - 1, ms);  // discard the first
        }

        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaFreeHost(p_a));
        CUDA_CHECK(cudaFreeHost(p_d));
        CUDA_CHECK(cudaFreeHost(p_out));
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_d));
        CUDA_CHECK(cudaFree(d_out));
        std::fprintf(stderr, "\r  size 2^%d = %zu nodes   ", lg, n);
    }
    std::fclose(f);
    std::fprintf(stderr, "\nwrote %s\n", path.c_str());
    return 0;
}

int main(int argc, char** argv) {
    size_t nodes = 8u << 20;
    int reps = 20;
    sta::Layout layout = sta::Layout::Soa;
    std::string out_path = "data/gpu_sta.csv";
    std::string breakdown_path = "data/gpu_breakdown.csv";
    bool size_sweep_mode = false;
    int size_lo = 14, size_hi = 24, size_trials = 3;
    std::string size_path = "data/size_gpu.csv";

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--nodes") nodes = static_cast<size_t>(std::atof(argv[++i]));
        else if (a == "--reps") reps = std::atoi(argv[++i]);
        else if (a == "--layout") layout = (std::string(argv[++i]) == "aos") ? sta::Layout::Aos : sta::Layout::Soa;
        else if (a == "--out") out_path = argv[++i];
        else if (a == "--breakdown-out") breakdown_path = argv[++i];
        else if (a == "--size-sweep") size_sweep_mode = true;
        else if (a == "--size-lo") size_lo = std::atoi(argv[++i]);
        else if (a == "--size-hi") size_hi = std::atoi(argv[++i]);
        else if (a == "--size-out") size_path = argv[++i];
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }

    if (size_sweep_mode) return run_size_sweep(size_lo, size_hi, size_trials, layout, size_path);

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

    // ---- 3. unified memory, no explicit copies at all ----------------------
    // On a PCIe machine this is usually WORSE than staged copies: the driver
    // migrates pages on demand and you pay fault latency. On GH200, where Grace
    // and Hopper share a coherent NVLink-C2C link, this is the interesting
    // path -- the fix for figure 4 is not a faster copy, it is not copying.
    double managed_eps = 0.0, managed_gbs = 0.0;
    {
        float *m_a = nullptr, *m_d = nullptr, *m_out = nullptr;
        CUDA_CHECK(cudaMallocManaged(&m_a, edge_bytes));
        CUDA_CHECK(cudaMallocManaged(&m_d, edge_bytes));
        CUDA_CHECK(cudaMallocManaged(&m_out, out_bytes));
        for (size_t e = 0; e < edges; ++e) {  // touched on the host, as real input would be
            m_a[e] = h_a[e];
            m_d[e] = h_d[e];
        }
        const auto m0 = std::chrono::steady_clock::now();
        const int mreps = 3;
        for (int i = 0; i < mreps; ++i) {
            propagate_kernel<<<grid, block>>>(m_a, m_d, m_out, n, layout);
            CUDA_CHECK(cudaDeviceSynchronize());
            // Read one element back on the host each pass, so the result really
            // has to be visible to the CPU -- otherwise nothing migrates back.
            volatile float sink = m_out[i];
            (void)sink;
        }
        const double msecs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - m0).count();
        managed_eps = static_cast<double>(edges) * mreps / msecs;
        managed_gbs = static_cast<double>(pass_bytes) * mreps / msecs / 1e9;
        std::fprintf(stderr, "managed (no explicit copies): %.2f GB/s\n", managed_gbs);
        CUDA_CHECK(cudaFree(m_a));
        CUDA_CHECK(cudaFree(m_d));
        CUDA_CHECK(cudaFree(m_out));
    }

    if (FILE* f = std::fopen(out_path.c_str(), "w")) {
        std::fprintf(f, "impl,layout,edges_per_sec,gb_per_sec,checksum_ok\n");
        std::fprintf(f, "gpu_kernel,%s,%.0f,%.2f,%d\n", layout_name, kern_eps, kern_gbs, match);
        std::fprintf(f, "gpu_with_transfer,%s,%.0f,%.2f,%d\n", layout_name, xfer_eps, xfer_gbs,
                     match);
        std::fprintf(f, "gpu_managed,%s,%.0f,%.2f,%d\n", layout_name, managed_eps, managed_gbs,
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
