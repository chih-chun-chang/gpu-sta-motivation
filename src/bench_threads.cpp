// Throughput vs. number of threads.
//
// Reference benchmark, not part of the main figure set.
//
// Measures throughput against thread count for two work items:
//
//   --mode=cpu   compute-bound (Mandelbrot rows). Kept for the roofline
//                contrast: the same GPU gives ~72x here versus ~8.8x on the
//                bandwidth-bound STA kernel, and the difference is arithmetic
//                intensity.
//   --mode=io    a blocking sleep. Throughput climbs far past the core count
//                because a blocking thread is not using a core at all -- a
//                useful reminder that "does adding threads help?" depends
//                entirely on which resource you ran out of.
//
// The STA benchmarks (bench_sta, bench_sta_gpu) are what the talk uses.
//
// Method: spawn N threads that each loop over work items forever. After a
// warmup, sample a global completed-item counter over a fixed window. Thread
// creation and teardown sit outside the window on purpose -- we want the
// steady-state throughput of N concurrent workers, not the cost of spawning
// them (bench_pool measures that separately).
//
// Output: CSV on stdout.

#include "workload.hpp"

#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

enum class Mode { Io, Cpu };

namespace {

struct Config {
    Mode mode = Mode::Cpu;
    int sleep_us = 10000;   // io mode only; 10 ms matches the latency implied by the talk
    int warmup_ms = 150;
    int measure_ms = 400;
    int trials = 3;
    int max_threads = 4096;
};

double run_trial(const Config& cfg, int nthreads, uint64_t* checksum_out) {
    std::atomic<bool> stop{false};
    std::atomic<long long> done{0};
    std::atomic<uint64_t> checksum{0};
    std::barrier ready(nthreads + 1);

    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (int i = 0; i < nthreads; ++i) {
        threads.emplace_back([&, i] {
            std::vector<uint16_t> row(cfg.mode == Mode::Cpu ? wl::kWidth : 0);
            uint64_t local_sum = 0;
            long long n = 0;
            ready.arrive_and_wait();
            while (!stop.load(std::memory_order_relaxed)) {
                if (cfg.mode == Mode::Io) {
                    std::this_thread::sleep_for(std::chrono::microseconds(cfg.sleep_us));
                } else {
                    // Stride the rows so threads touch different parts of the
                    // image and no thread repeats a row it has in cache.
                    const int py = static_cast<int>((n * 31 + i * 7)) % wl::kHeight;
                    local_sum += wl::mandel_row(py, row.data());
                }
                ++n;
                done.fetch_add(1, std::memory_order_relaxed);
            }
            checksum.fetch_add(local_sum, std::memory_order_relaxed);
        });
    }

    ready.arrive_and_wait();  // every thread exists and is about to start
    std::this_thread::sleep_for(std::chrono::milliseconds(cfg.warmup_ms));

    const long long c0 = done.load(std::memory_order_relaxed);
    const auto t0 = Clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(cfg.measure_ms));
    const long long c1 = done.load(std::memory_order_relaxed);
    const auto t1 = Clock::now();

    stop.store(true, std::memory_order_relaxed);
    for (auto& t : threads) t.join();

    const double secs = std::chrono::duration<double>(t1 - t0).count();
    if (checksum_out) *checksum_out = checksum.load();
    return static_cast<double>(c1 - c0) / secs;
}

std::vector<int> make_sweep(Mode mode, int maxn) {
    std::vector<int> v;
    auto push = [&](int n) {
        if (n >= 1 && n <= maxn && (v.empty() || v.back() != n)) v.push_back(n);
    };
    if (mode == Mode::Cpu) {
        // Dense around the core count (14C / 20T), where the ceiling appears,
        // then out to heavy oversubscription to show it stays flat.
        for (int n = 1; n <= 48; ++n) push(n);
        for (int n = 52; n <= 128; n += 4) push(n);
        for (int n = 144; n <= 512; n += 16) push(n);
    } else {
        // Blocking work scales far past the core count, so sweep much wider.
        for (int n = 1; n <= 32; ++n) push(n);
        for (int n = 36; n <= 128; n += 4) push(n);
        for (int n = 144; n <= 512; n += 16) push(n);
        for (int n = 576; n <= 4096; n += 64) push(n);
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::atoi(argv[++i]); };
        if (a == "--mode") {
            cfg.mode = (std::string(argv[++i]) == "cpu") ? Mode::Cpu : Mode::Io;
        } else if (a == "--sleep-us") {
            cfg.sleep_us = next();
        } else if (a == "--warmup-ms") {
            cfg.warmup_ms = next();
        } else if (a == "--measure-ms") {
            cfg.measure_ms = next();
        } else if (a == "--trials") {
            cfg.trials = next();
        } else if (a == "--max-threads") {
            cfg.max_threads = next();
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }

    const bool is_cpu = (cfg.mode == Mode::Cpu);
    const char* mode_name = is_cpu ? "cpu" : "io";
    // cpu: one item is one Mandelbrot row. io: one item is one blocking request.
    const char* unit = is_cpu ? "rows" : "tasks";
    std::fprintf(stderr, "mode=%s unit=%s/sec hw_concurrency=%u window=%dms trials=%d\n",
                 mode_name, unit, std::thread::hardware_concurrency(),
                 cfg.measure_ms, cfg.trials);

    std::printf("mode,unit,threads,trial,throughput\n");
    const std::vector<int> sweep = make_sweep(cfg.mode, cfg.max_threads);
    for (size_t si = 0; si < sweep.size(); ++si) {
        const int n = sweep[si];
        for (int t = 0; t < cfg.trials; ++t) {
            const double tput = run_trial(cfg, n, nullptr);
            std::printf("%s,%s,%d,%d,%.1f\n", mode_name, unit, n, t, tput);
        }
        std::fflush(stdout);
        std::fprintf(stderr, "\r  %zu/%zu  threads=%-6d", si + 1, sweep.size(), n);
    }
    std::fprintf(stderr, "\ndone\n");
    return 0;
}
