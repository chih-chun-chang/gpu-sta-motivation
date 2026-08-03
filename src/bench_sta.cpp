// CPU side of the STA propagation benchmark.
//
// The threading is std::for_each with std::execution::par_unseq -- the C++17
// parallel algorithm an engineer would actually reach for. libstdc++ implements
// those policies on Intel TBB, whose work-stealing pool is the thread pool.
// We never create a std::thread here.
//
// GOTCHA WORTH A SLIDE: if TBB is not linked in, libstdc++ silently degrades
// par/par_unseq to sequential execution. No warning, no error, no link failure.
// Run with --check-parallel to have the benchmark tell you which one you got.
//
// The parallelism sweep uses tbb::global_control, which caps the pool for the
// whole process -- so "number of threads" and "pool size" are the same knob
// here, unlike a hand-rolled pool where they are separate.
//
// Throughput is reported in timing edges/sec (nodes * kFanin), the natural STA
// unit, and in achieved GB/s so the bandwidth ceiling is visible.
//
// Output: CSV on stdout.

#include "sta.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <oneapi/tbb/global_control.h>

using Clock = std::chrono::steady_clock;

namespace {

struct Config {
    size_t nodes = 8u << 20;  // 8M nodes * 8 fanin = 64M edges, ~544 MB working set
    sta::Layout layout = sta::Layout::Soa;
    int reps = 5;
    int trials = 3;
    bool check_parallel = false;
    bool allow_sequential = false;
};

// One propagation pass over the whole graph.
void propagate_all(const std::vector<float>& arrival_in, const std::vector<float>& delay,
                   std::vector<float>& out, sta::Layout layout) {
    const float* a = arrival_in.data();
    const float* d = delay.data();
    const float* base = out.data();
    const size_t n = out.size();
    std::for_each(std::execution::par_unseq, out.begin(), out.end(),
                  [a, d, base, n, layout](float& o) {
                      const size_t i = static_cast<size_t>(&o - base);
                      o = sta::propagate(a, d, layout, n, i);
                  });
}

// Act 1: one std::thread per work item, spawned as fast as the machine allows.
// This is the strawman, and it is bounded by thread creation, not by the work:
// a clone() costs ~7.5 us against ~50 ns of arithmetic, a 150:1 ratio. Nothing
// about the number of threads changes that, so this is a single number rather
// than a sweep.
double run_thread_per_item(const std::vector<float>& arrival_in,
                           const std::vector<float>& delay, std::vector<float>& out,
                           sta::Layout layout, int measure_ms) {
    const float* a = arrival_in.data();
    const float* d = delay.data();
    float* o = out.data();
    const size_t n = out.size();

    std::atomic<long long> done{0};
    std::atomic<long long> live{0};
    const auto t0 = Clock::now();
    const auto deadline = t0 + std::chrono::milliseconds(measure_ms);

    size_t i = 0;
    while (Clock::now() < deadline) {
        live.fetch_add(1, std::memory_order_relaxed);
        std::thread([&, i] {
            o[i] = sta::propagate(a, d, layout, n, i);
            done.fetch_add(1, std::memory_order_relaxed);
            live.fetch_sub(1, std::memory_order_relaxed);
        }).detach();
        // Stride so the spawned threads touch spread-out memory rather than
        // hammering one cache line.
        i = (i + 4099) % n;
    }
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    while (live.load(std::memory_order_relaxed) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return static_cast<double>(done.load()) / secs;  // nodes/sec
}

// One complete pass with one std::thread per node -- the honest runtime of act 1
// at a given problem size, rather than a throughput extrapolation.
double thread_per_item_pass(const std::vector<float>& arrival_in,
                            const std::vector<float>& delay, std::vector<float>& out,
                            sta::Layout layout) {
    const float* a = arrival_in.data();
    const float* d = delay.data();
    float* o = out.data();
    const size_t n = out.size();
    std::atomic<long long> live{0};

    const auto t0 = Clock::now();
    for (size_t i = 0; i < n; ++i) {
        live.fetch_add(1, std::memory_order_relaxed);
        std::thread([a, d, o, n, layout, i, &live] {
            o[i] = sta::propagate(a, d, layout, n, i);
            live.fetch_sub(1, std::memory_order_relaxed);
        }).detach();
    }
    while (live.load(std::memory_order_relaxed) > 0) std::this_thread::yield();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Runtime against problem size, for each CPU strategy.
void size_sweep(const Config& cfg, int lo, int hi, size_t max_tpi) {
    std::printf("strategy,nodes,edges,trial,ms,gb_per_sec\n");
    const unsigned hw = std::thread::hardware_concurrency();

    for (int lg = lo; lg <= hi; ++lg) {
        const size_t n = size_t(1) << lg;
        const size_t edges = n * sta::kFanin;
        const size_t bytes = sta::bytes_per_pass(n);

        std::vector<float> arrival_in(edges), delay(edges), out(n);
        std::for_each(std::execution::par_unseq, arrival_in.begin(), arrival_in.end(),
                      [base = arrival_in.data()](float& x) {
                          x = sta::gen_arrival(static_cast<uint64_t>(&x - base));
                      });
        std::for_each(std::execution::par_unseq, delay.begin(), delay.end(),
                      [base = delay.data()](float& x) {
                          x = sta::gen_delay(static_cast<uint64_t>(&x - base));
                      });

        auto emit = [&](const char* strat, int trial, double ms) {
            std::printf("%s,%zu,%zu,%d,%.5f,%.3f\n", strat, n, edges, trial, ms,
                        bytes / (ms / 1e3) / 1e9);
        };
        auto timed = [&](const char* strat, size_t parallelism) {
            tbb::global_control gc(tbb::global_control::max_allowed_parallelism, parallelism);
            propagate_all(arrival_in, delay, out, cfg.layout);  // warm / fault in
            for (int t = 0; t < cfg.trials; ++t) {
                const auto t0 = Clock::now();
                propagate_all(arrival_in, delay, out, cfg.layout);
                emit(strat, t, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            }
        };

        timed("single_thread", 1);
        timed("for_each_par", hw);
        if (n <= max_tpi) {
            // One trial only: this strategy is ~5,000x slower and the sweep
            // would otherwise dominate the run time.
            emit("thread_per_item", 0, thread_per_item_pass(arrival_in, delay, out, cfg.layout));
        }
        std::fflush(stdout);
        std::fprintf(stderr, "\r  size 2^%d = %zu nodes   ", lg, n);
    }
    std::fprintf(stderr, "\ndone\n");
}

// Confirms whether the parallel policy is really running on more than one
// thread -- i.e. whether TBB got linked in.
size_t observed_threads(size_t n) {
    std::vector<size_t> probe(n);
    std::set<std::thread::id> ids;
    std::mutex m;
    const size_t* base = probe.data();
    std::for_each(std::execution::par, probe.begin(), probe.end(),
                  [&, base](size_t& x) {
                      double t = 1.0;
                      for (int k = 0; k < 64; ++k) t = t * 1.0000001 + 1e-9;
                      x = static_cast<size_t>(t);
                      if ((&x - base) % 65536 == 0) {
                          std::lock_guard<std::mutex> g(m);
                          ids.insert(std::this_thread::get_id());
                      }
                  });
    return ids.size();
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    bool size_sweep_mode = false;
    int size_lo = 14, size_hi = 24;
    size_t max_tpi = 1u << 20;  // thread-per-item is ~5,000x slower; cap the sweep
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--nodes") {
            cfg.nodes = static_cast<size_t>(std::atof(argv[++i]));
        } else if (a == "--layout") {
            cfg.layout = (std::string(argv[++i]) == "aos") ? sta::Layout::Aos : sta::Layout::Soa;
        } else if (a == "--reps") {
            cfg.reps = std::atoi(argv[++i]);
        } else if (a == "--trials") {
            cfg.trials = std::atoi(argv[++i]);
        } else if (a == "--check-parallel") {
            cfg.check_parallel = true;
        } else if (a == "--allow-sequential") {
            cfg.allow_sequential = true;
        } else if (a == "--size-sweep") {
            size_sweep_mode = true;
        } else if (a == "--size-lo") {
            size_lo = std::atoi(argv[++i]);
        } else if (a == "--size-hi") {
            size_hi = std::atoi(argv[++i]);
        } else if (a == "--max-tpi-nodes") {
            max_tpi = static_cast<size_t>(std::atof(argv[++i]));
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }

    const size_t n = cfg.nodes;
    const size_t edges = n * sta::kFanin;
    const size_t bytes = sta::bytes_per_pass(n);
    const char* layout_name = (cfg.layout == sta::Layout::Soa) ? "soa" : "aos";

    std::fprintf(stderr,
                 "nodes=%zu fanin=%d edges=%zu working_set=%.0f MB layout=%s hw=%u\n",
                 n, sta::kFanin, edges, bytes / 1048576.0, layout_name,
                 std::thread::hardware_concurrency());

    // ALWAYS verify the parallel policy is really parallel, and refuse to
    // produce numbers if it is not.
    //
    // Missing TBB headers or library both fail loudly at build time, because
    // this file uses tbb::global_control directly. The case that does not is a
    // libstdc++ built without TBB support: everything links, everything runs,
    // and every CPU result is silently a single-threaded result. That would
    // make act 2 look like act 1 and the whole comparison would be wrong, so
    // it is a hard error rather than a warning.
    {
        const size_t seen = observed_threads(1u << 21);
        const unsigned hw = std::thread::hardware_concurrency();
        if (cfg.check_parallel) {
            std::fprintf(stderr, "std::execution::par ran on %zu distinct thread(s) -- %s\n",
                         seen, seen > 1 ? "TBB active" : "SEQUENTIAL FALLBACK");
        }
        if (seen <= 1 && hw > 1 && !cfg.allow_sequential) {
            std::fprintf(stderr,
                         "\nFATAL: std::execution::par is running SEQUENTIALLY on a machine\n"
                         "with %u hardware threads. libstdc++ implements the parallel\n"
                         "policies on Intel TBB and silently degrades to serial without it,\n"
                         "so every CPU number this would print would be wrong.\n\n"
                         "  check what the build resolved:  make tbb-info\n"
                         "  install it:                     conda install -c conda-forge tbb-devel\n"
                         "                                  sudo apt install libtbb-dev\n"
                         "                                  sudo dnf install tbb-devel\n"
                         "  then point the build at it:     make TBB_ROOT=$CONDA_PREFIX\n\n"
                         "Pass --allow-sequential to measure the serial fallback on purpose.\n",
                         hw);
            return 3;
        }
    }

    if (size_sweep_mode) {
        std::fprintf(stderr, "size sweep 2^%d .. 2^%d nodes, thread-per-item capped at %zu\n",
                     size_lo, size_hi, max_tpi);
        size_sweep(cfg, size_lo, size_hi, max_tpi);
        return 0;
    }

    // Build the two large vectors.
    std::fprintf(stderr, "generating %.0f MB of input...\n", bytes / 1048576.0);
    std::vector<float> arrival_in(edges), delay(edges), out(n);
    std::for_each(std::execution::par_unseq, arrival_in.begin(), arrival_in.end(),
                  [base = arrival_in.data()](float& x) {
                      x = sta::gen_arrival(static_cast<uint64_t>(&x - base));
                  });
    std::for_each(std::execution::par_unseq, delay.begin(), delay.end(),
                  [base = delay.data()](float& x) {
                      x = sta::gen_delay(static_cast<uint64_t>(&x - base));
                  });

    std::printf("layout,strategy,threads,trial,edges_per_sec,gb_per_sec\n");

    // Act 1 first: one thread per work item.
    for (int t = 0; t < cfg.trials; ++t) {
        const double nodes_s = run_thread_per_item(arrival_in, delay, out, cfg.layout, 2000);
        std::printf("%s,thread_per_item,0,%d,%.0f,%.4f\n", layout_name, t,
                    nodes_s * sta::kFanin, nodes_s * (bytes / n) / 1e9);
    }
    std::fflush(stdout);
    std::fprintf(stderr, "act 1 (thread per item) done\n");

    std::vector<int> sweep;
    for (int p = 1; p <= 24; ++p) sweep.push_back(p);
    for (int p = 28; p <= 64; p += 4) sweep.push_back(p);
    for (int p = 80; p <= 256; p += 16) sweep.push_back(p);

    uint64_t reference = 0;
    for (size_t si = 0; si < sweep.size(); ++si) {
        const int p = sweep[si];
        // Cap the TBB pool. This is the thread-pool size knob.
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism,
                               static_cast<size_t>(p));
        propagate_all(arrival_in, delay, out, cfg.layout);  // warm up / fault in

        for (int t = 0; t < cfg.trials; ++t) {
            const auto t0 = Clock::now();
            for (int r = 0; r < cfg.reps; ++r) propagate_all(arrival_in, delay, out, cfg.layout);
            const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
            const double eps = static_cast<double>(edges) * cfg.reps / secs;
            const double gbs = static_cast<double>(bytes) * cfg.reps / secs / 1e9;
            std::printf("%s,pool,%d,%d,%.0f,%.2f\n", layout_name, p, t, eps, gbs);
        }
        std::fflush(stdout);

        const uint64_t c = sta::checksum(out.data(), n);
        if (si == 0) {
            reference = c;
            std::fprintf(stderr, "checksum=%llu\n", static_cast<unsigned long long>(c));
        } else if (c != reference) {
            std::fprintf(stderr, "\nCHECKSUM MISMATCH at threads=%d\n", p);
            return 1;
        }
        std::fprintf(stderr, "\r  %zu/%zu  threads=%-4d", si + 1, sweep.size(), p);
    }

    std::fprintf(stderr, "\ndone\n");
    return 0;
}
