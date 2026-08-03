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

    if (cfg.check_parallel) {
        const size_t seen = observed_threads(1u << 22);
        std::fprintf(stderr, "std::execution::par ran on %zu distinct thread(s) -- %s\n", seen,
                     seen > 1 ? "TBB active" : "SEQUENTIAL FALLBACK, TBB is not linked!");
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

    std::printf("layout,threads,trial,edges_per_sec,gb_per_sec\n");

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
            std::printf("%s,%d,%d,%.0f,%.2f\n", layout_name, p, t, eps, gbs);
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
