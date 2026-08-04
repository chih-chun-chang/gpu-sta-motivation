// CPU side of the statistical (POCV) propagation benchmark.
//
// Same harness as ../src/bench_sta.cpp: std::for_each with par_unseq, pool size
// through tbb::global_control, no hand-rolled threading. Reports achieved
// bandwidth and achieved GFLOP/s so the kernel can be placed on the roofline.
//
// Output: CSV on stdout.

#include "ssta.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <string>
#include <thread>
#include <vector>

#include <oneapi/tbb/global_control.h>

using Clock = std::chrono::steady_clock;

namespace {

struct Buffers {
    std::vector<float> prm, prs, pfm, pfs, crm, crs, cfm, cfs;
    std::vector<uint8_t> sense;
    std::vector<float> o_rm, o_rs, o_fm, o_fs;
};

void propagate_all(const Buffers& b, Buffers& out, size_t n) {
    const float *prm = b.prm.data(), *prs = b.prs.data(), *pfm = b.pfm.data(),
                *pfs = b.pfs.data(), *crm = b.crm.data(), *crs = b.crs.data(),
                *cfm = b.cfm.data(), *cfs = b.cfs.data();
    const uint8_t* sn = b.sense.data();
    float *orm = out.o_rm.data(), *ors = out.o_rs.data(), *ofm = out.o_fm.data(),
          *ofs = out.o_fs.data();
    const float* base = out.o_rm.data();

    std::for_each(std::execution::par_unseq, out.o_rm.begin(), out.o_rm.end(),
                  [=](float& slot) {
                      const size_t i = static_cast<size_t>(&slot - base);
                      const ssta::Result r = ssta::propagate(prm, prs, pfm, pfs, crm, crs,
                                                             cfm, cfs, sn, n, i);
                      orm[i] = r.rise_mean;
                      ors[i] = r.rise_std;
                      ofm[i] = r.fall_mean;
                      ofs[i] = r.fall_std;
                  });
}

}  // namespace

int main(int argc, char** argv) {
    size_t nodes = 8u << 20;
    int reps = 5, trials = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--nodes") nodes = static_cast<size_t>(std::atof(argv[++i]));
        else if (a == "--reps") reps = std::atoi(argv[++i]);
        else if (a == "--trials") trials = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }

    const size_t n = nodes, edges = n * ssta::kFanin;
    const size_t bytes = ssta::bytes_per_pass(n);
    const size_t flops = ssta::flops_per_pass(n);
    std::fprintf(stderr,
                 "nodes=%zu fanin=%d edges=%zu working_set=%.0f MB "
                 "intensity=%.3f flops/byte hw=%u\n",
                 n, ssta::kFanin, edges, bytes / 1048576.0,
                 static_cast<double>(flops) / bytes, std::thread::hardware_concurrency());

    Buffers b, out;
    auto fill = [&](std::vector<float>& v, uint64_t salt, bool is_std) {
        v.resize(edges);
        float* p = v.data();
        std::for_each(std::execution::par_unseq, v.begin(), v.end(), [=](float& x) {
            const uint64_t e = static_cast<uint64_t>(&x - p);
            x = is_std ? ssta::gen_std(e, salt) : ssta::gen_mean(e, salt);
        });
    };
    fill(b.prm, 0x11, false); fill(b.prs, 0x12, true);
    fill(b.pfm, 0x13, false); fill(b.pfs, 0x14, true);
    fill(b.crm, 0x21, false); fill(b.crs, 0x22, true);
    fill(b.cfm, 0x23, false); fill(b.cfs, 0x24, true);
    b.sense.resize(edges);
    {
        uint8_t* p = b.sense.data();
        std::for_each(std::execution::par_unseq, b.sense.begin(), b.sense.end(),
                      [=](uint8_t& x) { x = ssta::gen_sense(static_cast<uint64_t>(&x - p)); });
    }
    out.o_rm.resize(n); out.o_rs.resize(n); out.o_fm.resize(n); out.o_fs.resize(n);

    std::printf("threads,trial,gb_per_sec,gflop_per_sec,ms\n");
    std::vector<int> sweep;
    for (int p = 1; p <= 24; ++p) sweep.push_back(p);
    for (int p = 28; p <= 64; p += 4) sweep.push_back(p);

    uint64_t reference = 0;
    for (size_t si = 0; si < sweep.size(); ++si) {
        const int p = sweep[si];
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism,
                               static_cast<size_t>(p));
        propagate_all(b, out, n);  // warm

        for (int t = 0; t < trials; ++t) {
            const auto t0 = Clock::now();
            for (int r = 0; r < reps; ++r) propagate_all(b, out, n);
            const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
            std::printf("%d,%d,%.3f,%.3f,%.4f\n", p, t,
                        static_cast<double>(bytes) * reps / secs / 1e9,
                        static_cast<double>(flops) * reps / secs / 1e9,
                        secs * 1e3 / reps);
        }
        std::fflush(stdout);

        const uint64_t c = ssta::checksum(out.o_rm.data(), out.o_rs.data(), out.o_fm.data(),
                                          out.o_fs.data(), n);
        if (si == 0) {
            reference = c;
            std::fprintf(stderr, "checksum=%llu\n", static_cast<unsigned long long>(c));
        } else if (c != reference) {
            std::fprintf(stderr, "\nCHECKSUM MISMATCH at threads=%d\n", p);
            return 1;
        }
        std::fprintf(stderr, "\r  %zu/%zu threads=%-4d", si + 1, sweep.size(), p);
    }
    std::fprintf(stderr, "\ndone\n");
    return 0;
}
