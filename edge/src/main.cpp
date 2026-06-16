// Edge inference CLI for the reactor surrogate.
//
//   ./synfuel_edge --bench [N]   Benchmark inference latency over N calls (default 2e6).
//   ./synfuel_edge               Read inputs from stdin (one row of 5 floats per line),
//                                write the predicted 6-state per line. This is the
//                                request/response interface the Phase 6 bridge drives.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "mlp.h"

namespace {

int run_bench(long n) {
    float in[5] = {3.0f, 25.0f, 505.0f, 1.0f, 6.0e4f};
    float out[6];
    // Warm up.
    for (int i = 0; i < 1000; ++i) synfuel::predict(in, out);

    double checksum = 0.0;  // accumulate so the optimiser cannot elide the work
    auto t0 = std::chrono::high_resolution_clock::now();
    for (long i = 0; i < n; ++i) {
        in[2] = 470.0f + static_cast<float>(i % 70);  // vary an input each call
        synfuel::predict(in, out);
        checksum += out[5];
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    double per_call = ns / static_cast<double>(n);
    std::printf("inferences      : %ld\n", n);
    std::printf("mean latency    : %.1f ns  (%.4f us)\n", per_call, per_call / 1000.0);
    std::printf("throughput      : %.2f million inferences/sec\n", 1000.0 / per_call);
    std::fprintf(stderr, "checksum=%.6f\n", checksum);  // keep the result live
    return 0;
}

int run_stdin() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        float in[5];
        bool ok = true;
        for (float& v : in) {
            if (!(ss >> v)) { ok = false; break; }
        }
        if (!ok) {
            std::fprintf(stderr, "skipping malformed line: %s\n", line.c_str());
            continue;
        }
        float out[6];
        synfuel::predict(in, out);
        std::printf("%.6f %.6f %.6f %.6f %.6f %.6f\n",
                    out[0], out[1], out[2], out[3], out[4], out[5]);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--bench") {
        long n = (argc >= 3) ? std::strtol(argv[2], nullptr, 10) : 2'000'000L;
        return run_bench(n);
    }
    return run_stdin();
}
