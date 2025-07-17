#include <benchmark/benchmark.h>

#include "benchmark_gpu.h"

int main(int argc, char** argv) {
    constexpr int N[] = {1<<10, 1<<12, 1<<14, 1<<16, 1<<18, 1<<20};

    for (int n : N) {
        benchmark::RegisterBenchmark("BM_GPUTest", BM_GPUTest)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    for (int n : N) {
        benchmark::RegisterBenchmark("ARIT_GPUTest", ARIT_GPUTest)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}