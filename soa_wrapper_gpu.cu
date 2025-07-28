#include <benchmark/benchmark.h>

#include "benchmark_gpu.h"
#include "benchmark_find_max_gpu.h"
#include "benchmark_estimate_pi_gpu.h"
#include "benchmark_bitonic_sort_gpu.h"

int main(int argc, char** argv) {
    constexpr int N[] = {1<<10, 1<<12, 1<<14, 1<<16, 1<<18, 1<<20};
    // constexpr int N[] = {1<<20};

    /*
    for (int n : N) {
        benchmark::RegisterBenchmark("BM_GPUTest", BM_GPUTest)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }
    */

    for (int n : N) {
        benchmark::RegisterBenchmark("MAX_GPUTest", MAX_GPUTest)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    
    for (int n : N) {
        benchmark::RegisterBenchmark("BITONIC_Simp", BITONIC_Simp)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }
    

    for (int n : N) {
        benchmark::RegisterBenchmark("PiSimp_GPUTest", PiSimp_GPUTest)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}