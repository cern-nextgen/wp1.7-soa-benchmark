#include <benchmark/benchmark.h>

#include "benchmark_gpu.h"
#include "benchmark_find_max_gpu.h"
#include "benchmark_estimate_pi_gpu.h"
#include "benchmark_bitonic_sort_gpu.h"

template<wrapper::layout L>
struct CreateWrapper {
    wrapper::wrapper<S3_1, device_memory_array, L> operator()(int n) {
        if constexpr (L == wrapper::layout::soa) return {n, n, n};
        else return {n};
    }
};

int main(int argc, char** argv) {
    constexpr int N[] = {1<<10, 1<<12, 1<<14, 1<<16, 1<<18, 1<<20};
    // constexpr int N[] = {1<<20};

    /*
    for (int n : N) {
        benchmark::RegisterBenchmark("BM_GPUTest", BM_GPUTest)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }
    */

    for (int n : N) {
        using Create = CreateWrapper<wrapper::layout::soa>;
        using KernelInput = wrapper::wrapper<S3_1, std::span, wrapper::layout::soa>;
        benchmark::RegisterBenchmark("MAX_GPUTest_SOA", MAX_GPUTest<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    for (int n : N) {
        using Create = CreateWrapper<wrapper::layout::aos>;
        using KernelInput = wrapper::wrapper<S3_1, std::span, wrapper::layout::aos>;
        benchmark::RegisterBenchmark("MAX_GPUTest_AOS", MAX_GPUTest<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
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