#include "memlayout/structs.h"

constexpr memlayout::Layout L = memlayout::Layout::aos;

using S2ArrayType = memlayout::Wrapper<S2, memlayout::pointer, L>;

#include "benchmarks/easy.h"
#include "benchmarks/easycompute.h"

INSTANTIATE_BENCHMARKS_F1(EasyRW,      S2ArrayType, N_GPU, GPUBackend);
INSTANTIATE_BENCHMARKS_F1(EasyCompute, S2ArrayType, N_GPU, GPUBackend);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::AddCustomContext("name", "Template AoS GPU");
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
