#include "memlayout/structs.h"

#include "benchmarks/easy.h"
#include "benchmarks/easycompute.h"

constexpr memlayout::Layout L = memlayout::Layout::aos;

using S2ArrayType       = memlayout::Wrapper<S2,       memlayout::pointer, L>;
using S10ArrayType      = memlayout::Wrapper<S10,      memlayout::pointer, L>;
using S32ArrayType      = memlayout::Wrapper<S32,      memlayout::pointer, L>;
using S64ArrayType      = memlayout::Wrapper<S64,      memlayout::pointer, L>;
using SnbodyArrayType   = memlayout::Wrapper<Snbody,   memlayout::pointer, L>;
using SstencilArrayType = memlayout::Wrapper<Sstencil, memlayout::pointer, L>;
using PxPyPzMArrayType  = memlayout::Wrapper<PxPyPzM,  memlayout::pointer, L>;

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
