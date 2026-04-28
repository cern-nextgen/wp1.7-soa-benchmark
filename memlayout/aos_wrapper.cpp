#include "structs.h"

constexpr memlayout::Layout L = memlayout::Layout::aos;

using S2ArrayType       = memlayout::Wrapper<S2,       memlayout::pointer, L>;
using S10ArrayType      = memlayout::Wrapper<S10,      memlayout::pointer, L>;
using S32ArrayType      = memlayout::Wrapper<S32,      memlayout::pointer, L>;
using S64ArrayType      = memlayout::Wrapper<S64,      memlayout::pointer, L>;
using SnbodyArrayType   = memlayout::Wrapper<Snbody,   memlayout::pointer, L>;
using SstencilArrayType = memlayout::Wrapper<Sstencil, memlayout::pointer, L>;
using PxPyPzMArrayType  = memlayout::Wrapper<PxPyPzM,  memlayout::pointer, L>;

#include "benchmarks/bm_easy.h"
#include "benchmarks/bm_real.h"
#include "benchmarks/bm_strided.h"
#include "benchmarks/bm_hard.h"
#include "benchmarks/bm_nbody.h"
#include "benchmarks/bm_stencil.h"
#include "benchmarks/bm_invmass.h"

INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyRW,      S2ArrayType,       N_Large);
INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyCompute, S2ArrayType,       N);
INSTANTIATE_BENCHMARKS_F1(BM_CPURealRW,      S10ArrayType,      N);
INSTANTIATE_BENCHMARKS_F1(BM_CPUStrided,     S32ArrayType,      N_Large);
INSTANTIATE_BENCHMARKS_F1(BM_CPUHardRW,      S64ArrayType,      N);
INSTANTIATE_BENCHMARKS_F1(BM_nbody,          SnbodyArrayType,   N);
INSTANTIATE_BENCHMARKS_F1(BM_stencil,        SstencilArrayType, N_Large);

INSTANTIATE_BENCHMARKS_F2(BM_InvariantMass, PxPyPzMArrayType, PxPyPzMArrayType, N_Large);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::AddCustomContext("name", "Template AoS");
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
