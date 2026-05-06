#include "memlayout/structs.h"
using namespace memlayout;

constexpr memlayout::Layout L = memlayout::Layout::soa;

using S2ArrayType       = memlayout::Wrapper<S2,       memlayout::pointer, L>;
using S10ArrayType      = memlayout::Wrapper<S10,      memlayout::pointer, L>;
using S32ArrayType      = memlayout::Wrapper<S32,      memlayout::pointer, L>;
using S64ArrayType      = memlayout::Wrapper<S64,      memlayout::pointer, L>;
using SnbodyArrayType   = memlayout::Wrapper<Snbody,   memlayout::pointer, L>;
using SstencilArrayType = memlayout::Wrapper<Sstencil, memlayout::pointer, L>;
using PxPyPzMArrayType  = memlayout::Wrapper<PxPyPzM,  memlayout::pointer, L>;

#include "benchmarks/easy.h"
#include "benchmarks/easycompute.h"
#include "benchmarks/real.h"
#include "benchmarks/strided.h"
#include "benchmarks/hard.h"
#include "benchmarks/nbody.h"
#include "benchmarks/stencil.h"
#include "benchmarks/invmass.h"

INSTANTIATE_BENCHMARKS_F1(EasyRW,      S2ArrayType,       N_Large, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(EasyCompute, S2ArrayType,       N, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(RealRW,      S10ArrayType,      N, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(Strided,     S32ArrayType,      N_Large, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(HardRW,      S64ArrayType,      N, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(NBody,       SnbodyArrayType,   N, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(Stencil,     SstencilArrayType, N_Large, CPUBackend);

INSTANTIATE_BENCHMARKS_F2(InvariantMass, PxPyPzMArrayType, PxPyPzMArrayType, N_Large, CPUBackend);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::AddCustomContext("name", "Template SoA CPU");
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
