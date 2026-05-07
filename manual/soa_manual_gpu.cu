#define SOA_MANUAL

#include <benchmarks/common.h>

#include "manual/soa_structs.h"

using namespace manual;

#include "benchmarks/easy.h"
#include "benchmarks/easycompute.h"
#include "benchmarks/real.h"
#include "benchmarks/strided.h"
#include "benchmarks/hard.h"
#include "benchmarks/nbody.h"
#include "benchmarks/stencil.h"
#include "benchmarks/invmass.h"

INSTANTIATE_BENCHMARKS_F1(EasyRW,      manual::S2,       N_large_gpu, GPUBackend);
INSTANTIATE_BENCHMARKS_F1(EasyCompute, manual::S2,       N_small_gpu, GPUBackend);
INSTANTIATE_BENCHMARKS_F1(RealRW,      manual::S10,      N_small_gpu, GPUBackend);
INSTANTIATE_BENCHMARKS_F1(Strided,     manual::S32,      N_large_gpu, GPUBackend);
INSTANTIATE_BENCHMARKS_F1(HardRW,      manual::S64,      N_small_gpu, GPUBackend);
INSTANTIATE_BENCHMARKS_F1(NBody,       manual::Snbody,   N_small_gpu, GPUBackend);
INSTANTIATE_BENCHMARKS_F1(Stencil,     manual::Sstencil, N_large_gpu, GPUBackend);

INSTANTIATE_BENCHMARKS_F2(InvariantMass, manual::PxPyPzM, manual::PxPyPzM, N_large_gpu, GPUBackend);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::AddCustomContext("name", "Manual SoA GPU");
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
