#define AOS_MANUAL

#include <benchmarks/common.h>

#include "manual/aos_structs.h"

using namespace manual;

#include "benchmarks/easy.h"
#include "benchmarks/easycompute.h"
#include "benchmarks/real.h"
#include "benchmarks/strided.h"
#include "benchmarks/hard.h"
#include "benchmarks/nbody.h"
#include "benchmarks/stencil.h"
#include "benchmarks/invmass.h"

INSTANTIATE_BENCHMARKS_F1(EasyRW,      manual::S2,       N_Large, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(EasyCompute, manual::S2,       N, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(RealRW,      manual::S10,      N, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(Strided,     manual::S32,      N_Large, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(HardRW,      manual::S64,      N, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(NBody,       manual::Snbody,   N, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(Stencil,     manual::Sstencil, N_Large, CPUBackend);

INSTANTIATE_BENCHMARKS_F2(InvariantMass, manual::PxPyPzM, manual::PxPyPzM, N_Large, CPUBackend);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::AddCustomContext("name", "Manual AoS GPU");
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
