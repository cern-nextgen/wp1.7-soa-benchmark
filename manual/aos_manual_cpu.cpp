#define AOS_MANUAL

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

INSTANTIATE_BENCHMARKS_F1(EasyRW,      S2,       N_Large, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(EasyCompute, S2,       N, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(RealRW,      S10,      N, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(Strided,     S32,      N_Large, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(HardRW,      S64,      N, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(NBody,       Snbody,   N, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(Stencil,     Sstencil, N_Large, CPUBackend);

INSTANTIATE_BENCHMARKS_F2(InvariantMass, Particle, Particle, N_Large, CPUBackend);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::AddCustomContext("name", "Manual AoS CPU");
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
