#include "reflections/structs.h"
using namespace reflections;

#include "benchmarks/easy.h"
#include "benchmarks/easycompute.h"
#include "benchmarks/real.h"
#include "benchmarks/strided.h"
#include "benchmarks/hard.h"
#include "benchmarks/nbody.h"
#include "benchmarks/stencil.h"
#include "benchmarks/invmass.h"

INSTANTIATE_BENCHMARKS_F1(EasyRW,      S2SoA,       N_large_cpu, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(EasyCompute, S2SoA,       N_small_cpu, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(RealRW,      S10SoA,      N_small_cpu, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(Strided,     S32SoA,      N_large_cpu, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(HardRW,      S64SoA,      N_small_cpu, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(NBody,       SnbodySoA,   N_small_cpu, CPUBackend);
INSTANTIATE_BENCHMARKS_F1(Stencil,     SstencilSoA, N_large_cpu, CPUBackend);

INSTANTIATE_BENCHMARKS_F2(InvariantMass, PxPyPzMSoA, PxPyPzMSoA, N_large_cpu, CPUBackend);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::AddCustomContext("name", "Reflection SoA CPU");
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
