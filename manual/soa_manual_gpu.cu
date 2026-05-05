#define SOA_MANUAL

#include "manual/soa_structs.h"

#include "benchmarks/easy.h"
#include "benchmarks/easycompute.h"

INSTANTIATE_BENCHMARKS_F1(EasyRW,      S2, N_GPU, GPUBackend);
INSTANTIATE_BENCHMARKS_F1(EasyCompute, S2, N_GPU, GPUBackend);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::AddCustomContext("name", "Manual SoA GPU");
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
