#define AOS_MANUAL

#include "benchmarks/common.h"

struct S2 {
    int x0, x1;
};

/// Fixtures ///

template <typename S, typename N, Backend B = Backend::GPU>
class Fixture1 : public benchmark::Fixture {
public:
    static constexpr auto n = N::value;
    static constexpr Backend backend = B;
    S* t = nullptr;

    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;
    void SetUp(benchmark::State&) override   { t = backend_allocator<B>::template alloc<S>(n); }
    void TearDown(benchmark::State&) override { backend_allocator<B>::free(t); t = nullptr; }
};

/// Benchmarks ///

#include "benchmarks/easy.h"
#include "benchmarks/easycompute.h"

INSTANTIATE_BENCHMARKS_F1(EasyRW,      S2, N_GPU);
INSTANTIATE_BENCHMARKS_F1(EasyCompute, S2, N_GPU);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::AddCustomContext("name", "Manual AoS GPU");
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
