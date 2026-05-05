#define SOA_MANUAL

#include <type_traits>
#include <Eigen/Core>
#include "benchmarks/common.h"

//////////////// Allocation helpers (Backend-templated)

template <Backend B>
struct mallocator {
    std::size_t n;
    template <class... Ptrs>
    void operator()(Ptrs&... args) const {
        ((args = backend_allocator<B>::template alloc<std::remove_reference_t<decltype(*args)>>(n)), ...);
    }
};

template <Backend B>
struct deallocator {
    template <class... Ptrs>
    void operator()(Ptrs&... args) const {
        ((backend_allocator<B>::free(args), args = nullptr), ...);
    }
};

template <Backend B, class S>
void allocate(S& s, std::size_t n) { s.apply(mallocator<B>{n}); }

template <Backend B, class S>
void deallocate(S& s) { s.apply(deallocator<B>{}); }

//////////////// Data structures

#define SOA_APPLY(...) \
    template <class F> void apply(F&& f) { f(__VA_ARGS__); } \
    template <class F> void apply(F&& f) const { f(__VA_ARGS__); }

struct S2 {
    int *__restrict__ x0, *__restrict__ x1;
    SOA_APPLY(x0, x1)
};

/// Fixtures ///

template <typename S, typename N, Backend B = Backend::GPU>
class Fixture1 : public benchmark::Fixture {
public:
    static constexpr auto n = N::value;
    static constexpr Backend backend = B;
    S t;

    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;
    void SetUp(benchmark::State&) override   { allocate<B>(t, n); }
    void TearDown(benchmark::State&) override { deallocate<B>(t); }
};

/// Benchmarks ///

#include "benchmarks/easy.h"
#include "benchmarks/easycompute.h"

INSTANTIATE_BENCHMARKS_F1(EasyRW,      S2, N_GPU);
INSTANTIATE_BENCHMARKS_F1(EasyCompute, S2, N_GPU);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::AddCustomContext("name", "Manual SoA GPU");
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
