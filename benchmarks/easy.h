#ifndef BENCHMARKS_EASY_H
#define BENCHMARKS_EASY_H

#include "benchmarks/common.h"

// Free functions at namespace scope so nvcc allows extended __host__ __device__ lambdas inside them
// (extended lambdas cannot appear inside private class methods, which BENCHMARK_TEMPLATE_METHOD_F generates).

template <Backend B, class T>
void run_EasyRW(benchmark::State &state, std::size_t n, T t)
{
    for (auto _ : state) {
        state.PauseTiming();
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x0, i) = 0;
            MEMBER_ACCESS(t, x1, i) = 0;
        });
        backend_allocator<B>::synchronize();
        state.ResumeTiming();

        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x0, i) += 2;
            MEMBER_ACCESS(t, x1, i) += 2;
        });
    }

    backend_allocator<B>::synchronize();
    for (size_t i = 0; i < n; ++i) {
        CheckResult(state, 2, MEMBER_ACCESS(t, x0, i), "x0");
        CheckResult(state, 2, MEMBER_ACCESS(t, x1, i), "x1");
    }
    state.counters["n_elem"] = n;
}

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, EasyRW)(benchmark::State &state)
{
    constexpr Backend B = std::remove_reference_t<decltype(*this)>::backend;
    run_EasyRW<B>(state, this->n, this->t);
}

#endif // BENCHMARKS_EASY_H
