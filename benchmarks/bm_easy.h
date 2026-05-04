#pragma once
#include "benchmarks/common.h"
#include "benchmarks/backend.h"

// Free functions at namespace scope so nvcc allows extended __host__ __device__ lambdas inside them
// (extended lambdas cannot appear inside private class methods, which BENCHMARK_TEMPLATE_METHOD_F generates).

template <Backend B, class T>
void run_BM_CPUEasyRW(benchmark::State &state, std::size_t n, T t)
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

template <Backend B, class T>
void run_BM_CPUEasyCompute(benchmark::State &state, std::size_t n, T t)
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
            MEMBER_ACCESS(t, x0, i) = std::sin(MEMBER_ACCESS(t, x0, i)) +
                                      std::cos(MEMBER_ACCESS(t, x0, i)) * std::tan(MEMBER_ACCESS(t, x0, i)) +
                                      std::sqrt(std::abs(MEMBER_ACCESS(t, x0, i))) * 15.0f -
                                      std::exp(MEMBER_ACCESS(t, x0, i)) * 2.0f;
            MEMBER_ACCESS(t, x1, i) = std::sin(MEMBER_ACCESS(t, x1, i)) +
                                      std::cos(MEMBER_ACCESS(t, x1, i)) * std::tan(MEMBER_ACCESS(t, x1, i)) +
                                      std::sqrt(std::abs(MEMBER_ACCESS(t, x1, i))) * 15.0f -
                                      std::exp(MEMBER_ACCESS(t, x1, i)) * 2.0f;
        });
    }

    backend_allocator<B>::synchronize();
    // f(0) = sin(0) + cos(0)*tan(0) + sqrt(0)*15 - exp(0)*2 = -2
    const int expected = static_cast<int>(std::sin(0) + std::cos(0) * std::tan(0) +
                                          std::sqrt(std::abs(0)) * 15.0f - std::exp(0) * 2.0f);
    for (size_t i = 0; i < n; ++i) {
        CheckResult(state, expected, MEMBER_ACCESS(t, x0, i), "x0");
        CheckResult(state, expected, MEMBER_ACCESS(t, x1, i), "x1");
    }
    state.counters["n_elem"] = n;
}

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, BM_CPUEasyRW)(benchmark::State &state)
{
    constexpr Backend B = std::remove_reference_t<decltype(*this)>::backend;
    run_BM_CPUEasyRW<B>(state, this->n, this->t);
}

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, BM_CPUEasyCompute)(benchmark::State &state)
{
    constexpr Backend B = std::remove_reference_t<decltype(*this)>::backend;
    run_BM_CPUEasyCompute<B>(state, this->n, this->t);
}
