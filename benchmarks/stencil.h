#ifndef BENCHMARKS_STENCIL_H
#define BENCHMARKS_STENCIL_H

#include "benchmarks/common.h"

#include <cmath>

template <Backend B, class T>
void run_Stencil(benchmark::State &state, std::size_t n, T t)
{
    constexpr double L = 1.0;
    const double dx = L / static_cast<double>(n - 1);
    const double dx2 = dx * dx;

    parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
        const double x = static_cast<double>(i) * dx;
        MEMBER_ACCESS(t, rhs, i) = -x * (x + 3) * std::exp(x);
    });
    backend_allocator<B>::synchronize();

    for (auto _ : state) {
        state.PauseTiming();
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, src, i) = 0.0;
            MEMBER_ACCESS(t, dst, i) = 0.0;
        });
        backend_allocator<B>::synchronize();
        state.ResumeTiming();

        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            if (i == 0 || i >= n - 1) return;
            const double u_left  = MEMBER_ACCESS(t, src, i - 1);
            const double u_right = MEMBER_ACCESS(t, src, i + 1);
            const double f       = MEMBER_ACCESS(t, rhs, i);
            MEMBER_ACCESS(t, dst, i) = 0.5 * (u_left + u_right + dx2 * f);
        });
        backend_allocator<B>::synchronize();

        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            if (i == 0 || i >= n - 1) return;
            const double u_left  = MEMBER_ACCESS(t, dst, i - 1);
            const double u_right = MEMBER_ACCESS(t, dst, i + 1);
            const double f       = MEMBER_ACCESS(t, rhs, i);
            MEMBER_ACCESS(t, src, i) = 0.5 * (u_left + u_right + dx2 * f);
        });
    }

    backend_allocator<B>::synchronize();
    state.counters["n_elem"] = static_cast<double>(n);
    state.counters["N^2_interactions"] =
        benchmark::Counter(static_cast<double>(n) * 2.0, benchmark::Counter::kIsRate);
}

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, Stencil)(benchmark::State &state)
{
    constexpr Backend B = std::remove_reference_t<decltype(*this)>::backend;
    run_Stencil<B>(state, this->n, this->t);
}

#endif // BENCHMARKS_STENCIL_H
