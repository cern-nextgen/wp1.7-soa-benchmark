#ifndef BENCHMARKS_STENCIL_H
#define BENCHMARKS_STENCIL_H

#include "benchmarks/common.h"

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, Stencil)(benchmark::State &state)
{
    auto n = this->n;
    auto &t = this->t;
    constexpr double L = 1.0;
    const double dx = L / static_cast<double>(n - 1);
    const double dx2 = dx * dx;

    // rhs is a function of position only — initialise once, never reset
    for (size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i) * dx;
        MEMBER_ACCESS(t, rhs, i) = -x * (x + 3) * std::exp(x);
    }

    for (auto _ : state) {
        state.PauseTiming();
        for (size_t i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, src, i) = 0.0;
            MEMBER_ACCESS(t, dst, i) = 0.0;
        }
        state.ResumeTiming();

        for (size_t i = 1; i < n - 1; ++i) {
            const double u_left  = MEMBER_ACCESS(t, src, i - 1);
            const double u_right = MEMBER_ACCESS(t, src, i + 1);
            const double f       = MEMBER_ACCESS(t, rhs, i);
            MEMBER_ACCESS(t, dst, i) = 0.5 * (u_left + u_right + dx2 * f);
        }

        for (size_t i = 1; i < n - 1; ++i) {
            const double u_left  = MEMBER_ACCESS(t, dst, i - 1);
            const double u_right = MEMBER_ACCESS(t, dst, i + 1);
            const double f       = MEMBER_ACCESS(t, rhs, i);
            MEMBER_ACCESS(t, src, i) = 0.5 * (u_left + u_right + dx2 * f);
        }
    }

    state.counters["n_elem"] = static_cast<double>(n);
    state.counters["N^2_interactions"] =
        benchmark::Counter(static_cast<double>(n) * 2.0, benchmark::Counter::kIsRate);
}

#endif // BENCHMARKS_STENCIL_H
