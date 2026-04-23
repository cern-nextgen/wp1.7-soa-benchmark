#pragma once
#include "benchmarks/common.h"

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, BM_CPUEasyRW)(benchmark::State &state)
{
    auto n = this->n;
    auto &t = this->t;

    for (auto _ : state) {
        state.PauseTiming();
        for (size_t i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x0, i) = 0;
            MEMBER_ACCESS(t, x1, i) = 0;
        }
        state.ResumeTiming();

        for (size_t i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x0, i) += 2;
            MEMBER_ACCESS(t, x1, i) += 2;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        CheckResult(state, 2, MEMBER_ACCESS(t, x0, i), "x0");
        CheckResult(state, 2, MEMBER_ACCESS(t, x1, i), "x1");
    }
    state.counters["n_elem"] = n;
}

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, BM_CPUEasyCompute)(benchmark::State &state)
{
    auto n = this->n;
    auto &t = this->t;

    for (auto _ : state) {
        state.PauseTiming();
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x0, i) = 0;
            MEMBER_ACCESS(t, x1, i) = 0;
        }
        state.ResumeTiming();

        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x0, i) = std::sin(MEMBER_ACCESS(t, x0, i)) +
                                      std::cos(MEMBER_ACCESS(t, x0, i)) * std::tan(MEMBER_ACCESS(t, x0, i)) +
                                      std::sqrt(std::abs(MEMBER_ACCESS(t, x0, i))) * 15.0f -
                                      std::exp(MEMBER_ACCESS(t, x0, i)) * 2.0f;
            MEMBER_ACCESS(t, x1, i) = std::sin(MEMBER_ACCESS(t, x1, i)) +
                                      std::cos(MEMBER_ACCESS(t, x1, i)) * std::tan(MEMBER_ACCESS(t, x1, i)) +
                                      std::sqrt(std::abs(MEMBER_ACCESS(t, x1, i))) * 15.0f -
                                      std::exp(MEMBER_ACCESS(t, x1, i)) * 2.0f;
        }
    }

    // f(0) = sin(0) + cos(0)*tan(0) + sqrt(0)*15 - exp(0)*2 = -2
    const int expected = static_cast<int>(std::sin(0) + std::cos(0) * std::tan(0) +
                                          std::sqrt(std::abs(0)) * 15.0f - std::exp(0) * 2.0f);
    for (int i = 0; i < n; ++i) {
        CheckResult(state, expected, MEMBER_ACCESS(t, x0, i), "x0");
        CheckResult(state, expected, MEMBER_ACCESS(t, x1, i), "x1");
    }
    state.counters["n_elem"] = n;
}
