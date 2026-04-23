#pragma once
#include "benchmarks/common.h"

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, BM_CPUStrided)(benchmark::State &state)
{
    auto n = this->n;
    auto &t = this->t;

    for (size_t i = 0; i < n; ++i) {
        MEMBER_ACCESS(t, x0,  i) = 0.0f;
        MEMBER_ACCESS(t, x1,  i) = (float)i; MEMBER_ACCESS(t, x2,  i) = (float)i;
        MEMBER_ACCESS(t, x3,  i) = (float)i; MEMBER_ACCESS(t, x4,  i) = (float)i;
        MEMBER_ACCESS(t, x5,  i) = (float)i; MEMBER_ACCESS(t, x6,  i) = (float)i;
        MEMBER_ACCESS(t, x7,  i) = (float)i; MEMBER_ACCESS(t, x8,  i) = (float)i;
        MEMBER_ACCESS(t, x9,  i) = (float)i; MEMBER_ACCESS(t, x10, i) = (float)i;
        MEMBER_ACCESS(t, x11, i) = (float)i; MEMBER_ACCESS(t, x12, i) = (float)i;
        MEMBER_ACCESS(t, x13, i) = (float)i; MEMBER_ACCESS(t, x14, i) = (float)i;
        MEMBER_ACCESS(t, x15, i) = (float)i; MEMBER_ACCESS(t, x16, i) = (float)i;
        MEMBER_ACCESS(t, x17, i) = (float)i; MEMBER_ACCESS(t, x18, i) = (float)i;
        MEMBER_ACCESS(t, x19, i) = (float)i; MEMBER_ACCESS(t, x20, i) = (float)i;
        MEMBER_ACCESS(t, x21, i) = (float)i; MEMBER_ACCESS(t, x22, i) = (float)i;
        MEMBER_ACCESS(t, x23, i) = (float)i; MEMBER_ACCESS(t, x24, i) = (float)i;
        MEMBER_ACCESS(t, x25, i) = (float)i; MEMBER_ACCESS(t, x26, i) = (float)i;
        MEMBER_ACCESS(t, x27, i) = (float)i; MEMBER_ACCESS(t, x28, i) = (float)i;
        MEMBER_ACCESS(t, x29, i) = (float)i; MEMBER_ACCESS(t, x30, i) = (float)i;
        MEMBER_ACCESS(t, x31, i) = (float)i;
    }

    constexpr size_t stride = 23;

    // x1..x31 are read-only; x0 is overwritten deterministically each iteration
    for (auto _ : state) {
        for (size_t i = 0; i * stride < n; i += stride) {
            MEMBER_ACCESS(t, x0, i) = std::sqrt(
                MEMBER_ACCESS(t, x1,  i) + MEMBER_ACCESS(t, x2,  i) + MEMBER_ACCESS(t, x3,  i) +
                MEMBER_ACCESS(t, x4,  i) + MEMBER_ACCESS(t, x5,  i) + MEMBER_ACCESS(t, x6,  i) +
                MEMBER_ACCESS(t, x7,  i) + MEMBER_ACCESS(t, x8,  i) + MEMBER_ACCESS(t, x9,  i) +
                MEMBER_ACCESS(t, x10, i) + MEMBER_ACCESS(t, x11, i) + MEMBER_ACCESS(t, x12, i) +
                MEMBER_ACCESS(t, x13, i) + MEMBER_ACCESS(t, x14, i) + MEMBER_ACCESS(t, x15, i) +
                MEMBER_ACCESS(t, x16, i) + MEMBER_ACCESS(t, x17, i) + MEMBER_ACCESS(t, x18, i) +
                MEMBER_ACCESS(t, x19, i) + MEMBER_ACCESS(t, x20, i) + MEMBER_ACCESS(t, x21, i) +
                MEMBER_ACCESS(t, x22, i) + MEMBER_ACCESS(t, x23, i) + MEMBER_ACCESS(t, x24, i) +
                MEMBER_ACCESS(t, x25, i) + MEMBER_ACCESS(t, x26, i) + MEMBER_ACCESS(t, x27, i) +
                MEMBER_ACCESS(t, x28, i) + MEMBER_ACCESS(t, x29, i) + MEMBER_ACCESS(t, x30, i) +
                MEMBER_ACCESS(t, x31, i));
            benchmark::DoNotOptimize(MEMBER_ACCESS(t, x0, i));
        }
    }

    for (size_t i = 0; i * stride < n; i += stride) {
        int difference = (int)std::abs(MEMBER_ACCESS(t, x0, i) - std::sqrt(31.0f * i));
        CheckResult(state, 0, difference, "difference");
    }
    state.counters["n_elem"] = n;
}
