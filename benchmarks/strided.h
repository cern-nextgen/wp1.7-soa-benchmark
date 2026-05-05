#ifndef BENCHMARKS_STRIDED_H
#define BENCHMARKS_STRIDED_H

#include "benchmarks/common.h"

#include <cmath>
#include <string>

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, Strided)(benchmark::State &state)
{
    auto n = this->n;
    auto &t = this->t;

    for (size_t i = 0; i < n; ++i) {
        MEMBER_ACCESS(t, x0,  i) = 0;
        MEMBER_ACCESS(t, x1,  i) = i; MEMBER_ACCESS(t, x2,  i) = i;
        MEMBER_ACCESS(t, x3,  i) = i; MEMBER_ACCESS(t, x4,  i) = i;
        MEMBER_ACCESS(t, x5,  i) = i; MEMBER_ACCESS(t, x6,  i) = i;
        MEMBER_ACCESS(t, x7,  i) = i; MEMBER_ACCESS(t, x8,  i) = i;
        MEMBER_ACCESS(t, x9,  i) = i; MEMBER_ACCESS(t, x10, i) = i;
        MEMBER_ACCESS(t, x11, i) = i; MEMBER_ACCESS(t, x12, i) = i;
        MEMBER_ACCESS(t, x13, i) = i; MEMBER_ACCESS(t, x14, i) = i;
        MEMBER_ACCESS(t, x15, i) = i; MEMBER_ACCESS(t, x16, i) = i;
        MEMBER_ACCESS(t, x17, i) = i; MEMBER_ACCESS(t, x18, i) = i;
        MEMBER_ACCESS(t, x19, i) = i; MEMBER_ACCESS(t, x20, i) = i;
        MEMBER_ACCESS(t, x21, i) = i; MEMBER_ACCESS(t, x22, i) = i;
        MEMBER_ACCESS(t, x23, i) = i; MEMBER_ACCESS(t, x24, i) = i;
        MEMBER_ACCESS(t, x25, i) = i; MEMBER_ACCESS(t, x26, i) = i;
        MEMBER_ACCESS(t, x27, i) = i; MEMBER_ACCESS(t, x28, i) = i;
        MEMBER_ACCESS(t, x29, i) = i; MEMBER_ACCESS(t, x30, i) = i;
        MEMBER_ACCESS(t, x31, i) = i;
    }

    constexpr size_t stride = 23;

    for (auto _ : state) {
        for (size_t j = 0; j < n; ++j) {
            std::size_t i = (j * stride) % n;
            MEMBER_ACCESS(t, x0, i) = (uint32_t)std::sqrt(
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
        }
    }

    for (size_t j = 0; j < n; ++j) {
        std::size_t i = j * stride % n;
        uint32_t expected = (uint32_t)std::sqrt(31 * i);
        uint32_t obtained = MEMBER_ACCESS(t, x0, i);
        CheckResult(state, expected, obtained, "i == " + std::to_string(i));
    }

    state.counters["n_elem"] = n;
}

#endif // BENCHMARKS_STRIDED_H
