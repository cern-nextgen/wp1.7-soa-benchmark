#pragma once
#include "benchmarks/common.h"

inline double rand_double()
{
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

BENCHMARK_TEMPLATE_METHOD_F(Fixture2, InvariantMass)(benchmark::State &state)
{
    auto n = this->n;
    auto &v1 = this->t1;
    auto &v2 = this->t2;

    // v1 and v2 are read-only inside the loop — initialise once
    for (size_t i = 0; i < n; ++i) {
        MEMBER_ACCESS(v1, x, i) = rand_double();
        MEMBER_ACCESS(v1, y, i) = rand_double();
        MEMBER_ACCESS(v1, z, i) = rand_double();
        MEMBER_ACCESS(v1, M, i) = rand_double();
        MEMBER_ACCESS(v2, x, i) = rand_double();
        MEMBER_ACCESS(v2, y, i) = rand_double();
        MEMBER_ACCESS(v2, z, i) = rand_double();
        MEMBER_ACCESS(v2, M, i) = rand_double();
    }

    std::vector<double> results(n);
    for (auto _ : state) {
        #pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < n; ++i) {
            const auto p1_sq = MEMBER_ACCESS(v1, x, i) * MEMBER_ACCESS(v1, x, i) +
                               MEMBER_ACCESS(v1, y, i) * MEMBER_ACCESS(v1, y, i) +
                               MEMBER_ACCESS(v1, z, i) * MEMBER_ACCESS(v1, z, i);
            const auto p2_sq = MEMBER_ACCESS(v2, x, i) * MEMBER_ACCESS(v2, x, i) +
                               MEMBER_ACCESS(v2, y, i) * MEMBER_ACCESS(v2, y, i) +
                               MEMBER_ACCESS(v2, z, i) * MEMBER_ACCESS(v2, z, i);

            const auto m1_sq = MEMBER_ACCESS(v1, M, i) * MEMBER_ACCESS(v1, M, i);
            const auto m2_sq = MEMBER_ACCESS(v2, M, i) * MEMBER_ACCESS(v2, M, i);

            const auto r1 = m1_sq / p1_sq;
            const auto r2 = m2_sq / p2_sq;
            const auto x  = r1 + r2 + r1 * r2;

            const auto cx = MEMBER_ACCESS(v1, y, i) * MEMBER_ACCESS(v2, z, i) -
                            MEMBER_ACCESS(v2, y, i) * MEMBER_ACCESS(v1, z, i);
            const auto cy = MEMBER_ACCESS(v1, x, i) * MEMBER_ACCESS(v2, z, i) -
                            MEMBER_ACCESS(v2, x, i) * MEMBER_ACCESS(v1, z, i);
            const auto cz = MEMBER_ACCESS(v1, x, i) * MEMBER_ACCESS(v2, y, i) -
                            MEMBER_ACCESS(v2, x, i) * MEMBER_ACCESS(v1, y, i);

            const auto c = std::sqrt(cx * cx + cy * cy + cz * cz);
            const auto d = MEMBER_ACCESS(v1, x, i) * MEMBER_ACCESS(v2, x, i) +
                           MEMBER_ACCESS(v1, y, i) * MEMBER_ACCESS(v2, y, i) +
                           MEMBER_ACCESS(v1, z, i) * MEMBER_ACCESS(v2, z, i);
            const auto a     = std::atan2(c, d);
            const auto cos_a = std::cos(a);
            auto y = x;
            if (cos_a >= 0) {
                y = (x + std::sin(a) * std::sin(a)) / (std::sqrt(x + 1) + cos_a);
            } else {
                y = std::sqrt(x + 1) - cos_a;
            }

            const auto z = 2 * std::sqrt(p1_sq * p2_sq);
            results[i] = std::sqrt(m1_sq + m2_sq + y * z);
        }

        for (size_t i = 0; i < n; ++i) {
            benchmark::DoNotOptimize(results[i]);
        }
    }

    state.counters["n_elem"] = static_cast<double>(n);
}
