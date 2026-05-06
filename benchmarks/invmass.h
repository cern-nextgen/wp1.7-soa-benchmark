#ifndef BENCHMARKS_INVMASS_H
#define BENCHMARKS_INVMASS_H

#include "benchmarks/common.h"

#include <cmath>

inline double rand_double()
{
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

template <Backend B, class T1, class T2>
void run_InvariantMass(benchmark::State &state, std::size_t n, T1 v1, T2 v2)
{
    for (std::size_t i = 0; i < n; ++i) {
        MEMBER_ACCESS(v1, x, i) = rand_double();
        MEMBER_ACCESS(v1, y, i) = rand_double();
        MEMBER_ACCESS(v1, z, i) = rand_double();
        MEMBER_ACCESS(v1, M, i) = rand_double();
        MEMBER_ACCESS(v2, x, i) = rand_double();
        MEMBER_ACCESS(v2, y, i) = rand_double();
        MEMBER_ACCESS(v2, z, i) = rand_double();
        MEMBER_ACCESS(v2, M, i) = rand_double();
    }
    backend_allocator<B>::synchronize();

    double *results = backend_allocator<B>::template alloc<double>(n);

    for (auto _ : state) {
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            const double p1_sq = MEMBER_ACCESS(v1, x, i) * MEMBER_ACCESS(v1, x, i) +
                                 MEMBER_ACCESS(v1, y, i) * MEMBER_ACCESS(v1, y, i) +
                                 MEMBER_ACCESS(v1, z, i) * MEMBER_ACCESS(v1, z, i);
            const double p2_sq = MEMBER_ACCESS(v2, x, i) * MEMBER_ACCESS(v2, x, i) +
                                 MEMBER_ACCESS(v2, y, i) * MEMBER_ACCESS(v2, y, i) +
                                 MEMBER_ACCESS(v2, z, i) * MEMBER_ACCESS(v2, z, i);

            const double m1_sq = MEMBER_ACCESS(v1, M, i) * MEMBER_ACCESS(v1, M, i);
            const double m2_sq = MEMBER_ACCESS(v2, M, i) * MEMBER_ACCESS(v2, M, i);

            const double r1 = m1_sq / p1_sq;
            const double r2 = m2_sq / p2_sq;
            const double x  = r1 + r2 + r1 * r2;

            const double cx = MEMBER_ACCESS(v1, y, i) * MEMBER_ACCESS(v2, z, i) -
                              MEMBER_ACCESS(v2, y, i) * MEMBER_ACCESS(v1, z, i);
            const double cy = MEMBER_ACCESS(v1, x, i) * MEMBER_ACCESS(v2, z, i) -
                              MEMBER_ACCESS(v2, x, i) * MEMBER_ACCESS(v1, z, i);
            const double cz = MEMBER_ACCESS(v1, x, i) * MEMBER_ACCESS(v2, y, i) -
                              MEMBER_ACCESS(v2, x, i) * MEMBER_ACCESS(v1, y, i);

            const double c = std::sqrt(cx * cx + cy * cy + cz * cz);
            const double d = MEMBER_ACCESS(v1, x, i) * MEMBER_ACCESS(v2, x, i) +
                             MEMBER_ACCESS(v1, y, i) * MEMBER_ACCESS(v2, y, i) +
                             MEMBER_ACCESS(v1, z, i) * MEMBER_ACCESS(v2, z, i);
            const double a     = std::atan2(c, d);
            const double cos_a = std::cos(a);
            double y;
            if (cos_a >= 0) {
                y = (x + std::sin(a) * std::sin(a)) / (std::sqrt(x + 1) + cos_a);
            } else {
                y = std::sqrt(x + 1) - cos_a;
            }

            const double z = 2 * std::sqrt(p1_sq * p2_sq);
            results[i] = std::sqrt(m1_sq + m2_sq + y * z);
        });
        backend_allocator<B>::synchronize();

        for (std::size_t i = 0; i < n; ++i) {
            benchmark::DoNotOptimize(results[i]);
        }
    }

    backend_allocator<B>::free(results);
    state.counters["n_elem"] = static_cast<double>(n);
}

BENCHMARK_TEMPLATE_METHOD_F(Fixture2, InvariantMass)(benchmark::State &state)
{
    constexpr Backend B = std::remove_reference_t<decltype(*this)>::backend;
    run_InvariantMass<B>(state, this->n, this->t1, this->t2);
}

#endif // BENCHMARKS_INVMASS_H
