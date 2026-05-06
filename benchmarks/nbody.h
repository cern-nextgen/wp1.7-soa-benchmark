#ifndef BENCHMARKS_NBODY_H
#define BENCHMARKS_NBODY_H

#include "benchmarks/common.h"

#include <cmath>

template <Backend B, class T>
void run_NBody(benchmark::State &state, std::size_t n, T t)
{
    const float dt = 0.01f;
    const float softening = 1e-9f;

    for (auto _ : state) {
        state.PauseTiming();
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x,  i) = static_cast<float>(i) / static_cast<float>(n);
            MEMBER_ACCESS(t, y,  i) = static_cast<float>((i + n / 3) % n) / static_cast<float>(n);
            MEMBER_ACCESS(t, z,  i) = static_cast<float>((i + 2 * n / 3) % n) / static_cast<float>(n);
            MEMBER_ACCESS(t, vx, i) = 0.0f;
            MEMBER_ACCESS(t, vy, i) = 0.0f;
            MEMBER_ACCESS(t, vz, i) = 0.0f;
        });
        backend_allocator<B>::synchronize();
        state.ResumeTiming();

        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            float Fx = 0.0f;
            float Fy = 0.0f;
            float Fz = 0.0f;
            for (std::size_t j = 0; j < n; ++j) {
                if (i != j) {
                    float dx = MEMBER_ACCESS(t, x, j) - MEMBER_ACCESS(t, x, i);
                    float dy = MEMBER_ACCESS(t, y, j) - MEMBER_ACCESS(t, y, i);
                    float dz = MEMBER_ACCESS(t, z, j) - MEMBER_ACCESS(t, z, i);
                    float distSqr = dx * dx + dy * dy + dz * dz + softening;
                    float invDist = 1.0f / std::sqrt(distSqr);
                    float invDist3 = invDist * invDist * invDist;

                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            }

            MEMBER_ACCESS(t, vx, i) += dt * Fx;
            MEMBER_ACCESS(t, vy, i) += dt * Fy;
            MEMBER_ACCESS(t, vz, i) += dt * Fz;
        });
        backend_allocator<B>::synchronize();

        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x, i) += MEMBER_ACCESS(t, vx, i) * dt;
            MEMBER_ACCESS(t, y, i) += MEMBER_ACCESS(t, vy, i) * dt;
            MEMBER_ACCESS(t, z, i) += MEMBER_ACCESS(t, vz, i) * dt;
        });
    }

    backend_allocator<B>::synchronize();
    state.counters["n_elem"] = n;
    state.counters["N^2_interactions"] =
        benchmark::Counter(static_cast<double>(n) * static_cast<double>(n), benchmark::Counter::kIsRate);
}

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, NBody)(benchmark::State &state)
{
    constexpr Backend B = std::remove_reference_t<decltype(*this)>::backend;
    run_NBody<B>(state, this->n, this->t);
}

#endif // BENCHMARKS_NBODY_H
