#pragma once
#include "benchmarks/common.h"

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, NBody)(benchmark::State &state)
{
    auto n = this->n;
    auto &t = this->t;
    const float dt = 0.01f;
    const float softening = 1e-9f;

    std::vector<float> Fx(n);
    std::vector<float> Fy(n);
    std::vector<float> Fz(n);

    for (auto _ : state) {
        state.PauseTiming();
        for (size_t i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x,  i) = static_cast<float>(i) / static_cast<float>(n);
            MEMBER_ACCESS(t, y,  i) = static_cast<float>((i + n / 3) % n) / static_cast<float>(n);
            MEMBER_ACCESS(t, z,  i) = static_cast<float>((i + 2 * n / 3) % n) / static_cast<float>(n);
            MEMBER_ACCESS(t, vx, i) = 0.0f;
            MEMBER_ACCESS(t, vy, i) = 0.0f;
            MEMBER_ACCESS(t, vz, i) = 0.0f;
            Fx[i] = 0.0f;
            Fy[i] = 0.0f;
            Fz[i] = 0.0f;
        }
        state.ResumeTiming();

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i != j) {
                    float dx = MEMBER_ACCESS(t, x, j) - MEMBER_ACCESS(t, x, i);
                    float dy = MEMBER_ACCESS(t, y, j) - MEMBER_ACCESS(t, y, i);
                    float dz = MEMBER_ACCESS(t, z, j) - MEMBER_ACCESS(t, z, i);
                    float distSqr = dx * dx + dy * dy + dz * dz + softening;
                    float invDist = 1.0f / std::sqrt(distSqr);
                    float invDist3 = invDist * invDist * invDist;

                    Fx[i] += dx * invDist3;
                    Fy[i] += dy * invDist3;
                    Fz[i] += dz * invDist3;
                }
            }

            MEMBER_ACCESS(t, vx, i) += dt * Fx[i];
            MEMBER_ACCESS(t, vy, i) += dt * Fy[i];
            MEMBER_ACCESS(t, vz, i) += dt * Fz[i];
        }

        for (size_t i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x, i) += MEMBER_ACCESS(t, vx, i) * dt;
            MEMBER_ACCESS(t, y, i) += MEMBER_ACCESS(t, vy, i) * dt;
            MEMBER_ACCESS(t, z, i) += MEMBER_ACCESS(t, vz, i) * dt;
        }
    }

    state.counters["n_elem"] = n;
    state.counters["N^2_interactions"] =
        benchmark::Counter(static_cast<double>(n) * static_cast<double>(n), benchmark::Counter::kIsRate);
}
