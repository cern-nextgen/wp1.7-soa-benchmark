#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <benchmark/benchmark.h>
#include <math.h>
#include <sstream>
#include <format>

/* AoS-like access (except for the baseline) */
#ifdef SOA_BOOST
#define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ[INDEX].MEMBER()
#elif defined(SOA_MANUAL)
#define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ.MEMBER[INDEX]
#elif defined(AOS_MANUAL)
#define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ[INDEX].MEMBER
#else
#define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ[INDEX].MEMBER
#endif

constexpr size_t Alignment = 128;

constexpr size_t N_im = 10000000;
constexpr size_t N_stencil = 10000000;
constexpr size_t N_nbody = 10000;

// clang-format off
#define INSTANTIATE_BENCHMARKS_F1(BM, Type, N) \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM, Type, std::integral_constant<size_t, N[0]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM, Type, std::integral_constant<size_t, N[1]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM, Type, std::integral_constant<size_t, N[2]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM, Type, std::integral_constant<size_t, N[3]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM, Type, std::integral_constant<size_t, N[4]>)->Unit(benchmark::kMillisecond);

#define INSTANTIATE_BENCHMARKS_F2(BM, Type1, Type2, N) \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, BM, Type1, Type2, std::integral_constant<size_t, N[0]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, BM, Type1, Type2, std::integral_constant<size_t, N[1]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, BM, Type1, Type2, std::integral_constant<size_t, N[2]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, BM, Type1, Type2, std::integral_constant<size_t, N[3]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, BM, Type1, Type2, std::integral_constant<size_t, N[4]>)->Unit(benchmark::kMillisecond);
// clang-format on

template <typename S, typename N>
class Fixture1; // Forward Declaration

inline float rand_float()
{
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, BM_nbody)(benchmark::State &state)
{
    auto n = this->n;
    auto &t = this->t;
    float dt = 0.01f;
    const float softening = 1e-9f;

    std::vector<float> Fx(n, 0.0f);
    std::vector<float> Fy(n, 0.0f);
    std::vector<float> Fz(n, 0.0f);

    // Initialize positions and velocities
    for (size_t i = 0; i < n; ++i) {
        MEMBER_ACCESS(t, x, i) = rand_float();
        MEMBER_ACCESS(t, y, i) = rand_float();
        MEMBER_ACCESS(t, z, i) = rand_float();
        MEMBER_ACCESS(t, vx, i) = rand_float();
        MEMBER_ACCESS(t, vy, i) = rand_float();
        MEMBER_ACCESS(t, vz, i) = rand_float();
    }

    for (auto _ : state) {
        // Calculate the force
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

        // Integrate positions
        for (size_t i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x, i) += MEMBER_ACCESS(t, vx, i) * dt;
            MEMBER_ACCESS(t, y, i) += MEMBER_ACCESS(t, vy, i) * dt;
            MEMBER_ACCESS(t, z, i) += MEMBER_ACCESS(t, vz, i) * dt;
        }

        // benchmark::ClobberMemory();
    }

    state.counters["n_elem"] = n;
    state.counters["N^2_interactions"] =
        benchmark::Counter(static_cast<double>(n) * static_cast<double>(n), benchmark::Counter::kIsRate);
}

// Exact solution (for comparison)
inline double solution_poisson(const double x)
{
    return x * (x - 1) * std::exp(x);
}

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, BM_stencil)(benchmark::State &state)
{
    auto n = this->n;
    auto &t = this->t;
    // Domain: [0, L]
    constexpr double L = 1.0;
    const double dx = L / static_cast<double>(n - 1);
    const double dx2 = dx * dx;

    // Initialise the domain
    for (size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i) * dx;
        MEMBER_ACCESS(t, src, i) = 0.0;
        MEMBER_ACCESS(t, dst, i) = 0.0;
        MEMBER_ACCESS(t, rhs, i) = -x * (x + 3) * std::exp(x);
    }

    for (auto _ : state) {
        for (size_t i = 1; i < n - 1; ++i) {
            const double u_left = MEMBER_ACCESS(t, src, i - 1);
            const double u_right = MEMBER_ACCESS(t, src, i + 1);
            const double f = MEMBER_ACCESS(t, rhs, i);
            MEMBER_ACCESS(t, dst, i) = 0.5 * (u_left + u_right + dx2 * f);
        }

        // Maybe use a pointer swap instead
        for (size_t i = 1; i < n - 1; ++i) {
            const double u_left = MEMBER_ACCESS(t, dst, i - 1);
            const double u_right = MEMBER_ACCESS(t, dst, i + 1);
            const double f = MEMBER_ACCESS(t, rhs, i);
            MEMBER_ACCESS(t, src, i) = 0.5 * (u_left + u_right + dx2 * f);
        }
    }

    /*
    for (int i = 1; i < n; ++i) {
        const double x = static_cast<double>(i) * dx;
        const double u_ex = solution_poisson(x);
        CheckResult(state, u_ex, MEMBER_ACCESS(t, src, i), "src");
    }
    */

    state.counters["n_elem"] = static_cast<double>(n);
    state.counters["N^2_interactions"] = benchmark::Counter(static_cast<double>(n) * 2.0, benchmark::Counter::kIsRate);
}

template <typename T1, typename T2, typename N>
class Fixture2; // forward declaration

inline double rand_double()
{
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

BENCHMARK_TEMPLATE_METHOD_F(Fixture2, BM_InvariantMass)(benchmark::State &state)
{
    auto n = this->n;
    auto &v1 = this->t1;
    auto &v2 = this->t2;

    // Initialise x,y,z,M vectors
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
    size_t stride = 1;
    for (auto _ : state) {
#pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < n; i += stride) {
                // Numerically stable computation of Invariant Masses
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
                const auto x = r1 + r2 + r1 * r2;

                const auto cx = MEMBER_ACCESS(v1, y, i) * MEMBER_ACCESS(v2, z, i) -
                                MEMBER_ACCESS(v2, y, i) * MEMBER_ACCESS(v1, z, i);
                const auto cy = MEMBER_ACCESS(v1, x, i) * MEMBER_ACCESS(v2, z, i) -
                                MEMBER_ACCESS(v2, x, i) * MEMBER_ACCESS(v1, z, i);
                const auto cz = MEMBER_ACCESS(v1, x, i) * MEMBER_ACCESS(v2, y, i) -
                                MEMBER_ACCESS(v2, x, i) * MEMBER_ACCESS(v1, y, i);

                // norm of cross product
                const auto c = std::sqrt(cx * cx + cy * cy + cz * cz);

                // dot product
                const auto d = MEMBER_ACCESS(v1, x, i) * MEMBER_ACCESS(v2, x, i) +
                               MEMBER_ACCESS(v1, y, i) * MEMBER_ACCESS(v2, y, i) +
                               MEMBER_ACCESS(v1, z, i) * MEMBER_ACCESS(v2, z, i);

                const auto a = std::atan2(c, d);

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
    }

    for (size_t i = 0; i < n; i++) {
        benchmark::DoNotOptimize(results[i]);
    }

    state.counters["n_elem"] = static_cast<double>(n);
    state.counters["stride"] = static_cast<double>(stride);
}

#endif // BENCHMARK_H
