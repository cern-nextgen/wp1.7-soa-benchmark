#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <benchmark/benchmark.h>

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

/* SoA-like access */
// #ifdef SOA_BOOST
// #define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ.MEMBER(INDEX)
// #elif defined(SOA_MANUAL)
// #define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ.MEMBER[INDEX]
// #elif defined(AOS_MANUAL)
// #define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ[INDEX].MEMBER
// #else
// #define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ.MEMBER[INDEX]
// #endif

using Vector3D = Eigen::Vector3d;
using Matrix3D = Eigen::Matrix3d;

constexpr std::size_t N[] = {10, 100, 1000, 10000, 100000};
constexpr std::size_t N_Large[] = {10000, 100000, 1000000, 10000000, 100000000};
constexpr size_t Alignment = 128;

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


template <typename T>
static std::string ToString(const T &obj)
{
    std::stringstream ss;
    ss << obj;
    return ss.str();
}

// Helper function to check the result
template <typename Expected, typename Actual>
void CheckResult(benchmark::State &state, const Expected &expected, const Actual &actual,
                 const std::string &member_name)
{
    if (expected != actual) {
        state.SkipWithError(
            std::format("Wrong result in {}: expected {}, got {}", member_name, ToString(expected), ToString(actual)));
    }
}

template <typename S, typename N> class Fixture1; // Forward Declaration

// 2 data members, integers, 10 elements
BENCHMARK_TEMPLATE_METHOD_F(Fixture1, BM_CPUEasyRW)(benchmark::State& state) {
    auto n = this->n;
    auto &t = this->t;

    // Initialize the data members to zero
    for (size_t i = 0; i < n; ++i) {
        MEMBER_ACCESS(t, x0, i) = 0;
        MEMBER_ACCESS(t, x1, i) = 0;
    }

    // Perform read and write operations
    for (auto _ : state) {
        for (size_t i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x0, i) += 2;
            MEMBER_ACCESS(t, x1, i) += 2;
        }
    }

    // Check the result
    for (size_t i = 0; i < n; ++i) {
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x0, i), "x0");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x1, i), "x1");
    }

    state.counters["n_elem"] = n;
}


// 2 data members, integers, 10 elements
BENCHMARK_TEMPLATE_METHOD_F(Fixture1, BM_CPUEasyCompute)(benchmark::State& state) {
    auto n = this->n;
    auto &t = this->t;

    // Initialize the data members to zero
    for (int i = 0; i < n; ++i) {
        MEMBER_ACCESS(t, x0, i) = 0;
        MEMBER_ACCESS(t, x1, i) = 0;
    }

    // Perform read and write operations
    for (auto _ : state) {
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

    // Check the result
    int expected = 0;
    for (int i = 0; i < state.iterations(); ++i) {
        expected = std::sin(expected) + std::cos(expected) * std::tan(expected) +
                   std::sqrt(std::abs(expected)) * 15.0f - std::exp(expected) * 2.0f;
    }
    for (int i = 0; i < n; ++i) {
        CheckResult(state, expected, MEMBER_ACCESS(t, x0, i), "x0");
        CheckResult(state, expected, MEMBER_ACCESS(t, x1, i), "x1");
    }

    state.counters["n_elem"] = n;
}

// “Realistic case”:
//      10 data members (3 doubles, 3 float, 2 integer, 1 Vector3D, 1 Matrix)
BENCHMARK_TEMPLATE_METHOD_F(Fixture1, BM_CPURealRW)(benchmark::State& state) {
    auto n = this->n;
    auto &t = this->t;

    Matrix3D m = Matrix3D::Constant(2);
    Vector3D v = Vector3D::Constant(2);

    // Initialize the data members to zero
    for (int i = 0; i < n; ++i) {
        MEMBER_ACCESS(t, x0, i) = 0;
        MEMBER_ACCESS(t, x1, i) = 0;
        MEMBER_ACCESS(t, x2, i) = 0;
        MEMBER_ACCESS(t, x3, i) = 0;
        MEMBER_ACCESS(t, x4, i) = 0;
        MEMBER_ACCESS(t, x5, i) = 0;
        MEMBER_ACCESS(t, x6, i) = Vector3D::Zero();
        MEMBER_ACCESS(t, x7, i) = Vector3D::Zero();
        MEMBER_ACCESS(t, x8, i) = Matrix3D::Zero();
        MEMBER_ACCESS(t, x9, i) = Matrix3D::Zero();
    }

    // Perform read and write operations
    for (auto _ : state) {
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x0, i) += 2.f;
        }
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x1, i) += 2.f;
        }
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x2, i) += 2.0;
        }
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x3, i) += 2.0;
        }
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x4, i) += 2;
        }
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x5, i) += 2;
        }
#pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x6, i) += v;
        }
#pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x7, i) += v;
        }
#pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x8, i) += m;
        }
#pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x9, i) += m;
        }
    }

    // Check the result
    for (int i = 0; i < n; ++i) {
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x0, i), "x0");
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x1, i), "x1");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x2, i), "x2");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x3, i), "x3");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x4, i), "x4");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x5, i), "x5");
        CheckResult(state, v * state.iterations(), MEMBER_ACCESS(t, x6, i), "x6");
        CheckResult(state, v * state.iterations(), MEMBER_ACCESS(t, x7, i), "x7");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x8, i), "x8");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x9, i), "x9");
    }

    state.counters["n_elem"] = n;
}

// 100 data members (20 floats, 20 doubles, 20 integers, 20 Eigen vector, 20 Eigen matrices)
BENCHMARK_TEMPLATE_METHOD_F(Fixture1, BM_CPUHardRW)(benchmark::State& state) {
    auto n = this->n;
    auto &t = this->t;

    Matrix3D m = Matrix3D::Constant(2);
    Vector3D v = Vector3D::Constant(2);

    // clang-format off
    // Initialize the data members to zero
    for (int i = 0; i < n; ++i) {
        MEMBER_ACCESS(t, x0, i) = 0.f;  MEMBER_ACCESS(t, x1, i) = 0.f;  MEMBER_ACCESS(t, x2, i) = 0.f;
        MEMBER_ACCESS(t, x3, i) = 0.f;  MEMBER_ACCESS(t, x4, i) = 0.f;  MEMBER_ACCESS(t, x5, i) = 0.f;
        MEMBER_ACCESS(t, x6, i) = 0.f;  MEMBER_ACCESS(t, x7, i) = 0.f;  MEMBER_ACCESS(t, x8, i) = 0.f;
        MEMBER_ACCESS(t, x9, i) = 0.f;  MEMBER_ACCESS(t, x10, i) = 0.f; MEMBER_ACCESS(t, x11, i) = 0.f;
        MEMBER_ACCESS(t, x12, i) = 0.f;
        MEMBER_ACCESS(t, x13, i) = 0.;  MEMBER_ACCESS(t, x14, i) = 0.;  MEMBER_ACCESS(t, x15, i) = 0.;
        MEMBER_ACCESS(t, x16, i) = 0.;  MEMBER_ACCESS(t, x17, i) = 0.;  MEMBER_ACCESS(t, x18, i) = 0.;
        MEMBER_ACCESS(t, x19, i) = 0.;  MEMBER_ACCESS(t, x20, i) = 0.;  MEMBER_ACCESS(t, x21, i) = 0.;
        MEMBER_ACCESS(t, x22, i) = 0.;  MEMBER_ACCESS(t, x23, i) = 0.;  MEMBER_ACCESS(t, x24, i) = 0.;
        MEMBER_ACCESS(t, x25, i) = 0.;
        MEMBER_ACCESS(t, x26, i) = 0;   MEMBER_ACCESS(t, x27, i) = 0;   MEMBER_ACCESS(t, x28, i) = 0;
        MEMBER_ACCESS(t, x29, i) = 0;   MEMBER_ACCESS(t, x30, i) = 0;   MEMBER_ACCESS(t, x31, i) = 0;
        MEMBER_ACCESS(t, x32, i) = 0;   MEMBER_ACCESS(t, x33, i) = 0;   MEMBER_ACCESS(t, x34, i) = 0;
        MEMBER_ACCESS(t, x35, i) = 0;   MEMBER_ACCESS(t, x36, i) = 0;   MEMBER_ACCESS(t, x37, i) = 0;
        MEMBER_ACCESS(t, x38, i) = 0;
        MEMBER_ACCESS(t, x39, i) = Vector3D::Zero(); MEMBER_ACCESS(t, x40, i) = Vector3D::Zero();
        MEMBER_ACCESS(t, x41, i) = Vector3D::Zero(); MEMBER_ACCESS(t, x42, i) = Vector3D::Zero();
        MEMBER_ACCESS(t, x43, i) = Vector3D::Zero(); MEMBER_ACCESS(t, x44, i) = Vector3D::Zero();
        MEMBER_ACCESS(t, x45, i) = Vector3D::Zero(); MEMBER_ACCESS(t, x46, i) = Vector3D::Zero();
        MEMBER_ACCESS(t, x47, i) = Vector3D::Zero(); MEMBER_ACCESS(t, x48, i) = Vector3D::Zero();
        MEMBER_ACCESS(t, x49, i) = Vector3D::Zero(); MEMBER_ACCESS(t, x50, i) = Vector3D::Zero();
        MEMBER_ACCESS(t, x51, i) = Matrix3D::Zero(); MEMBER_ACCESS(t, x52, i) = Matrix3D::Zero();
        MEMBER_ACCESS(t, x53, i) = Matrix3D::Zero(); MEMBER_ACCESS(t, x54, i) = Matrix3D::Zero();
        MEMBER_ACCESS(t, x55, i) = Matrix3D::Zero(); MEMBER_ACCESS(t, x56, i) = Matrix3D::Zero();
        MEMBER_ACCESS(t, x57, i) = Matrix3D::Zero(); MEMBER_ACCESS(t, x58, i) = Matrix3D::Zero();
        MEMBER_ACCESS(t, x59, i) = Matrix3D::Zero(); MEMBER_ACCESS(t, x60, i) = Matrix3D::Zero();
        MEMBER_ACCESS(t, x61, i) = Matrix3D::Zero();MEMBER_ACCESS(t, x62, i) = Matrix3D::Zero();
        MEMBER_ACCESS(t, x63, i) = Matrix3D::Zero();
    }

    // Perform read and write operations
    for (auto _ : state) {
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x0, i) += 2.f;  MEMBER_ACCESS(t, x1, i) += 2.f;  MEMBER_ACCESS(t, x2, i) += 2.f;
            MEMBER_ACCESS(t, x3, i) += 2.f;  MEMBER_ACCESS(t, x4, i) += 2.f;
        }
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x5, i) += 2.f;  MEMBER_ACCESS(t, x6, i) += 2.f;  MEMBER_ACCESS(t, x7, i) += 2.f;
            MEMBER_ACCESS(t, x8, i) += 2.f;  MEMBER_ACCESS(t, x9, i) += 2.f;
        }
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x10, i) += 2.f; MEMBER_ACCESS(t, x11, i) += 2.f; MEMBER_ACCESS(t, x12, i) += 2.f;
        }
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x13, i) += 2.0; MEMBER_ACCESS(t, x14, i) += 2.0; MEMBER_ACCESS(t, x15, i) += 2.0;
            MEMBER_ACCESS(t, x16, i) += 2.0; MEMBER_ACCESS(t, x17, i) += 2.0;
        }
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x18, i) += 2.0; MEMBER_ACCESS(t, x19, i) += 2.0; MEMBER_ACCESS(t, x20, i) += 2.0;
            MEMBER_ACCESS(t, x21, i) += 2.0; MEMBER_ACCESS(t, x22, i) += 2.0;
        }
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x23, i) += 2.0; MEMBER_ACCESS(t, x24, i) += 2.0; MEMBER_ACCESS(t, x25, i) += 2.0;
        }
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x26, i) += 2;   MEMBER_ACCESS(t, x27, i) += 2;   MEMBER_ACCESS(t, x28, i) += 2;
            MEMBER_ACCESS(t, x29, i) += 2;   MEMBER_ACCESS(t, x30, i) += 2;
        }
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x31, i) += 2;   MEMBER_ACCESS(t, x32, i) += 2;   MEMBER_ACCESS(t, x33, i) += 2;
            MEMBER_ACCESS(t, x34, i) += 2;   MEMBER_ACCESS(t, x35, i) += 2;
        }
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x36, i) += 2;   MEMBER_ACCESS(t, x37, i) += 2;   MEMBER_ACCESS(t, x38, i) += 2;
        }
        #pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x39, i) += v;   MEMBER_ACCESS(t, x40, i) += v;   MEMBER_ACCESS(t, x41, i) += v;
            MEMBER_ACCESS(t, x42, i) += v;   MEMBER_ACCESS(t, x43, i) += v;
        }
        #pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x44, i) += v;   MEMBER_ACCESS(t, x45, i) += v;   MEMBER_ACCESS(t, x46, i) += v;
            MEMBER_ACCESS(t, x47, i) += v;   MEMBER_ACCESS(t, x48, i) += v;
        }
        #pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x49, i) += v;   MEMBER_ACCESS(t, x50, i) += v;   MEMBER_ACCESS(t, x51, i) += m;
            MEMBER_ACCESS(t, x52, i) += m;   MEMBER_ACCESS(t, x53, i) += m;
        }
        #pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x54, i) += m;   MEMBER_ACCESS(t, x55, i) += m;   MEMBER_ACCESS(t, x56, i) += m;
            MEMBER_ACCESS(t, x57, i) += m;   MEMBER_ACCESS(t, x58, i) += m;
        }
        #pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x59, i) += m;   MEMBER_ACCESS(t, x60, i) += m;   MEMBER_ACCESS(t, x61, i) += m;
            MEMBER_ACCESS(t, x62, i) += m;   MEMBER_ACCESS(t, x63, i) += m;
        }
    }

    // Check the result
    for (int i = 0; i < n; ++i) {
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x0, i), "x0");
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x1, i), "x1");
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x2, i), "x2");
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x3, i), "x3");
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x4, i), "x4");
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x5, i), "x5");
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x6, i), "x6");
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x7, i), "x7");
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x8, i), "x8");
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x9, i), "x9");
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x10, i), "x10");
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x11, i), "x11");
        CheckResult(state, 2.f * state.iterations(), MEMBER_ACCESS(t, x12, i), "x12");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x13, i), "x13");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x14, i), "x14");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x15, i), "x15");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x16, i), "x16");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x17, i), "x17");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x18, i), "x18");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x19, i), "x19");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x20, i), "x20");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x21, i), "x21");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x22, i), "x22");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x23, i), "x23");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x24, i), "x24");
        CheckResult(state, 2. * state.iterations(), MEMBER_ACCESS(t, x25, i), "x25");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x26, i), "x26");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x27, i), "x27");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x28, i), "x28");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x29, i), "x29");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x30, i), "x30");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x31, i), "x31");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x32, i), "x32");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x33, i), "x33");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x34, i), "x34");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x35, i), "x35");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x36, i), "x36");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x37, i), "x37");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x38, i), "x38");
        CheckResult(state, v * state.iterations(), MEMBER_ACCESS(t, x39, i), "x39");
        CheckResult(state, v * state.iterations(), MEMBER_ACCESS(t, x40, i), "x40");
        CheckResult(state, v * state.iterations(), MEMBER_ACCESS(t, x41, i), "x41");
        CheckResult(state, v * state.iterations(), MEMBER_ACCESS(t, x42, i), "x42");
        CheckResult(state, v * state.iterations(), MEMBER_ACCESS(t, x43, i), "x43");
        CheckResult(state, v * state.iterations(), MEMBER_ACCESS(t, x44, i), "x44");
        CheckResult(state, v * state.iterations(), MEMBER_ACCESS(t, x45, i), "x45");
        CheckResult(state, v * state.iterations(), MEMBER_ACCESS(t, x46, i), "x46");
        CheckResult(state, v * state.iterations(), MEMBER_ACCESS(t, x47, i), "x47");
        CheckResult(state, v * state.iterations(), MEMBER_ACCESS(t, x48, i), "x48");
        CheckResult(state, v * state.iterations(), MEMBER_ACCESS(t, x49, i), "x49");
        CheckResult(state, v * state.iterations(), MEMBER_ACCESS(t, x50, i), "x50");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x51, i), "x51");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x52, i), "x52");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x53, i), "x53");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x54, i), "x54");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x55, i), "x55");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x56, i), "x56");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x57, i), "x57");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x58, i), "x58");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x59, i), "x59");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x60, i), "x60");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x61, i), "x61");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x62, i), "x62");
        CheckResult(state, m * state.iterations(), MEMBER_ACCESS(t, x63, i), "x63");
    }
    // clang-format on

    state.counters["n_elem"] = n;
}

inline float rand_float()
{
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, BM_nbody)(benchmark::State& state) {
    auto n = this->n;
    auto &t = this->t;
    float dt = 0.01f;
    const float softening = 1e-9f;

    std::vector<float> Fx(n, 0.0f);
    std::vector<float> Fy(n, 0.0f);
    std::vector<float> Fz(n, 0.0f);

    // Inizializza le posizioni e velocità
    for (size_t i = 0; i < n; ++i) {
        MEMBER_ACCESS(t, x, i) = rand_float();
        MEMBER_ACCESS(t, y, i) = rand_float();
        MEMBER_ACCESS(t, z, i) = rand_float();
        MEMBER_ACCESS(t, vx, i) = rand_float();
        MEMBER_ACCESS(t, vy, i) = rand_float();
        MEMBER_ACCESS(t, vz, i) = rand_float();
    }

    for (auto _ : state) {
        // Calcolo delle forze
        for (size_t i = 0; i < n; ++i) {
            Fx[i] = 0.0f;
            Fy[i] = 0.0f;
            Fz[i] = 0.0f;

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

        // Integrazione posizioni
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

// Exact solution (for comparison)
inline double solution_poisson(const double x)
{
    return x * (x - 1) * std::exp(x);
}

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, BM_stencil)(benchmark::State& state) {
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

template<typename T1, typename T2, typename N> class Fixture2; // forward declaration

BENCHMARK_TEMPLATE_METHOD_F(Fixture2, BM_InvariantMass)(benchmark::State& state) {
    auto n = this->n;
    auto &v1 = this->t1;
    auto &v2 = this->t2;

    // Initialise x,y,z,M vectors
    for (size_t i = 0; i < n; ++i) {
        MEMBER_ACCESS(v1, x, i) = i * 5;
        MEMBER_ACCESS(v1, y, i) = i * 3;
        MEMBER_ACCESS(v1, z, i) = i;
        MEMBER_ACCESS(v1, M, i) = i / 2.;
        MEMBER_ACCESS(v2, x, i) = i;
        MEMBER_ACCESS(v2, y, i) = i * 20;
        MEMBER_ACCESS(v2, z, i) = i / 9.;
        MEMBER_ACCESS(v2, M, i) = i * 42;
    }

    std::vector<double> results(n);
    for (auto _ : state) {
        for (size_t i = 0; i < n; ++i) {
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

            const auto cx =
                MEMBER_ACCESS(v1, y, i) * MEMBER_ACCESS(v2, z, i) - MEMBER_ACCESS(v2, y, i) * MEMBER_ACCESS(v1, z, i);
            const auto cy =
                MEMBER_ACCESS(v1, x, i) * MEMBER_ACCESS(v2, z, i) - MEMBER_ACCESS(v2, x, i) * MEMBER_ACCESS(v1, z, i);
            const auto cz =
                MEMBER_ACCESS(v1, x, i) * MEMBER_ACCESS(v2, y, i) - MEMBER_ACCESS(v2, x, i) * MEMBER_ACCESS(v1, y, i);

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

    state.counters["n_elem"] = static_cast<double>(n);
}

#endif // BENCHMARK_H
