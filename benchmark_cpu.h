#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <string>

#include <Eigen/Core>
#include <benchmark/benchmark.h>

// No AoS access for manual version
#ifdef SOA_MANUAL
#define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ.MEMBER[INDEX]
#else
#define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ[INDEX].MEMBER
#endif

using Vector3D = Eigen::Vector3d;
using Matrix3D = Eigen::Matrix3d;

constexpr std::size_t N[] = {10, 100, 1000, 10000, 100000};

// Helper function to check the result
template <typename Expected, typename Actual>
void CheckResult(benchmark::State &state, const Expected &expected, const Actual &actual,
                 const std::string &member_name)
{
    if (expected != actual) {
        std::stringstream ss;
        ss << "Wrong result in " << member_name
           << ": expected " << expected
           << ", got " << actual;
        state.SkipWithError(ss.str());
    }
}

// 2 data members, integers, 10 elements
template <typename T>
void BM_CPUEasyRW(benchmark::State &state, T t)
{
    auto n = state.range(0);

    // Initialize the data members to zero
    for (int i = 0; i < n; ++i) {
        MEMBER_ACCESS(t, x0, i) = 0;
        MEMBER_ACCESS(t, x1, i) = 0;
    }

    // Perform read and write operations
    for (auto _ : state) {
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x0, i) += 2;
            MEMBER_ACCESS(t, x1, i) += 2;
        }
    }

    // Check the result
    for (int i = 0; i < n; ++i) {
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x0, i), "x0");
        CheckResult(state, 2 * state.iterations(), MEMBER_ACCESS(t, x1, i), "x1");
    }

    state.counters["n_elem"] = n;
}

// 2 data members, integers, 10 elements
template <typename T>
void BM_CPUEasyCompute(benchmark::State &state, T t)
{
    auto n = state.range(0);

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
template <typename T>
void BM_CPURealRW(benchmark::State &state, T t)
{
    auto n = state.range(0);

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
template <typename T>
void BM_CPUHardRW(benchmark::State &state, T t)
{
    auto n = state.range(0);

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

    state.counters["n_elem"] = n;
}

#endif // BENCHMARK_H
