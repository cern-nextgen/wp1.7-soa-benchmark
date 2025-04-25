#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <Eigen/Core>

#ifdef SOA_BOOST
    #define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ[INDEX].MEMBER()
#elif defined(SOA_MANUAL)
    #define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ.MEMBER[INDEX]
#else
    #define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ[INDEX].MEMBER
#endif

using Vector3D = Eigen::Vector3d;
using Matrix3D = Eigen::Matrix3d;

constexpr size_t N[] = {10, 100, 1000, 10000, 100000};

// 2 data members, integers, 64 alignment, 10 elements
template <typename T>
void BM_CPUEasyRW(benchmark::State &state, T t) {
    constexpr auto repetitions = 1;

    for (auto _ : state) {
        for (size_t _ = 0; _ < repetitions; ++_) {
            for (int i = 0; i < state.range(0); ++i) {
                MEMBER_ACCESS(t, x0, i) += 2;
                MEMBER_ACCESS(t, x1, i) += 2;
            }
        }
    }

    state.counters["n_elem"] = state.range(0);
    state.counters["reps"] = repetitions;
}

// 2 data members, integers, 64 alignment, 10 elements
template <typename T>
void BM_CPUEasyCompute(benchmark::State &state, T t) {
    constexpr auto repetitions = 1;

    for (auto _ : state) {
        for (size_t _ = 0; _ < repetitions; ++_) {
            for (int i = 0; i < state.range(0); ++i) {
                MEMBER_ACCESS(t, x0, i) = 1 + MEMBER_ACCESS(t, x0, i) * MEMBER_ACCESS(t, x0, i) * MEMBER_ACCESS(t, x0, i) *
                                         MEMBER_ACCESS(t, x0, i) * MEMBER_ACCESS(t, x0, i) +
                                         7 * MEMBER_ACCESS(t, x1, i) * MEMBER_ACCESS(t, x1, i) *
                                         MEMBER_ACCESS(t, x1, i) -
                                         6 * MEMBER_ACCESS(t, x0, i) * MEMBER_ACCESS(t, x1, i) *
                                         MEMBER_ACCESS(t, x1, i) +
                                         3 * MEMBER_ACCESS(t, x0, i) * MEMBER_ACCESS(t, x0, i) *
                                         MEMBER_ACCESS(t, x1, i);
                MEMBER_ACCESS(t, x1, i) = 1 + MEMBER_ACCESS(t, x1, i) * MEMBER_ACCESS(t, x1, i) * MEMBER_ACCESS(t, x1, i) *
                                         MEMBER_ACCESS(t, x1, i) -
                                         5 * MEMBER_ACCESS(t, x0, i) * MEMBER_ACCESS(t, x0, i) *
                                         MEMBER_ACCESS(t, x0, i) +
                                         4 * MEMBER_ACCESS(t, x0, i) * MEMBER_ACCESS(t, x1, i) *
                                         MEMBER_ACCESS(t, x1, i) -
                                         2 * MEMBER_ACCESS(t, x1, i) * MEMBER_ACCESS(t, x0, i) *
                                         MEMBER_ACCESS(t, x0, i);
            }
        }
    }

    state.counters["n_elem"] = state.range(0);
    state.counters["reps"] = repetitions;
}


// “Realistic case”:
//      10 data members (3 doubles, 3 float, 2 integer, 1 Vector3D, 1 Matrix), 64 alignment, 100000
template <typename T>
void BM_CPURealRW(benchmark::State &state, T t) {
    constexpr auto repetitions = 1;

    Matrix3D m;
    m << 2, 2, 2, 2, 2, 2, 2, 2, 2;
    Vector3D v(2, 2, 2);

    for (auto _ : state) {
        for (size_t _ = 0; _ < repetitions; ++_) {
            for (int i = 0; i < state.range(0); ++i) {
                MEMBER_ACCESS(t, x0, i) += 2.f;
                MEMBER_ACCESS(t, x1, i) += 2.f;
                MEMBER_ACCESS(t, x2, i) += 2.;
                MEMBER_ACCESS(t, x3, i) += 2.;
                MEMBER_ACCESS(t, x4, i) += 2;
                MEMBER_ACCESS(t, x5, i) += 2;
                MEMBER_ACCESS(t, x6, i) += v;
                MEMBER_ACCESS(t, x7, i) += v;
                MEMBER_ACCESS(t, x8, i) += m;
                MEMBER_ACCESS(t, x9, i) += m;
            }
        }
    }

    state.counters["n_elem"] = state.range(0);
    state.counters["reps"] = repetitions;
}


// 100 data members (20 floats, 20 doubles, 20 integers, 20 Eigen vector, 20 Eigen matrices),
// 64 alignment, 10e4 elements
template <typename T>
void BM_CPUHardRW(benchmark::State &state, T t) {
    constexpr auto repetitions = 1;

    Matrix3D m;
    m << 2, 2, 2, 2, 2, 2, 2, 2, 2;
    Vector3D v(2, 2, 2);

    for (auto _ : state) {
        for (size_t _ = 0; _ < repetitions; ++_) {
            for (int i = 0; i < state.range(0); ++i) {
                MEMBER_ACCESS(t, x0, i)  += 2.f;
                MEMBER_ACCESS(t, x1, i)  += 2.f;
                MEMBER_ACCESS(t, x2, i)  += 2.f;
                MEMBER_ACCESS(t, x3, i)  += 2.f;
                MEMBER_ACCESS(t, x4, i)  += 2.f;
                MEMBER_ACCESS(t, x5, i)  += 2.f;
                MEMBER_ACCESS(t, x6, i)  += 2.f;
                MEMBER_ACCESS(t, x7, i)  += 2.f;
                MEMBER_ACCESS(t, x8, i)  += 2.f;
                MEMBER_ACCESS(t, x9, i)  += 2.f;
                MEMBER_ACCESS(t, x10, i) += 2.f;
                MEMBER_ACCESS(t, x11, i) += 2.f;
                MEMBER_ACCESS(t, x12, i) += 2.f;
                MEMBER_ACCESS(t, x13, i) += 2.;
                MEMBER_ACCESS(t, x14, i) += 2.;
                MEMBER_ACCESS(t, x15, i) += 2.;
                MEMBER_ACCESS(t, x16, i) += 2.;
                MEMBER_ACCESS(t, x17, i) += 2.;
                MEMBER_ACCESS(t, x18, i) += 2.;
                MEMBER_ACCESS(t, x19, i) += 2.;
                MEMBER_ACCESS(t, x20, i) += 2.;
                MEMBER_ACCESS(t, x21, i) += 2.;
                MEMBER_ACCESS(t, x22, i) += 2.;
                MEMBER_ACCESS(t, x23, i) += 2.;
                MEMBER_ACCESS(t, x24, i) += 2.;
                MEMBER_ACCESS(t, x25, i) += 2.;
                MEMBER_ACCESS(t, x26, i) += 2;
                MEMBER_ACCESS(t, x27, i) += 2;
                MEMBER_ACCESS(t, x28, i) += 2;
                MEMBER_ACCESS(t, x29, i) += 2;
                MEMBER_ACCESS(t, x30, i) += 2;
                MEMBER_ACCESS(t, x31, i) += 2;
                MEMBER_ACCESS(t, x32, i) += 2;
                MEMBER_ACCESS(t, x33, i) += 2;
                MEMBER_ACCESS(t, x34, i) += 2;
                MEMBER_ACCESS(t, x35, i) += 2;
                MEMBER_ACCESS(t, x36, i) += 2;
                MEMBER_ACCESS(t, x37, i) += 2;
                MEMBER_ACCESS(t, x38, i) += 2;
                MEMBER_ACCESS(t, x39, i) += v;
                MEMBER_ACCESS(t, x40, i) += v;
                MEMBER_ACCESS(t, x41, i) += v;
                MEMBER_ACCESS(t, x42, i) += v;
                MEMBER_ACCESS(t, x43, i) += v;
                MEMBER_ACCESS(t, x44, i) += v;
                MEMBER_ACCESS(t, x45, i) += v;
                MEMBER_ACCESS(t, x46, i) += v;
                MEMBER_ACCESS(t, x47, i) += v;
                MEMBER_ACCESS(t, x48, i) += v;
                MEMBER_ACCESS(t, x49, i) += v;
                MEMBER_ACCESS(t, x50, i) += v;
                MEMBER_ACCESS(t, x51, i) += m;
                MEMBER_ACCESS(t, x52, i) += m;
                MEMBER_ACCESS(t, x53, i) += m;
                MEMBER_ACCESS(t, x54, i) += m;
                MEMBER_ACCESS(t, x55, i) += m;
                MEMBER_ACCESS(t, x56, i) += m;
                MEMBER_ACCESS(t, x57, i) += m;
                MEMBER_ACCESS(t, x58, i) += m;
                MEMBER_ACCESS(t, x59, i) += m;
                MEMBER_ACCESS(t, x60, i) += m;
                MEMBER_ACCESS(t, x61, i) += m;
                MEMBER_ACCESS(t, x62, i) += m;
                MEMBER_ACCESS(t, x63, i) += m;
            }
        }
    }

    state.counters["n_elem"] = state.range(0);
    state.counters["reps"] = repetitions;
}

#endif  // BENCHMARK_H
