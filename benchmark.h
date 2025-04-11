#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <Eigen/Core>

#ifdef SOA_BOOST
    #define MEMBER_ACCESS(NAME) NAME()
#else
    #define MEMBER_ACCESS(NAME) NAME
#endif

using Vector3D = Eigen::Vector3d;
using Matrix3D = Eigen::Matrix3d;

constexpr auto BM_CPUEasyRW_nelem = 10;

// 2 data members, integers, 64 alignment, 10 elements
template <typename T>
void BM_CPUEasyRW(benchmark::State &state, T &t) {
    constexpr auto repetitions = 1000000;

    for (auto _ : state) {
        for (size_t _; _ < repetitions; ++_) {
            for (int i = 0; i < BM_CPUEasyRW_nelem; ++i) {
                t[i].MEMBER_ACCESS(x0) += 2;
                t[i].MEMBER_ACCESS(x1) += 2;
            } 
        }
    }
}

constexpr auto BM_CPUEasyCompute_nelem = 10;

// 2 data members, integers, 64 alignment, 10 elements
template <typename T>
void BM_CPUEasyCompute(benchmark::State &state, T &t) {
    constexpr auto repetitions = 1000000;

    for (auto _ : state) {
        for (size_t _; _ < repetitions; ++_) {
            for (int i = 0; i < BM_CPUEasyCompute_nelem; ++i) {
                t[i].MEMBER_ACCESS(x0) = 1 + t[i].MEMBER_ACCESS(x0) * t[i].MEMBER_ACCESS(x0) * t[i].MEMBER_ACCESS(x0) * 
                                         t[i].MEMBER_ACCESS(x0) * t[i].MEMBER_ACCESS(x0) + 
                                         7 * t[i].MEMBER_ACCESS(x1) * t[i].MEMBER_ACCESS(x1) * 
                                         t[i].MEMBER_ACCESS(x1) - 
                                         6 * t[i].MEMBER_ACCESS(x0) * t[i].MEMBER_ACCESS(x1) * 
                                         t[i].MEMBER_ACCESS(x1) + 
                                         3 * t[i].MEMBER_ACCESS(x0) * t[i].MEMBER_ACCESS(x0) * 
                                         t[i].MEMBER_ACCESS(x1);
                t[i].MEMBER_ACCESS(x1) = 1 + t[i].MEMBER_ACCESS(x1) * t[i].MEMBER_ACCESS(x1) * t[i].MEMBER_ACCESS(x1) * 
                                         t[i].MEMBER_ACCESS(x1) - 
                                         5 * t[i].MEMBER_ACCESS(x0) * t[i].MEMBER_ACCESS(x0) * 
                                         t[i].MEMBER_ACCESS(x0) + 
                                         4 * t[i].MEMBER_ACCESS(x0) * t[i].MEMBER_ACCESS(x1) * 
                                         t[i].MEMBER_ACCESS(x1) - 
                                         2 * t[i].MEMBER_ACCESS(x1) * t[i].MEMBER_ACCESS(x0) * 
                                         t[i].MEMBER_ACCESS(x0);
            } 
        }
    }
}

constexpr auto BM_CPUHardRW_nelem = 10e4;

// 100 data members (20 floats, 20 doubles, 20 integers, 20 Eigen vector, 20 Eigen matrices), 
// 64 alignment, 10e4 elements
template <typename T>
void BM_CPUHardRW(benchmark::State &state, T &t) {
    constexpr auto repetitions = 1;

    Matrix3D m; 
    m << 2, 2, 2, 2, 2, 2, 2, 2, 2;
    Vector3D v(2, 2, 2);

    for (auto _ : state) {
        for (size_t _; _ < repetitions; ++_) {
            for (int i = 0; i < BM_CPUHardRW_nelem; ++i) {
                t[i].MEMBER_ACCESS(x0)  += 2.f;
                t[i].MEMBER_ACCESS(x1)  += 2.f;
                t[i].MEMBER_ACCESS(x2)  += 2.f;
                t[i].MEMBER_ACCESS(x3)  += 2.f;
                t[i].MEMBER_ACCESS(x4)  += 2.f;
                t[i].MEMBER_ACCESS(x5)  += 2.f;
                t[i].MEMBER_ACCESS(x6)  += 2.f;
                t[i].MEMBER_ACCESS(x7)  += 2.f;
                t[i].MEMBER_ACCESS(x8)  += 2.f;
                t[i].MEMBER_ACCESS(x9)  += 2.f;
                t[i].MEMBER_ACCESS(x10) += 2.f;
                t[i].MEMBER_ACCESS(x11) += 2.f;
                t[i].MEMBER_ACCESS(x12) += 2.f;
                t[i].MEMBER_ACCESS(x13) += 2.;
                t[i].MEMBER_ACCESS(x14) += 2.;
                t[i].MEMBER_ACCESS(x15) += 2.;
                t[i].MEMBER_ACCESS(x16) += 2.;
                t[i].MEMBER_ACCESS(x17) += 2.;
                t[i].MEMBER_ACCESS(x18) += 2.;
                t[i].MEMBER_ACCESS(x19) += 2.;
                t[i].MEMBER_ACCESS(x20) += 2.;
                t[i].MEMBER_ACCESS(x21) += 2.;
                t[i].MEMBER_ACCESS(x22) += 2.;
                t[i].MEMBER_ACCESS(x23) += 2.;
                t[i].MEMBER_ACCESS(x24) += 2.;
                t[i].MEMBER_ACCESS(x25) += 2.;
                t[i].MEMBER_ACCESS(x26) += 2;
                t[i].MEMBER_ACCESS(x27) += 2;
                t[i].MEMBER_ACCESS(x28) += 2;
                t[i].MEMBER_ACCESS(x29) += 2;
                t[i].MEMBER_ACCESS(x30) += 2;
                t[i].MEMBER_ACCESS(x31) += 2;
                t[i].MEMBER_ACCESS(x32) += 2;
                t[i].MEMBER_ACCESS(x33) += 2;
                t[i].MEMBER_ACCESS(x34) += 2;
                t[i].MEMBER_ACCESS(x35) += 2;
                t[i].MEMBER_ACCESS(x36) += 2;
                t[i].MEMBER_ACCESS(x37) += 2;
                t[i].MEMBER_ACCESS(x38) += 2;
                t[i].MEMBER_ACCESS(x39) += v;
                t[i].MEMBER_ACCESS(x40) += v;
                t[i].MEMBER_ACCESS(x41) += v;
                t[i].MEMBER_ACCESS(x42) += v;
                t[i].MEMBER_ACCESS(x43) += v;
                t[i].MEMBER_ACCESS(x44) += v;
                t[i].MEMBER_ACCESS(x45) += v;
                t[i].MEMBER_ACCESS(x46) += v;
                t[i].MEMBER_ACCESS(x47) += v;
                t[i].MEMBER_ACCESS(x48) += v;
                t[i].MEMBER_ACCESS(x49) += v;
                t[i].MEMBER_ACCESS(x50) += v;
                t[i].MEMBER_ACCESS(x51) += m;
                t[i].MEMBER_ACCESS(x52) += m;
                t[i].MEMBER_ACCESS(x53) += m;
                t[i].MEMBER_ACCESS(x54) += m;
                t[i].MEMBER_ACCESS(x55) += m;
                t[i].MEMBER_ACCESS(x56) += m;
                t[i].MEMBER_ACCESS(x57) += m;
                t[i].MEMBER_ACCESS(x58) += m;
                t[i].MEMBER_ACCESS(x59) += m;
                t[i].MEMBER_ACCESS(x60) += m;
                t[i].MEMBER_ACCESS(x61) += m;
                t[i].MEMBER_ACCESS(x62) += m;
                t[i].MEMBER_ACCESS(x63) += m;
            } 
        }
    }
}

// “Realistic case”: 
//      10 data members (3 doubles, 3 float, 2 integer, 1 Vector3D, 1 Matrix), 64 alignment, 100000
template <typename T>
void BM_CPURealRW(benchmark::State &state, T &t) {
    constexpr auto n_elements = 100000;
    constexpr auto repetitions = 1;

    for (auto _ : state) {
        for (size_t _; _ < repetitions; ++_) {
            for (int i = 0; i < n_elements; ++i) {
                t[i].MEMBER_ACCESS(x0) += (double) 2;
                t[i].MEMBER_ACCESS(x1) += (double) 2;
                t[i].MEMBER_ACCESS(x2) += (double) 2;
                t[i].MEMBER_ACCESS(x3) += (float) 2;
                t[i].MEMBER_ACCESS(x4) += (float) 2;
                t[i].MEMBER_ACCESS(x5) += (float)2;
                t[i].MEMBER_ACCESS(x6) += 2;
                t[i].MEMBER_ACCESS(x7) += 2;
                t[i].MEMBER_ACCESS(x8) += Eigen::Vector3d::Constant(2.0);
                t[i].MEMBER_ACCESS(x9) += Eigen::Matrix3d::Constant(2.0);
            } 
        }
    }
}

#endif  // BENCHMARK_H
