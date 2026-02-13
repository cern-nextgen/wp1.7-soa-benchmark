#define EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO 0

#include "benchmark.h"
#include <Eigen/Core>
#include "wrapper.h"
#include <vector>

#include <iostream>
#include <experimental/meta>

struct S2 {
    int &x0, &x1;
};

struct S10 {
    float &x0, &x1;
    double &x2, &x3;
    int &x4, &x5;
    Eigen::Vector3d &x6, &x7;
    Eigen::Matrix3d &x8, &x9;
};

struct S32 {
    float &x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7, &x8, &x9, &x10, &x11, &x12, &x13, &x14, &x15, &x16, &x17, &x18, &x19,
        &x20, &x21, &x22, &x23, &x24, &x25, &x26, &x27, &x28, &x29, &x30, &x31;
};

struct S64 {
    double &x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7, &x8, &x9, &x10, &x11, &x12;
    float &x13, &x14, &x15, &x16, &x17, &x18, &x19, &x20, &x21, &x22, &x23, &x24, &x25;
    int &x26, &x27, &x28, &x29, &x30, &x31, &x32, &x33, &x34, &x35, &x36, &x37, &x38;
    Eigen::Vector3d &x39, &x40, &x41, &x42, &x43, &x44, &x45, &x46, &x47, &x48, &x49, &x50;
    Eigen::Matrix3d &x51, &x52, &x53, &x54, &x55, &x56, &x57, &x58, &x59, &x60, &x61, &x62, &x63;
};

struct Snbody {
    float &x, &y, &z, &vx, &vy, &vz;
};

struct Sstencil {
    double &src, &dst, &rhs;
};

struct PxPyPzM {
    double &x, &y, &z, &M;
};

/// Register Benchmarks ///
template <typename S, typename N>
class Fixture1 : public benchmark::Fixture {
 public:
    using SoA = Wrapper<S, std::vector>;

    SoA t;

    static constexpr auto n = N::value;

    void SetUp(::benchmark::State &state /* unused */) override
    {
        // Initialize each vector in the SoA with n elements.
        auto construct_soa = [&]<size_t... Is>(std::index_sequence<Is...>) {
            return SoA{typename[:type_of(nsdms(^^typename SoA::Base)[Is]):](n)...};
        };
        t = construct_soa(std::make_index_sequence<nsdms(^^S).size()>());
    }

    Fixture1() : t() {}
};

INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyRW, S2, N_Large);
INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyCompute, S2, N);
INSTANTIATE_BENCHMARKS_F1(BM_CPURealRW, S10, N);
INSTANTIATE_BENCHMARKS_F1(BM_CPUHardRW, S64, N);
INSTANTIATE_BENCHMARKS_F1(BM_nbody, Snbody, N);
INSTANTIATE_BENCHMARKS_F1(BM_stencil, Sstencil, N_Large);

template <typename S1, typename S2, typename N>
class Fixture2 : public benchmark::Fixture {
 public:
    using SoA1 = Wrapper<S1, std::vector>;
    using SoA2 = Wrapper<S2, std::vector>;

    SoA1 t1;
    SoA2 t2;

    static constexpr auto n = N::value;

    void SetUp(::benchmark::State &state /* unused */) override
    {
        // Initialize each vector in the SoA with n elements.
        auto construct_soa = [&]<typename SoA, size_t... Is>(std::index_sequence<Is...>) {
            return SoA{typename[:type_of(nsdms(^^typename SoA::Base)[Is]):](n)...};
        };
        t1 = construct_soa.template operator()<SoA1>(std::make_index_sequence<nsdms(^^S1).size()>());
        t2 = construct_soa.template operator()<SoA2>(std::make_index_sequence<nsdms(^^S2).size()>());
    }

    Fixture2() : t1(), t2() {}
};

INSTANTIATE_BENCHMARKS_F2(BM_InvariantMass, PxPyPzM, PxPyPzM, N_Large);

BENCHMARK_MAIN();
