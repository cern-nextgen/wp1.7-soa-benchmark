#define AOS_MANUAL

#include <Eigen/Core>

#include "benchmark.h"

struct S2 {
    int x0, x1;
};

struct S10 {
    float x0, x1;
    double x2, x3;
    int x4, x5;
    Eigen::Vector3d x6, x7;
    Eigen::Matrix3d x8, x9;
};

struct S32 {
    float x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31;
};

struct S64 {
    float x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
    double x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25;
    int x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38;
    Eigen::Vector3d x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50;
    Eigen::Matrix3d x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63;
};

struct Snbody {
    float x, y, z, vx, vy, vz;
};

struct Sstencil {
    double src, dst, rhs;
};

struct PxPyPzM {
    double x, y, z, M;
};

/// Register Benchmarks ///
template <typename S, typename N>
class Fixture1 : public benchmark::Fixture {
 public:
    static constexpr auto n = N::value;
    S t[n];
};

INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyRW, S2, N_Large);
INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyCompute, S2, N);
INSTANTIATE_BENCHMARKS_F1(BM_CPURealRW, S10, N);
INSTANTIATE_BENCHMARKS_F1(BM_CPUCacheAssociativity, S32, N_Large);
INSTANTIATE_BENCHMARKS_F1(BM_CPUHardRW, S64, N);
INSTANTIATE_BENCHMARKS_F1(BM_nbody, Snbody, N);
INSTANTIATE_BENCHMARKS_F1(BM_stencil, Sstencil, N_Large);

template <typename S1, typename S2, typename N>
class Fixture2 : public benchmark::Fixture {
 public:
    static constexpr auto n = N::value;
    S1 t1[n];
    S2 t2[n];
};

INSTANTIATE_BENCHMARKS_F2(BM_InvariantMass, PxPyPzM, PxPyPzM, N_Large);

BENCHMARK_MAIN();
