#ifndef MANUAL_AOS_STRUCTS_H
#define MANUAL_AOS_STRUCTS_H

#include <Eigen/Core>
#include "benchmarks/common.h"

namespace manual {

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
    uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,
          x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31;
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

// Realistic AoS particle with extra fields beyond (x,y,z,M) to model real-world padding cost
struct Particle {
    int id;
    double x, y, z, M;
    double fX, fY, fZ, fM;
    double poscovmatrix[9];
};

/// Fixtures ///

template <class S, class N, class BackendType>
class Fixture1 : public benchmark::Fixture {
public:
    static constexpr auto n = N::value;
    static constexpr Backend backend = BackendType::value;
    S* t = nullptr;

    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;
    void SetUp(benchmark::State&)    { t = backend_allocator<backend>::template alloc<S>(n); }
    void TearDown(benchmark::State&)  { backend_allocator<backend>::free(t); t = nullptr; }
};

template <class S1, class S2, class N, class BackendType>
class Fixture2 : public benchmark::Fixture {
public:
    static constexpr auto n = N::value;
    static constexpr Backend backend = BackendType::value;
    S1* t1 = nullptr;
    S2* t2 = nullptr;

    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;
    void SetUp(benchmark::State&)  {
        t1 = backend_allocator<backend>::template alloc<S1>(n);
        t2 = backend_allocator<backend>::template alloc<S2>(n);
    }
    void TearDown(benchmark::State&)  {
        backend_allocator<backend>::free(t1); t1 = nullptr;
        backend_allocator<backend>::free(t2); t2 = nullptr;
    }
};

} // namespace manual

#endif // MANUAL_AOS_STRUCTS_H
