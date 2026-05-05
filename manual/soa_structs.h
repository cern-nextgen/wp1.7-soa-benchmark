#ifndef MANUAL_SOA_STRUCTS_H
#define MANUAL_SOA_STRUCTS_H

#include <type_traits>
#include <Eigen/Core>
#include "benchmarks/common.h"

//////////////// Allocation helpers (Backend-templated)

template <Backend B>
struct mallocator {
    std::size_t n;
    template <class... Ptrs>
    void operator()(Ptrs&... args) const {
        ((args = backend_allocator<B>::template alloc<std::remove_reference_t<decltype(*args)>>(n)), ...);
    }
};

template <Backend B>
struct deallocator {
    template <class... Ptrs>
    void operator()(Ptrs&... args) const {
        ((backend_allocator<B>::free(args), args = nullptr), ...);
    }
};

template <Backend B, class S>
void allocate(S& s, std::size_t n) { s.apply(mallocator<B>{n}); }

template <Backend B, class S>
void deallocate(S& s) { s.apply(deallocator<B>{}); }

//////////////// Data structures

#define SOA_APPLY(...) \
    template <class F> void apply(F&& f) { f(__VA_ARGS__); } \
    template <class F> void apply(F&& f) const { f(__VA_ARGS__); }

struct S2 {
    int *__restrict__ x0, *__restrict__ x1;
    SOA_APPLY(x0, x1)
};

struct S10 {
    float *__restrict__ x0, *__restrict__ x1;
    double *__restrict__ x2, *__restrict__ x3;
    int *__restrict__ x4, *__restrict__ x5;
    Eigen::Vector3d *__restrict__ x6, *__restrict__ x7;
    Eigen::Matrix3d *__restrict__ x8, *__restrict__ x9;
    SOA_APPLY(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)
};

struct S32 {
    uint32_t *__restrict__ x0,  *__restrict__ x1,  *__restrict__ x2,  *__restrict__ x3,
          *__restrict__ x4,  *__restrict__ x5,  *__restrict__ x6,  *__restrict__ x7,
          *__restrict__ x8,  *__restrict__ x9,  *__restrict__ x10, *__restrict__ x11,
          *__restrict__ x12, *__restrict__ x13, *__restrict__ x14, *__restrict__ x15,
          *__restrict__ x16, *__restrict__ x17, *__restrict__ x18, *__restrict__ x19,
          *__restrict__ x20, *__restrict__ x21, *__restrict__ x22, *__restrict__ x23,
          *__restrict__ x24, *__restrict__ x25, *__restrict__ x26, *__restrict__ x27,
          *__restrict__ x28, *__restrict__ x29, *__restrict__ x30, *__restrict__ x31;
    SOA_APPLY(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,
              x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31)
};

struct S64 {
    float           *__restrict__ x0,  *__restrict__ x1,  *__restrict__ x2,
                    *__restrict__ x3,  *__restrict__ x4,  *__restrict__ x5,
                    *__restrict__ x6,  *__restrict__ x7,  *__restrict__ x8,
                    *__restrict__ x9,  *__restrict__ x10, *__restrict__ x11,
                    *__restrict__ x12;
    double          *__restrict__ x13, *__restrict__ x14, *__restrict__ x15,
                    *__restrict__ x16, *__restrict__ x17, *__restrict__ x18,
                    *__restrict__ x19, *__restrict__ x20, *__restrict__ x21,
                    *__restrict__ x22, *__restrict__ x23, *__restrict__ x24,
                    *__restrict__ x25;
    int             *__restrict__ x26, *__restrict__ x27, *__restrict__ x28,
                    *__restrict__ x29, *__restrict__ x30, *__restrict__ x31,
                    *__restrict__ x32, *__restrict__ x33, *__restrict__ x34,
                    *__restrict__ x35, *__restrict__ x36, *__restrict__ x37,
                    *__restrict__ x38;
    Eigen::Vector3d *__restrict__ x39, *__restrict__ x40, *__restrict__ x41,
                    *__restrict__ x42, *__restrict__ x43, *__restrict__ x44,
                    *__restrict__ x45, *__restrict__ x46, *__restrict__ x47,
                    *__restrict__ x48, *__restrict__ x49, *__restrict__ x50;
    Eigen::Matrix3d *__restrict__ x51, *__restrict__ x52, *__restrict__ x53,
                    *__restrict__ x54, *__restrict__ x55, *__restrict__ x56,
                    *__restrict__ x57, *__restrict__ x58, *__restrict__ x59,
                    *__restrict__ x60, *__restrict__ x61, *__restrict__ x62,
                    *__restrict__ x63;
    SOA_APPLY(
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,
        x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25,
        x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38,
        x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50,
        x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63
    )
};

struct Snbody {
    float *__restrict__ x, *__restrict__ y, *__restrict__ z;
    float *__restrict__ vx, *__restrict__ vy, *__restrict__ vz;
    SOA_APPLY(x, y, z, vx, vy, vz)
};

struct Sstencil {
    double *__restrict__ src, *__restrict__ dst, *__restrict__ rhs;
    SOA_APPLY(src, dst, rhs)
};

struct PxPyPzM {
    double *__restrict__ x, *__restrict__ y, *__restrict__ z, *__restrict__ M;
    SOA_APPLY(x, y, z, M)
};

/// Fixtures ///

template <typename S, typename N, typename BT = CPUBackend>
class Fixture1 : public benchmark::Fixture {
public:
    static constexpr auto n = N::value;
    static constexpr Backend backend = BT::value;
    S t;

    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;
    void SetUp(benchmark::State&) override   { allocate<backend>(t, n); }
    void TearDown(benchmark::State&) override { deallocate<backend>(t); }
};

template <typename S1, typename S2, typename N, typename BT = CPUBackend>
class Fixture2 : public benchmark::Fixture {
public:
    static constexpr auto n = N::value;
    static constexpr Backend backend = BT::value;
    S1 t1;
    S2 t2;

    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;
    void SetUp(benchmark::State&) override   { allocate<backend>(t1, n); allocate<backend>(t2, n); }
    void TearDown(benchmark::State&) override { deallocate<backend>(t1); deallocate<backend>(t2); }
};

#endif // MANUAL_SOA_STRUCTS_H
