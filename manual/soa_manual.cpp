#define SOA_MANUAL

#include <cstdlib>
#include <meta>
#include <vector>
#include <Eigen/Core>
#include "benchmarks/common.h"

template <class T>
T* alloc(size_t n) { return static_cast<T*>(std::malloc(n * sizeof(T))); }

//////////////// Reflection helpers (same pattern as reflections/wrapper.h)

namespace __impl {
template <auto... vals>
struct replicator_type {
    template <typename F>
    constexpr auto operator>>(F body) const -> decltype(auto) {
        return body.template operator()<vals...>();
    }
};
template <auto... vals>
replicator_type<vals...> replicator = {};
} // namespace __impl

consteval auto nsdms(std::meta::info type) -> std::vector<std::meta::info> {
    return nonstatic_data_members_of(type, std::meta::access_context::unchecked());
}

template <typename R>
consteval auto expand_all(R range) {
    std::vector<std::meta::info> args;
    for (auto r : range)
        args.push_back(reflect_constant(r));
    return substitute(^^__impl::replicator, args);
}

//////////////// Generic allocate / deallocate over all pointer members

template <class S>
void allocate(S& s, size_t n) {
    [: expand_all(nsdms(^^S)) :] >> [&]<auto... Ms>() {
        ((s.[:Ms:] = alloc<std::remove_pointer_t<typename[:type_of(Ms):]>>(n)), ...);
    };
}

template <class S>
void deallocate(S& s) {
    [: expand_all(nsdms(^^S)) :] >> [&]<auto... Ms>() {
        (std::free(s.[:Ms:]), ...);
    };
}

/// Data structures: SoA with raw pointers, one allocation per member array ///

struct S2 {
    int *__restrict__ x0, *__restrict__ x1;
};

struct S10 {
    float *__restrict__ x0, *__restrict__ x1;
    double *__restrict__ x2, *__restrict__ x3;
    int *__restrict__ x4, *__restrict__ x5;
    Eigen::Vector3d *__restrict__ x6, *__restrict__ x7;
    Eigen::Matrix3d *__restrict__ x8, *__restrict__ x9;
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
};

struct Snbody {
    float *__restrict__ x, *__restrict__ y, *__restrict__ z;
    float *__restrict__ vx, *__restrict__ vy, *__restrict__ vz;
};

struct Sstencil {
    double *__restrict__ src, *__restrict__ dst, *__restrict__ rhs;
};

struct PxPyPzM {
    double *__restrict__ x, *__restrict__ y, *__restrict__ z, *__restrict__ M;
};

/// Fixtures ///

template <typename S, typename N>
class Fixture1 : public benchmark::Fixture {
public:
    static constexpr auto n = N::value;
    S t;

    void SetUp(benchmark::State &) override { allocate(t, n); }
    void TearDown(benchmark::State &) override { deallocate(t); }
};

template <typename S1, typename S2, typename N>
class Fixture2 : public benchmark::Fixture {
public:
    static constexpr auto n = N::value;
    S1 t1;
    S2 t2;

    void SetUp(benchmark::State &) override { allocate(t1, n); allocate(t2, n); }
    void TearDown(benchmark::State &) override { deallocate(t1); deallocate(t2); }
};

/// Benchmarks ///

#include "benchmarks/bm_easy.h"
#include "benchmarks/bm_real.h"
#include "benchmarks/bm_strided.h"
#include "benchmarks/bm_hard.h"
#include "benchmarks/bm_nbody.h"
#include "benchmarks/bm_stencil.h"
#include "benchmarks/bm_invmass.h"

INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyRW,      S2,       N_Large);
INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyCompute, S2,       N);
INSTANTIATE_BENCHMARKS_F1(BM_CPURealRW,      S10,      N);
INSTANTIATE_BENCHMARKS_F1(BM_CPUStrided,     S32,      N_Large);
INSTANTIATE_BENCHMARKS_F1(BM_CPUHardRW,      S64,      N);
INSTANTIATE_BENCHMARKS_F1(BM_nbody,          Snbody,   N);
INSTANTIATE_BENCHMARKS_F1(BM_stencil,        Sstencil, N_Large);

INSTANTIATE_BENCHMARKS_F2(BM_InvariantMass, PxPyPzM, PxPyPzM, N_Large);

BENCHMARK_MAIN();
