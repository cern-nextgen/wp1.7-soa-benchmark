#include <cstdlib>
#include <Eigen/Core>
#include "benchmarks/common.h"
#include "reflections/wrapper.h"

struct mallocator {
    std::size_t n;
    template <class... Args>
    void operator()(Args*&... args) const { ((args = static_cast<Args*>(std::malloc(n * sizeof(Args)))), ...); }
};

struct deallocator {
    template <class... Args>
    void operator()(Args*&... args) const { ((std::free(args), args = nullptr), ...); }
};

template <class ArrayType>
void allocate(ArrayType& w, std::size_t n) { w.apply(mallocator{n}); }

template <class ArrayType>
void deallocate(ArrayType& w) { w.apply(deallocator{}); }

struct S2 { int &x0, &x1; };

struct S10 {
    float &x0, &x1;
    double &x2, &x3;
    int &x4, &x5;
    Eigen::Vector3d &x6, &x7;
    Eigen::Matrix3d &x8, &x9;
};

struct S32 {
    uint32_t &x0,  &x1,  &x2,  &x3,  &x4,  &x5,  &x6,  &x7,
          &x8,  &x9,  &x10, &x11, &x12, &x13, &x14, &x15,
          &x16, &x17, &x18, &x19, &x20, &x21, &x22, &x23,
          &x24, &x25, &x26, &x27, &x28, &x29, &x30, &x31;
};

struct S64 {
    float          &x0,  &x1,  &x2,  &x3,  &x4,  &x5,  &x6,  &x7,  &x8,  &x9,  &x10, &x11, &x12;
    double         &x13, &x14, &x15, &x16, &x17, &x18, &x19, &x20, &x21, &x22, &x23, &x24, &x25;
    int            &x26, &x27, &x28, &x29, &x30, &x31, &x32, &x33, &x34, &x35, &x36, &x37, &x38;
    Eigen::Vector3d &x39, &x40, &x41, &x42, &x43, &x44, &x45, &x46, &x47, &x48, &x49, &x50;
    Eigen::Matrix3d &x51, &x52, &x53, &x54, &x55, &x56, &x57, &x58, &x59, &x60, &x61, &x62, &x63;
};

struct Snbody   { float &x, &y, &z, &vx, &vy, &vz; };
struct Sstencil { double &src, &dst, &rhs; };
struct PxPyPzM  { double &x, &y, &z, &M; };

using S2SoA       = Wrapper<S2,       pointer>;
using S10SoA      = Wrapper<S10,      pointer>;
using S32SoA      = Wrapper<S32,      pointer>;
using S64SoA      = Wrapper<S64,      pointer>;
using SnbodySoA   = Wrapper<Snbody,   pointer>;
using SstencilSoA = Wrapper<Sstencil, pointer>;
using PxPyPzMSoA  = Wrapper<PxPyPzM,  pointer>;

/// Fixtures ///

template <typename ArrayType, typename N>
class Fixture1 : public benchmark::Fixture {
public:
    static constexpr auto n = N::value;
    static constexpr Backend backend = Backend::CPU;
    ArrayType t;

    void SetUp(benchmark::State&) override { allocate(t, n); }
    void TearDown(benchmark::State&) override { deallocate(t); }
};

template <typename ArrayType1, typename ArrayType2, typename N>
class Fixture2 : public benchmark::Fixture {
public:
    static constexpr auto n = N::value;
    static constexpr Backend backend = Backend::CPU;
    ArrayType1 t1;
    ArrayType2 t2;

    void SetUp(benchmark::State&) override { allocate(t1, n); allocate(t2, n); }
    void TearDown(benchmark::State&) override { deallocate(t1); deallocate(t2); }
};

/// Benchmarks ///

#include "benchmarks/easy.h"
#include "benchmarks/easycompute.h"
#include "benchmarks/real.h"
#include "benchmarks/strided.h"
#include "benchmarks/hard.h"
#include "benchmarks/nbody.h"
#include "benchmarks/stencil.h"
#include "benchmarks/invmass.h"

INSTANTIATE_BENCHMARKS_F1(EasyRW,      S2SoA,       N_Large);
INSTANTIATE_BENCHMARKS_F1(EasyCompute, S2SoA,       N);
INSTANTIATE_BENCHMARKS_F1(RealRW,      S10SoA,      N);
INSTANTIATE_BENCHMARKS_F1(Strided,     S32SoA,      N_Large);
INSTANTIATE_BENCHMARKS_F1(HardRW,      S64SoA,      N);
INSTANTIATE_BENCHMARKS_F1(NBody,       SnbodySoA,   N);
INSTANTIATE_BENCHMARKS_F1(Stencil,     SstencilSoA, N_Large);

INSTANTIATE_BENCHMARKS_F2(InvariantMass, PxPyPzMSoA, PxPyPzMSoA, N_Large);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::AddCustomContext("name", "Reflection SoA CPU");
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
