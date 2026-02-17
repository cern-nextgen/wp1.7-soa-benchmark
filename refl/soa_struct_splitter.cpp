#define EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO 0

#include "benchmark.h"
#include "struct_splitter.h"
#include <Eigen/Core>
#include <vector>

struct S2 {
    int &x0, &x1;
};

template <size_t... Is>
struct SubS2;
consteval
{
    SplitStruct<S2, SubS2>(SplitOp({0}), SplitOp({1}));
}

using S2SoA = PartitionedContainerContiguous<S2, SubS2<0>, SubS2<1>>;

struct S10 {
    float &x0, &x1;
    double &x2, &x3;
    int &x4, &x5;
    Eigen::Vector3d &x6, &x7;
    Eigen::Matrix3d &x8, &x9;
};

template <size_t... Is>
struct SubS10;
consteval
{
    SplitStruct<S10, SubS10>(SplitOp({0}), SplitOp({1}), SplitOp({2}), SplitOp({3}), SplitOp({4}), SplitOp({5}),
                             SplitOp({6}), SplitOp({7}), SplitOp({8}), SplitOp({9}));
}

using S10SoA = PartitionedContainerContiguous<S10, SubS10<0>, SubS10<1>, SubS10<2>, SubS10<3>, SubS10<4>, SubS10<5>,
                                              SubS10<6>, SubS10<7>, SubS10<8>, SubS10<9>>;

struct S32 {
    float &x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7, &x8, &x9, &x10, &x11, &x12, &x13, &x14, &x15, &x16, &x17, &x18, &x19,
        &x20, &x21, &x22, &x23, &x24, &x25, &x26, &x27, &x28, &x29, &x30, &x31;
};

template <size_t... Is>
struct SubS32;
consteval
{
    SplitStruct<S32, SubS32>(SplitOp({0}), SplitOp({1}), SplitOp({2}), SplitOp({3}), SplitOp({4}), SplitOp({5}),
                             SplitOp({6}), SplitOp({7}), SplitOp({8}), SplitOp({9}), SplitOp({10}), SplitOp({11}),
                             SplitOp({12}), SplitOp({13}), SplitOp({14}), SplitOp({15}), SplitOp({16}), SplitOp({17}),
                             SplitOp({18}), SplitOp({19}), SplitOp({20}), SplitOp({21}), SplitOp({22}), SplitOp({23}),
                             SplitOp({24}), SplitOp({25}), SplitOp({26}), SplitOp({27}), SplitOp({28}), SplitOp({29}),
                             SplitOp({30}), SplitOp({31}));
}

using S32SoA =
    PartitionedContainerContiguous<S32, SubS32<0>, SubS32<1>, SubS32<2>, SubS32<3>, SubS32<4>, SubS32<5>, SubS32<6>,
                                   SubS32<7>, SubS32<8>, SubS32<9>, SubS32<10>, SubS32<11>, SubS32<12>, SubS32<13>,
                                   SubS32<14>, SubS32<15>, SubS32<16>, SubS32<17>, SubS32<18>, SubS32<19>, SubS32<20>,
                                   SubS32<21>, SubS32<22>, SubS32<23>, SubS32<24>, SubS32<25>, SubS32<26>, SubS32<27>,
                                   SubS32<28>, SubS32<29>, SubS32<30>, SubS32<31>>;

struct S64 {
    double &x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7, &x8, &x9, &x10, &x11, &x12;
    float &x13, &x14, &x15, &x16, &x17, &x18, &x19, &x20, &x21, &x22, &x23, &x24, &x25;
    int &x26, &x27, &x28, &x29, &x30, &x31, &x32, &x33, &x34, &x35, &x36, &x37, &x38;
    Eigen::Vector3d &x39, &x40, &x41, &x42, &x43, &x44, &x45, &x46, &x47, &x48, &x49, &x50;
    Eigen::Matrix3d &x51, &x52, &x53, &x54, &x55, &x56, &x57, &x58, &x59, &x60, &x61, &x62, &x63;
};

template <size_t... Is>
struct SubS64;
consteval
{
    SplitStruct<S64, SubS64>(
        SplitOp({0}), SplitOp({1}), SplitOp({2}), SplitOp({3}), SplitOp({4}), SplitOp({5}), SplitOp({6}), SplitOp({7}),
        SplitOp({8}), SplitOp({9}), SplitOp({10}), SplitOp({11}), SplitOp({12}), SplitOp({13}), SplitOp({14}),
        SplitOp({15}), SplitOp({16}), SplitOp({17}), SplitOp({18}), SplitOp({19}), SplitOp({20}), SplitOp({21}),
        SplitOp({22}), SplitOp({23}), SplitOp({24}), SplitOp({25}), SplitOp({26}), SplitOp({27}), SplitOp({28}),
        SplitOp({29}), SplitOp({30}), SplitOp({31}), SplitOp({32}), SplitOp({33}), SplitOp({34}), SplitOp({35}),
        SplitOp({36}), SplitOp({37}), SplitOp({38}), SplitOp({39}), SplitOp({40}), SplitOp({41}), SplitOp({42}),
        SplitOp({43}), SplitOp({44}), SplitOp({45}), SplitOp({46}), SplitOp({47}), SplitOp({48}), SplitOp({49}),
        SplitOp({50}), SplitOp({51}), SplitOp({52}), SplitOp({53}), SplitOp({54}), SplitOp({55}), SplitOp({56}),
        SplitOp({57}), SplitOp({58}), SplitOp({59}), SplitOp({60}), SplitOp({61}), SplitOp({62}), SplitOp({63}));
}

using S64SoA = PartitionedContainerContiguous<
    S64, SubS64<0>, SubS64<1>, SubS64<2>, SubS64<3>, SubS64<4>, SubS64<5>, SubS64<6>, SubS64<7>, SubS64<8>, SubS64<9>,
    SubS64<0>, SubS64<1>, SubS64<2>, SubS64<3>, SubS64<4>, SubS64<5>, SubS64<6>, SubS64<7>, SubS64<8>, SubS64<9>,
    SubS64<0>, SubS64<1>, SubS64<2>, SubS64<3>, SubS64<4>, SubS64<5>, SubS64<6>, SubS64<7>, SubS64<8>, SubS64<9>,
    SubS64<0>, SubS64<1>, SubS64<2>, SubS64<3>, SubS64<4>, SubS64<5>, SubS64<6>, SubS64<7>, SubS64<8>, SubS64<9>,
    SubS64<0>, SubS64<1>, SubS64<2>, SubS64<3>, SubS64<4>, SubS64<5>, SubS64<6>, SubS64<7>, SubS64<8>, SubS64<9>,
    SubS64<0>, SubS64<1>, SubS64<2>, SubS64<3>, SubS64<4>, SubS64<5>, SubS64<6>, SubS64<7>, SubS64<8>, SubS64<9>,
    SubS64<0>, SubS64<1>, SubS64<2>, SubS64<3>>;

struct Snbody {
    float &x, &y, &z, &vx, &vy, &vz;
};

template <size_t... Is>
struct SubSnbody;
consteval
{
    SplitStruct<Snbody, SubSnbody>(SplitOp({0}), SplitOp({1}), SplitOp({2}), SplitOp({3}), SplitOp({4}), SplitOp({5}));
}

using SnbodySoA = PartitionedContainerContiguous<Snbody, SubSnbody<0>, SubSnbody<1>, SubSnbody<2>, SubSnbody<3>,
                                                 SubSnbody<4>, SubSnbody<5>>;

struct Sstencil {
    double &src, &dst, &rhs;
};

template <size_t... Is>
struct SubSstencil;
consteval
{
    SplitStruct<Sstencil, SubSstencil>(SplitOp({0}), SplitOp({1}), SplitOp({2}));
}

using SstencilSoA = PartitionedContainerContiguous<Sstencil, SubSstencil<0>, SubSstencil<1>, SubSstencil<2>>;

struct PxPyPzM {
    double &x, &y, &z, &M;
};

template <size_t... Is>
struct SubPxPyPzM;
consteval
{
    SplitStruct<PxPyPzM, SubPxPyPzM>(SplitOp({0}), SplitOp({1}), SplitOp({2}), SplitOp({3}));
}

using PxPyPzMSoA = PartitionedContainerContiguous<PxPyPzM, SubPxPyPzM<0>, SubPxPyPzM<1>, SubPxPyPzM<2>, SubPxPyPzM<3>>;

/// Register Benchmarks ///
template <typename SoA, typename N>
class Fixture1 : public benchmark::Fixture {
 public:
    std::byte *buf;

    SoA t;

    static constexpr auto n = N::value;

    void SetUp(::benchmark::State &state /* unused */) override
    {
        buf = static_cast<std::byte *>(std::aligned_alloc(Alignment, SoA::ComputeSize(n, Alignment)));
        t = SoA(buf, n, Alignment);
    }

    void TearDown(::benchmark::State &state /* unused */) override { std::free(buf); }

    Fixture1() : t(nullptr, 0, Alignment) {}
};

// INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyRW, S2SoA, N_Large);
// INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyCompute, S2SoA, N);
// INSTANTIATE_BENCHMARKS_F1(BM_CPURealRW, S10SoA, N);
// INSTANTIATE_BENCHMARKS_F1(BM_CPUStrided, S32SoA, N_Large);
// INSTANTIATE_BENCHMARKS_F1(BM_CPUHardRW, S64SoA, N);
INSTANTIATE_BENCHMARKS_F1(BM_nbody, SnbodySoA, N);
INSTANTIATE_BENCHMARKS_F1(BM_stencil, SstencilSoA, N_Large);

template <typename SoA1, typename SoA2, typename N>
class Fixture2 : public benchmark::Fixture {
 public:
    std::byte *buf1, *buf2;

    SoA1 t1;
    SoA2 t2;

    static constexpr auto n = N::value;

    void SetUp(::benchmark::State &state /* unused */) override
    {
        buf1 = static_cast<std::byte *>(std::aligned_alloc(Alignment, SoA1::ComputeSize(n, Alignment)));
        t1 = SoA1(buf1, n, Alignment);
        buf2 = static_cast<std::byte *>(std::aligned_alloc(Alignment, SoA2::ComputeSize(n, Alignment)));
        t2 = SoA2(buf2, n, Alignment);
    }

    void TearDown(::benchmark::State &state /* unused */) override
    {
        std::free(buf1);
        std::free(buf2);
    }

    Fixture2() : t1(nullptr, 0, Alignment), t2(nullptr, 0, Alignment) {}
};

INSTANTIATE_BENCHMARKS_F2(BM_InvariantMass, PxPyPzMSoA, PxPyPzMSoA, N_Large);

BENCHMARK_MAIN();
