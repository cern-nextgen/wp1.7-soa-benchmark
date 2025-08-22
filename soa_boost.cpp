#define SOA_BOOST

#include <memory>

#include "benchmark.h"
#include "boost/SoALayout.h"
#include <Eigen/Core>

GENERATE_SOA_LAYOUT(SoALayout,
    SOA_COLUMN(int, x0),
    SOA_COLUMN(int, x1))

using SoA = SoALayout<>;

GENERATE_SOA_LAYOUT(MediumSoALayout,
    SOA_COLUMN(float, x0),
    SOA_COLUMN(float, x1),
    SOA_COLUMN(double, x2),
    SOA_COLUMN(double, x3),
    SOA_COLUMN(int, x4),
    SOA_COLUMN(int, x5),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, x6),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, x7),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x8),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x9))

using MediumSoA = MediumSoALayout<>;

GENERATE_SOA_LAYOUT(BigSoALayout,
    SOA_COLUMN(float, x0),
    SOA_COLUMN(float, x1),
    SOA_COLUMN(float, x2),
    SOA_COLUMN(float, x3),
    SOA_COLUMN(float, x4),
    SOA_COLUMN(float, x5),
    SOA_COLUMN(float, x6),
    SOA_COLUMN(float, x7),
    SOA_COLUMN(float, x8),
    SOA_COLUMN(float, x9),
    SOA_COLUMN(float, x10),
    SOA_COLUMN(float, x11),
    SOA_COLUMN(float, x12),
    SOA_COLUMN(double, x13),
    SOA_COLUMN(double, x14),
    SOA_COLUMN(double, x15),
    SOA_COLUMN(double, x16),
    SOA_COLUMN(double, x17),
    SOA_COLUMN(double, x18),
    SOA_COLUMN(double, x19),
    SOA_COLUMN(double, x20),
    SOA_COLUMN(double, x21),
    SOA_COLUMN(double, x22),
    SOA_COLUMN(double, x23),
    SOA_COLUMN(double, x24),
    SOA_COLUMN(double, x25),
    SOA_COLUMN(int, x26),
    SOA_COLUMN(int, x27),
    SOA_COLUMN(int, x28),
    SOA_COLUMN(int, x29),
    SOA_COLUMN(int, x30),
    SOA_COLUMN(int, x31),
    SOA_COLUMN(int, x32),
    SOA_COLUMN(int, x33),
    SOA_COLUMN(int, x34),
    SOA_COLUMN(int, x35),
    SOA_COLUMN(int, x36),
    SOA_COLUMN(int, x37),
    SOA_COLUMN(int, x38),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, x39),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, x40),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, x41),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, x42),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, x43),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, x44),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, x45),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, x46),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, x47),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, x48),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, x49),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, x50),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x51),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x52),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x53),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x54),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x55),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x56),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x57),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x58),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x59),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x60),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x61),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x62),
    SOA_EIGEN_COLUMN(Eigen::Matrix3d, x63))

using BigSoA = BigSoALayout<>;

GENERATE_SOA_LAYOUT(SoANbodyLayout,
    SOA_COLUMN(float, x),
    SOA_COLUMN(float, y),
    SOA_COLUMN(float, z),
    SOA_COLUMN(float, vx),
    SOA_COLUMN(float, vy),
    SOA_COLUMN(float, vz))

using SoANbody = SoANbodyLayout<>;

GENERATE_SOA_LAYOUT(Stencil,
    SOA_COLUMN(double, src),
    SOA_COLUMN(double, dst),
    SOA_COLUMN(double, rhs))

using SoAStencil = Stencil<>;

GENERATE_SOA_LAYOUT(PxPyPzM,
    SOA_COLUMN(double, x),
    SOA_COLUMN(double, y),
    SOA_COLUMN(double, z),
    SOA_COLUMN(double, M))
using SoAPxPyPzM = PxPyPzM<>;

/// Register Benchmarks ///
template <typename SoA, typename N>
class Fixture1 : public benchmark::Fixture {
 public:
    std::byte *buffer;
    using SoAView = SoA::View;
    SoAView t;

    static constexpr auto n = N::value;

    void SetUp(::benchmark::State &state /* unused */) override
    {
        buffer = reinterpret_cast<std::byte *>(aligned_alloc(Alignment, SoA::computeDataSize(n)));
        SoA soa(buffer, n);
        t = SoAView{soa};
    }

    void TearDown(::benchmark::State &state /* unused */) override { std::free(buffer); }
};

INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyRW, SoA, N_Large);
INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyCompute, SoA, N);
INSTANTIATE_BENCHMARKS_F1(BM_CPURealRW, MediumSoA, N);
INSTANTIATE_BENCHMARKS_F1(BM_CPUHardRW, BigSoA, N);
INSTANTIATE_BENCHMARKS_F1(BM_nbody, SoANbody, N);
INSTANTIATE_BENCHMARKS_F1(BM_stencil, SoAStencil, N_Large);

template <typename SoA1, typename SoA2, typename N>
class Fixture2 : public benchmark::Fixture {
 public:
    std::byte *buffer1, *buffer2;
    using SoAView1 = SoA1::View;
    using SoAView2 = SoA2::View;

    SoAView1 t1;
    SoAView2 t2;

    static constexpr auto n = N::value;

    void SetUp(::benchmark::State &state /* unused */) override
    {
        buffer1 = reinterpret_cast<std::byte *>(aligned_alloc(Alignment, SoA1::computeDataSize(n)));
        buffer2 = reinterpret_cast<std::byte *>(aligned_alloc(Alignment, SoA2::computeDataSize(n)));
        SoA1 soa1(buffer1, n);
        SoA2 soa2(buffer2, n);
        t1 = SoAView1{soa1};
        t2 = SoAView2{soa2};
    }

    void TearDown(::benchmark::State &state /* unused */) override
    {
        std::free(buffer1);
        std::free(buffer2);
    }
};

INSTANTIATE_BENCHMARKS_F2(BM_InvariantMass, SoAPxPyPzM, SoAPxPyPzM, N_Large);

BENCHMARK_MAIN();
