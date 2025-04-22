#define SOA_BOOST

#include <memory>

#include "benchmark.h"
#include "boost/SoALayout.h"
#include <Eigen/Core>

GENERATE_SOA_LAYOUT(SoALayout,
    SOA_COLUMN(int, x0),
    SOA_COLUMN(int, x1))

using SoA = SoALayout<>;
using SoAView = SoA::View;

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
using MediumSoAView = MediumSoA::View;

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
using BigSoAView = BigSoA::View;

int main(int argc, char** argv) {
    std::vector<void *> free_list;

    // Seperate loops to sort the output by benchmark.
    for (size_t n : N) {
        auto buffer = reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, SoA::computeDataSize(n)));
        SoA soa(buffer, n);
        SoAView soaView{soa};
        benchmark::RegisterBenchmark("BM_CPUEasyRW", BM_CPUEasyRW<SoAView>, soaView)->Arg(n)->Unit(benchmark::kMillisecond);
        free_list.push_back(buffer);
    }

    for (auto n : N) {
        auto buffer = reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, SoA::computeDataSize(n)));
        SoA fullsoa(buffer, n);
        SoAView fullsoaView{fullsoa};
        benchmark::RegisterBenchmark("BM_CPUEasyCompute", BM_CPUEasyCompute<SoAView>, fullsoaView)->Arg(n)->Unit(benchmark::kMillisecond);
        free_list.push_back(buffer);
    }

    for (auto n : N) {
        auto buffer = reinterpret_cast<std::byte *>(aligned_alloc(MediumSoA::alignment, MediumSoA::computeDataSize(n)));
        MediumSoA mediumsoa(buffer, n);
        MediumSoAView mediumsoaView{mediumsoa};
        benchmark::RegisterBenchmark("BM_CPURealRW", BM_CPURealRW<MediumSoAView>, mediumsoaView)->Arg(n)->Unit(benchmark::kMillisecond);
        free_list.push_back(buffer);
    }

    for (auto n : N) {
        auto buffer = reinterpret_cast<std::byte *>(aligned_alloc(BigSoA::alignment, BigSoA::computeDataSize(n)));
        BigSoA bigSoa(buffer, n);
        BigSoAView bigSoaView{bigSoa};
        benchmark::RegisterBenchmark("BM_CPUHardRW", BM_CPUHardRW<BigSoAView>, bigSoaView)->Arg(n)->Unit(benchmark::kMillisecond);
        free_list.push_back(buffer);
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    for (auto buffer : free_list) {
        std::free(buffer);
    }
}