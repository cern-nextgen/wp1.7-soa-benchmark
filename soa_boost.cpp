#define SOA_BOOST 

#include "benchmark.h"
#include "boost/SoALayout.h"
#include <Eigen/Dense>
#include <Eigen/Core>

GENERATE_SOA_LAYOUT(SoALayout,
    SOA_COLUMN(int, x0),
    SOA_COLUMN(int, x1))

using SoA = SoALayout<>;
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;

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
using BigSoAConstView = BigSoA::ConstView;  

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
using MediumSoAConstView = MediumSoA::ConstView;    

std::unique_ptr<std::byte, decltype(std::free) *> buffer{
    reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, SoA::computeDataSize(BM_CPUEasyRW_nelem))), std::free};
SoA soa(buffer.get(), BM_CPUEasyRW_nelem);
SoAView soaView{soa};
SoAConstView soaConstView{soa};

std::unique_ptr<std::byte, decltype(std::free) *> fullbuffer{
    reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, SoA::computeDataSize(BM_CPUEasyCompute_nelem))), std::free};
SoA fullsoa(fullbuffer.get(), BM_CPUEasyCompute_nelem);
SoAView fullsoaView{fullsoa};
SoAConstView fullsoaConstView{fullsoa};

std::unique_ptr<std::byte, decltype(std::free) *> medbuffer{
    reinterpret_cast<std::byte *>(aligned_alloc(MediumSoA::alignment, MediumSoA::computeDataSize(100000))), std::free};
MediumSoA mediumsoa(medbuffer.get(), 100000);
MediumSoAView mediumsoaView{mediumsoa};
MediumSoAConstView mediumsoaConstView{mediumsoa};

std::unique_ptr<std::byte, decltype(std::free) *> bigbuffer{
    reinterpret_cast<std::byte *>(aligned_alloc(BigSoA::alignment, BigSoA::computeDataSize(BM_CPUHardRW_nelem))), std::free};
BigSoA bigSoa(bigbuffer.get(), BM_CPUHardRW_nelem);
BigSoAView bigSoaView{bigSoa};
BigSoAConstView bigSoaConstView{bigSoa};

BENCHMARK_CAPTURE(BM_CPUEasyRW, cms_soa, soaView);

BENCHMARK_CAPTURE(BM_CPUEasyCompute, cms_full_soa, fullsoaView);

BENCHMARK_CAPTURE(BM_CPURealRW, cms_medium_soa, mediumsoaView);

BENCHMARK_CAPTURE(BM_CPUHardRW, cms_big_soa, bigSoaView);

BENCHMARK_MAIN();
