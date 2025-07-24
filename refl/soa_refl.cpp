
#define __cpp_lib_reflection 20250130 // eccp version

#define EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO 0
// #define RMPP_DEBUG

#include "benchmark.h"
#include <Eigen/Core>
#include "rmpp.h"
#include <vector>

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

struct S64 {
    double &x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7, &x8, &x9, &x10, &x11, &x12;
    float &x13, &x14, &x15, &x16, &x17, &x18, &x19, &x20, &x21, &x22, &x23, &x24, &x25;
    int &x26, &x27, &x28, &x29, &x30, &x31, &x32, &x33, &x34, &x35, &x36, &x37, &x38;
    Eigen::Vector3d &x39, &x40, &x41, &x42, &x43, &x44, &x45, &x46, &x47, &x48, &x49, &x50;
    Eigen::Matrix3d &x51, &x52, &x53, &x54, &x55, &x56, &x57, &x58, &x59, &x60, &x61, &x62, &x63;
};

int main(int argc, char **argv)
{
    // Seperate loops to sort the output by benchmark.
    for (auto n : N) {
        using SoA = rmpp::AoS2SoA<S2, 64>;
        auto byte_size = SoA::ComputeSize(n);
        auto buffer = reinterpret_cast<std::byte *>(aligned_alloc(64, byte_size));
        SoA t(buffer, byte_size, n);
        benchmark::RegisterBenchmark("BM_CPUEasyRW", BM_CPUEasyRW<SoA>, t)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    for (auto n : N) {
        using SoA = rmpp::AoS2SoA<S2, 64>;
        auto byte_size = SoA::ComputeSize(n);
        auto buffer = reinterpret_cast<std::byte *>(aligned_alloc(64, byte_size));
        SoA t(buffer, byte_size, n);
        benchmark::RegisterBenchmark("BM_CPUEasyCompute", BM_CPUEasyCompute<SoA>,
        t)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    for (auto n : N) {
        using SoA = rmpp::AoS2SoA<S10, 64>;
        auto byte_size = SoA::ComputeSize(n);
        auto buffer = reinterpret_cast<std::byte *>(aligned_alloc(64, byte_size));
        SoA t(buffer, byte_size, n);
        benchmark::RegisterBenchmark("BM_CPURealRW", BM_CPURealRW<SoA>, t)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    for (auto n : N) {
        using SoA = rmpp::AoS2SoA<S64, 64>;
        auto byte_size = SoA::ComputeSize(n);
        auto buffer = reinterpret_cast<std::byte *>(aligned_alloc(64, byte_size));
        SoA t(buffer, byte_size, n);
        benchmark::RegisterBenchmark("BM_CPUHardRW", BM_CPUHardRW<SoA>, t)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
