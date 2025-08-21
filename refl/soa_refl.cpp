
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

struct Snbody {
    float &x, &y, &z, &vx, &vy, &vz;
};

struct Sstencil {
    double &src, &dst, &rhs;
};

struct PxPyPzM {
    double &x, &y, &z, &M;
};

template <typename S>
void RegisterBenchmarkHelper(const char* name, auto bm_func, auto& free_list, auto &N)
{
    for (auto n : N) {
        using SoA = rmpp::AoS2SoA<S, 64>;
        auto byte_size = SoA::ComputeSize(n);
        auto buffer = reinterpret_cast<std::byte *>(aligned_alloc(64, byte_size));
        SoA t(buffer, byte_size, n);
        benchmark::RegisterBenchmark(name, bm_func, t)->Arg(n)->Unit(benchmark::kMillisecond);
        free_list.push_back(buffer);
    }
}

int main(int argc, char **argv)
{
    std::vector<std::byte *> free_list;

    // Separate loops to sort the output by benchmark.
    RegisterBenchmarkHelper<S2>("BM_CPUEasyRW", BM_CPUEasyRW<rmpp::AoS2SoA<S2, 64>>, free_list, N_Large);
    RegisterBenchmarkHelper<S2>("BM_CPUEasyCompute", BM_CPUEasyCompute<rmpp::AoS2SoA<S2, 64>>, free_list, N);
    RegisterBenchmarkHelper<S10>("BM_CPURealRW", BM_CPURealRW<rmpp::AoS2SoA<S10, 64>>, free_list, N);
    RegisterBenchmarkHelper<S64>("BM_CPUHardRW", BM_CPUHardRW<rmpp::AoS2SoA<S64, 64>>, free_list, N);
    RegisterBenchmarkHelper<Snbody>("BM_nbody", BM_nbody<rmpp::AoS2SoA<Snbody, 64>>, free_list, N);
    RegisterBenchmarkHelper<Sstencil>("BM_stencil", BM_stencil<rmpp::AoS2SoA<Sstencil, 64>>, free_list, N_Large);

    for (auto n : N_Large) {
        using SoA = rmpp::AoS2SoA<PxPyPzM, 64>;
        auto byte_size = SoA::ComputeSize(n);
        auto buffer1 = reinterpret_cast<std::byte *>(aligned_alloc(64, byte_size));
        auto buffer2 = reinterpret_cast<std::byte *>(aligned_alloc(64, byte_size));
        SoA t1(buffer1, byte_size, n);
        SoA t2(buffer2, byte_size, n);
        benchmark::RegisterBenchmark("BM_InvariantMass", BM_InvariantMass<SoA, SoA>, t1, t2)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    for (std::byte * buffer_ptr : free_list)  std::free(buffer_ptr);
}
