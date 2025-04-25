#define SOA_MANUAL

#include <vector>

#include "benchmark.h"
#include <Eigen/Core>

struct S2 {
    std::vector<int> x0, x1;

    S2(size_t n) : x0(n), x1(n) {}
};

struct S10 {
    std::vector<float> x0, x1;
    std::vector<double> x2, x3;
    std::vector<int> x4, x5;
    std::vector<Eigen::Vector3d> x6, x7;
    std::vector<Eigen::Matrix3d> x8, x9;

    S10(size_t n) : x0(n), x1(n), x2(n), x3(n), x4(n), x5(n), x6(n), x7(n), x8(n), x9(n) {}
};

struct S64 {
    std::vector<double> x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
    std::vector<float> x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25;
    std::vector<int> x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38;
    std::vector<Eigen::Vector3d> x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50;
    std::vector<Eigen::Matrix3d> x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63;

    S64(size_t n) : x0(n), x1(n), x2(n), x3(n), x4(n), x5(n), x6(n), x7(n), x8(n), x9(n),
        x10(n), x11(n), x12(n), x13(n), x14(n), x15(n), x16(n), x17(n), x18(n), x19(n),
        x20(n), x21(n), x22(n), x23(n), x24(n), x25(n), x26(n), x27(n), x28(n), x29(n),
        x30(n), x31(n), x32(n), x33(n), x34(n), x35(n), x36(n), x37(n), x38(n), x39(n),
        x40(n), x41(n), x42(n), x43(n), x44(n), x45(n), x46(n), x47(n), x48(n), x49(n),
        x50(n), x51(n), x52(n), x53(n), x54(n), x55(n), x56(n), x57(n), x58(n), x59(n),
        x60(n), x61(n), x62(n), x63(n) {}
};

int main(int argc, char** argv) {
    // Seperate loops to sort the output by benchmark.
    for (auto n : N) {
        S2 t2a(n);
        benchmark::RegisterBenchmark("BM_CPUEasyRW", BM_CPUEasyRW<S2>, t2a)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    for (auto n : N) {
        S2 t2b(n);
        benchmark::RegisterBenchmark("BM_CPUEasyCompute", BM_CPUEasyCompute<S2>, t2b)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    for (auto n : N) {
        S10 t10(n);
        benchmark::RegisterBenchmark("BM_CPURealRW", BM_CPURealRW<S10>, t10)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    for (auto n : N) {
        S64 t64(n);
        benchmark::RegisterBenchmark("BM_CPUHardRW", BM_CPUHardRW<S64>, t64)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}