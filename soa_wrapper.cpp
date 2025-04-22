#include <vector>
#include <format>

#include "benchmark.h"
#include "wrapper/factory.h"
#include "wrapper/wrapper.h"

#include <Eigen/Core>

template <template <class> class F>
struct S2 {
    F<int> x0, x1;
};

template <template <class> class F>
struct S10 {
    F<float> x0, x1;
    F<double> x2, x3;
    F<int> x4, x5;
    F<Eigen::Vector3d> x6, x7;
    F<Eigen::Matrix3d> x8, x9;
};

template <template <class> class F>
struct S64 {
    F<double> x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
    F<float> x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25;
    F<int> x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38;
    F<Eigen::Vector3d> x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50;
    F<Eigen::Matrix3d> x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63;
};

template <class T>
using my_vector = std::vector<T>;

int main(int argc, char** argv) {
    // Seperate loops to sort the output by benchmark.
    for (auto n : N) {
        auto t2a = factory::default_wrapper<my_vector, S2, wrapper::layout::soa>(n);
        benchmark::RegisterBenchmark(std::format("BM_CPUEasyRW/{}", n), BM_CPUEasyRW<decltype(t2a)>, t2a)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    for (auto n : N) {
        auto t2b = factory::default_wrapper<my_vector, S2, wrapper::layout::soa>(n);
        benchmark::RegisterBenchmark(std::format("BM_CPUEasyCompute/{}", n), BM_CPUEasyCompute<decltype(t2b)>, t2b)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    for (auto n : N) {
        auto t10 = factory::default_wrapper<my_vector, S10, wrapper::layout::soa>(n);
        benchmark::RegisterBenchmark(std::format("BM_CPURealRW/{}", n), BM_CPURealRW<decltype(t10)>, t10)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    for (auto n : N) {
        auto t64 = factory::default_wrapper<my_vector, S64, wrapper::layout::soa>(n);
        benchmark::RegisterBenchmark(std::format("BM_CPUHardRW/{}", n), BM_CPUHardRW<decltype(t64)>, t64)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}