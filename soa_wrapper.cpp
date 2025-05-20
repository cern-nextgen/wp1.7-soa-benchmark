#include <array>

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

template <template <class, std::size_t> class F, std::size_t n>
struct FixSize {
    template <class T>
    using type = F<T, n>;
};

template<template <template <class> class> class S, std::size_t Begin, std::size_t End, class F>
constexpr void constexpr_for(F&& f) {
  if constexpr (Begin < End) {
    constexpr int n = N[Begin];
    wrapper::wrapper<FixSize<std::array, n>::template type, S, wrapper::layout::soa> t;
    f(t, n);
    constexpr_for<S, Begin + 1, End>(std::forward<F>(f));
  }
}

int main(int argc, char** argv) {

    auto f_BM_CPUEasyRW = [] (auto t, std::size_t n) {
        benchmark::RegisterBenchmark("BM_CPUEasyRW", BM_CPUEasyRW<decltype(t)>, t)->Arg(n)->Unit(benchmark::kMillisecond);
    };
    constexpr_for<S2, 0, 5>(f_BM_CPUEasyRW);

    auto f_BM_CPUEasyCompute = [] (auto t, std::size_t n) {
        benchmark::RegisterBenchmark("BM_CPUEasyCompute", BM_CPUEasyCompute<decltype(t)>, t)->Arg(n)->Unit(benchmark::kMillisecond);
    };
    constexpr_for<S2, 0, 5>(f_BM_CPUEasyCompute);

    auto f_BM_CPURealRW = [] (auto t, std::size_t n) {
        benchmark::RegisterBenchmark("BM_CPURealRW", BM_CPURealRW<decltype(t)>, t)->Arg(n)->Unit(benchmark::kMillisecond);
    };
    constexpr_for<S10, 0, 5>(f_BM_CPURealRW);

    auto f_BM_CPUHardRW = [] (auto t, std::size_t n) {
        benchmark::RegisterBenchmark("BM_CPUHardRW", BM_CPUHardRW<decltype(t)>, t)->Arg(n)->Unit(benchmark::kMillisecond);
    };
    constexpr_for<S64, 0, 5>(f_BM_CPUHardRW);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}