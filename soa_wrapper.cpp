#include <span>

#include "benchmark.h"
#include "wrapper/factory.h"
#include "wrapper/wrapper.h"

#include <Eigen/Core>

template <template <class> class F>
struct S2 {
    template<template <class> class F_out>
    operator S2<F_out>() { return {x0, x1}; }
    F<int> x0, x1;
};

template <template <class> class F>
struct S10 {
    template<template <class> class F_out>
    operator S10<F_out>() { return {x0, x1, x2, x3, x4, x5, x6, x7, x8, x9}; }
    F<float> x0, x1;
    F<double> x2, x3;
    F<int> x4, x5;
    F<Eigen::Vector3d> x6, x7;
    F<Eigen::Matrix3d> x8, x9;
};

template <template <class> class F>
struct S64 {
    template<template <class> class F_out>
    operator S64<F_out>() { return {
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,
        x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25,
        x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38,
        x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50,
        x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63
    }; }
    F<float> x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
    F<double> x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25;
    F<int> x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38;
    F<Eigen::Vector3d> x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50;
    F<Eigen::Matrix3d> x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63;
};

int main(int argc, char** argv) {

    std::vector<std::byte *> buffer_pointers;

    for (std::size_t n : N) {
        std::size_t bytes = n * 2 * sizeof(int);
        buffer_pointers.emplace_back(new std::byte[bytes]);
        auto t2b = factory::buffer_wrapper<S2, wrapper::layout::soa>(buffer_pointers.back(), bytes);
        using wrapper_type = wrapper::wrapper<std::span, S2, wrapper::layout::soa>;
        wrapper_type t_span(t2b);
        benchmark::RegisterBenchmark("BM_CPUEasyRW", BM_CPUEasyRW<wrapper_type>, t_span)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    for (std::size_t n : N) {
        std::size_t bytes = n * 2 * sizeof(int);
        buffer_pointers.emplace_back(new std::byte[bytes]);
        auto t2b = factory::buffer_wrapper<S2, wrapper::layout::soa>(buffer_pointers.back(), bytes);
        using wrapper_type = wrapper::wrapper<std::span, S2, wrapper::layout::soa>;
        wrapper_type t_span(t2b);
        benchmark::RegisterBenchmark("BM_CPUEasyCompute", BM_CPUEasyCompute<wrapper_type>, t_span)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    for (std::size_t n : N) {
        std::size_t bytes = n * 2 * (sizeof(float) + sizeof(double) + sizeof(int) + sizeof(Eigen::Vector3d) + sizeof(Eigen::Matrix3d));
        buffer_pointers.emplace_back(new std::byte[bytes]);
        auto t10 = factory::buffer_wrapper<S10, wrapper::layout::soa>(buffer_pointers.back(), bytes);
        using wrapper_type = wrapper::wrapper<std::span, S10, wrapper::layout::soa>;
        wrapper_type t_span(t10);
        benchmark::RegisterBenchmark("BM_CPURealRW", BM_CPURealRW<wrapper_type>, t_span)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    for (std::size_t n : N) {
        std::size_t bytes = n * (13 * (sizeof(float) + sizeof(double) + sizeof(int) + sizeof(Eigen::Matrix3d)) + 12 * sizeof(Eigen::Vector3d));
        buffer_pointers.emplace_back(new std::byte[bytes]);
        auto t64 = factory::buffer_wrapper<S64, wrapper::layout::soa>(buffer_pointers.back(), bytes);
        using wrapper_type = wrapper::wrapper<std::span, S64, wrapper::layout::soa>;
        wrapper_type t_span(t64);
        benchmark::RegisterBenchmark("BM_CPUHardRW", BM_CPUHardRW<wrapper_type>, t_span)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    for (std::byte * buffer_ptr : buffer_pointers)  std::free(buffer_ptr);
}