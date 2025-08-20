#include <span>

#include "benchmark.h"
#include "wrapper/factory.h"
#include "wrapper/wrapper.h"

#include <Eigen/Core>

template <template <class> class F>
struct S2 {
    template<template <class> class F_new>
    operator S2<F_new>() { return {x0, x1}; }
    F<int> x0, x1;
};

template <template <class> class F>
struct S10 {
    template <template <class> class F_new>
    operator S10<F_new>() { return {x0, x1, x2, x3, x4, x5, x6, x7, x8, x9}; }
    F<float> x0, x1;
    F<double> x2, x3;
    F<int> x4, x5;
    F<Eigen::Vector3d> x6, x7;
    F<Eigen::Matrix3d> x8, x9;
};

template <template <class> class F>
struct S64 {
    template <template <class> class F_new>
    operator S64<F_new>() { return {
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

template <template <class> class F>
struct Snbody {
    template<template <class> class F_new>
    operator Snbody<F_new>() { return {x, y, z, vx, vy, vz}; }
    F<float> x, y, z, vx, vy, vz;
};

template <template <class> class F>
struct Sstencil {
    template<template <class> class F_new>
    operator Sstencil<F_new>() { return {src, dst, rhs}; }
    F<double> src, dst, rhs;
};

template <template <class> class F>
struct PxPyPzM {
    template<template <class> class F_new>
    operator PxPyPzM<F_new>() { return {x, y, z, M}; }
    F<double> x, y, z, M;
};

constexpr wrapper::layout L = wrapper::layout::soa;

template <template <template <class> class> class S>
void RegisterBenchmarkHelper(const char* name, auto bm_func, std::vector<std::byte*>& buffer_pointers, auto &N) {
    for (auto n : N) {
        std::size_t bytes = n * factory::get_size_in_bytes<S, L>();
        buffer_pointers.emplace_back(new std::byte[bytes]);
        auto t = factory::buffer_wrapper<S, L>(buffer_pointers.back(), bytes);
        using wrapper_type = wrapper::wrapper<S, std::span, L>;
        wrapper_type t_span(t);
        benchmark::RegisterBenchmark(name, bm_func, t_span)->Arg(n)->Unit(benchmark::kMillisecond);
    }
}

int main(int argc, char** argv) {

    std::vector<std::byte *> buffer_pointers;

    RegisterBenchmarkHelper<S2>("BM_CPUEasyRW", BM_CPUEasyRW<wrapper::wrapper<S2, std::span, L>>, buffer_pointers, N_Large);
    RegisterBenchmarkHelper<S2>("BM_CPUEasyCompute", BM_CPUEasyCompute<wrapper::wrapper<S2, std::span, L>>, buffer_pointers, N);
    RegisterBenchmarkHelper<S10>("BM_CPURealRW", BM_CPURealRW<wrapper::wrapper<S10, std::span, L>>, buffer_pointers, N);
    RegisterBenchmarkHelper<S64>("BM_CPUHardRW", BM_CPUHardRW<wrapper::wrapper<S64, std::span, L>>, buffer_pointers, N);
    RegisterBenchmarkHelper<Snbody>("BM_nbody", BM_nbody<wrapper::wrapper<Snbody, std::span, L>>, buffer_pointers, N);
    RegisterBenchmarkHelper<Sstencil>("BM_stencil", BM_stencil<wrapper::wrapper<Sstencil, std::span, L>>, buffer_pointers, N_Large);

    for (std::size_t n : N_Large) {
        std::size_t bytes = n * factory::get_size_in_bytes<PxPyPzM, L>();
        auto buffer1 = new std::byte[bytes];
        auto buffer2 = new std::byte[bytes];
        auto tpxpypxm1 = factory::buffer_wrapper<PxPyPzM, L>(buffer1, bytes);
        auto tpxpypxm2 = factory::buffer_wrapper<PxPyPzM, L>(buffer2, bytes);
        using wrapper_type = wrapper::wrapper<PxPyPzM, std::span, L>;
        wrapper_type t_span1(tpxpypxm1);
        wrapper_type t_span2(tpxpypxm2);
        benchmark::RegisterBenchmark("BM_InvariantMass", BM_InvariantMass<wrapper_type, wrapper_type>,
                                     t_span1, t_span2)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    for (std::byte * buffer_ptr : buffer_pointers)  std::free(buffer_ptr);
}
