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

/// Register Benchmarks ///
template <typename wrapper_span, typename N>
class Fixture1 : public benchmark::Fixture {
 public:
    static constexpr auto n = N::value;

    template <template <class> class F_type>
    using S = wrapper_span::template S_type<F_type>;

    std::byte *buffer;
    wrapper_span t;

    void SetUp(::benchmark::State &state) override
    {
        std::size_t bytes = n * factory::get_size_in_bytes<S, L>();
        buffer = new std::byte[bytes];
        auto s = factory::buffer_wrapper<S, L>(buffer, bytes);
        t = static_cast<wrapper_span>(s);
    }

    void TearDown(::benchmark::State &state) override { std::free(buffer); }
};

using S2span = wrapper::wrapper<S2, std::span, L>;
using S10span = wrapper::wrapper<S10, std::span, L>;
using S64span = wrapper::wrapper<S64, std::span, L>;
using Snbodyspan = wrapper::wrapper<Snbody, std::span, L>;
using Sstencilspan = wrapper::wrapper<Sstencil, std::span, L>;
using PxPyPzMspan = wrapper::wrapper<PxPyPzM, std::span, L>;

INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyRW, S2span, N_Large);
INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyCompute, S2span, N);
INSTANTIATE_BENCHMARKS_F1(BM_CPURealRW, S10span, N);
INSTANTIATE_BENCHMARKS_F1(BM_CPUHardRW, S64span, N);
INSTANTIATE_BENCHMARKS_F1(BM_nbody, Snbodyspan, N);
INSTANTIATE_BENCHMARKS_F1(BM_stencil, Sstencilspan, N_Large);

template <typename wrapper_span1, typename wrapper_span2, typename N>
class Fixture2 : public benchmark::Fixture {
 public:
    static constexpr auto n = N::value;

    template <template <class> class F_type>
    using S1 = wrapper_span1::template S_type<F_type>;

    template <template <class> class F_type>
    using S2 = wrapper_span2::template S_type<F_type>;

    std::byte *buffer1, *buffer2;
    wrapper_span1 t1;
    wrapper_span2 t2;

    void SetUp(::benchmark::State &state) override
    {
        std::size_t bytes1 = n * factory::get_size_in_bytes<S1, L>();
        buffer1 = new std::byte[bytes1];
        std::size_t bytes2 = n * factory::get_size_in_bytes<S2, L>();
        buffer2 = new std::byte[bytes2];

        auto s1 = factory::buffer_wrapper<S1, L>(buffer1, bytes1);
        auto s2 = factory::buffer_wrapper<S2, L>(buffer2, bytes2);
        t1 = static_cast<wrapper_span1>(s1);
        t2 = static_cast<wrapper_span2>(s2);
    }

    void TearDown(::benchmark::State &state) override { std::free(buffer1); std::free(buffer2); }
};

BENCHMARK_MAIN();

INSTANTIATE_BENCHMARKS_F2(BM_InvariantMass, PxPyPzMspan, PxPyPzMspan, N_Large);
