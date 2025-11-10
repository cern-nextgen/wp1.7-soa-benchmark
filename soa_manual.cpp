#define SOA_MANUAL

#include <vector>
#include <span>

#include "benchmark.h"

#include <iostream>

constexpr inline size_t align_size(size_t size)
{
    return ((size + Alignment - 1) / Alignment) * Alignment;
}

#include "snbody.h"

#include "sstencil.h"

#include "pxpypzm.h"

template <typename S, typename N>
class Fixture1 : public benchmark::Fixture {
 public:
    std::byte *buffer;
    S t;
    static constexpr auto n = N::value;

    void SetUp(::benchmark::State &state) override
    {
        buffer = reinterpret_cast<std::byte *__restrict__>(std::aligned_alloc(Alignment, S::size_bytes(n)));
        t = S(reinterpret_cast<std::byte *>(buffer), n);
    }

    void TearDown(::benchmark::State &state) override { std::free(buffer); }

    Fixture1() : t(nullptr, 0) {}
};

template <typename S1, typename S2, typename N>
class Fixture2 : public benchmark::Fixture {
 public:
    void *buffer1, *buffer2;
    S1 t1;
    S2 t2;

    static constexpr auto n = N::value;

    void SetUp(::benchmark::State &state) override
    {
        buffer1 = reinterpret_cast<std::byte *__restrict__>(std::aligned_alloc(Alignment, S1::size_bytes(n)));
        buffer2 = reinterpret_cast<std::byte *__restrict__>(std::aligned_alloc(Alignment, S2::size_bytes(n)));
        t1 = S1(reinterpret_cast<std::byte *>(buffer1), n);
        t2 = S2(reinterpret_cast<std::byte *>(buffer2), n);
    }

    void TearDown(::benchmark::State &state) override
    {
        std::free(buffer1);
        std::free(buffer2);
    }

    Fixture2() : t1(nullptr, 0), t2(nullptr, 0) {}
};

BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM_nbody, Snbody, std::integral_constant<size_t, N_nbody>)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM_stencil, Sstencil, std::integral_constant<size_t, N_stencil>)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, BM_InvariantMass, PxPyPzM, PxPyPzM, std::integral_constant<size_t, N_im>)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
