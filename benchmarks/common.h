#ifndef BENCHMARKS_COMMON_H
#define BENCHMARKS_COMMON_H

#include <Eigen/Core>
#include <benchmark/benchmark.h>
#include <sstream>
#include <string>
#include <cstddef>
#include "benchmarks/backend.h"

#ifdef SOA_MANUAL
#define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ.MEMBER[INDEX]
#elif defined(AOS_MANUAL)
#define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ[INDEX].MEMBER
#else
#define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ[INDEX].MEMBER
#endif

constexpr std::size_t N_small_cpu[] = {100, 1000, 10000, 100000, 1000000};
constexpr std::size_t N_large_cpu[] = {10000, 100000, 1000000, 10000000, 100000000};
// GPU sizes: shifted up vs. CPU (kernel launch overhead dominates at small n);
// upper end capped to fit in ~42 GB VRAM for the largest structs (S64 ~1.4 KB/elem
// for HardRW, S32 ~128 B/elem for Strided, InvariantMass holds two PxPyPzM arrays).
constexpr std::size_t N_small_gpu[] = {100, 1000, 10000, 100000, 1000000};
constexpr std::size_t N_large_gpu[] = {10000, 100000, 1000000, 10000000, 100000000};

// clang-format off
#define INSTANTIATE_BENCHMARKS_F1(Benchmark, Type, Sizes, Backend) \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, Benchmark, Type, std::integral_constant<std::size_t, Sizes[0]>, Backend)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, Benchmark, Type, std::integral_constant<std::size_t, Sizes[1]>, Backend)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, Benchmark, Type, std::integral_constant<std::size_t, Sizes[2]>, Backend)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, Benchmark, Type, std::integral_constant<std::size_t, Sizes[3]>, Backend)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, Benchmark, Type, std::integral_constant<std::size_t, Sizes[4]>, Backend)->Unit(benchmark::kMillisecond);

#define INSTANTIATE_BENCHMARKS_F2(Benchmark, Type1, Type2, Sizes, Backend) \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, Benchmark, Type1, Type2, std::integral_constant<std::size_t, Sizes[0]>, Backend)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, Benchmark, Type1, Type2, std::integral_constant<std::size_t, Sizes[1]>, Backend)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, Benchmark, Type1, Type2, std::integral_constant<std::size_t, Sizes[2]>, Backend)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, Benchmark, Type1, Type2, std::integral_constant<std::size_t, Sizes[3]>, Backend)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, Benchmark, Type1, Type2, std::integral_constant<std::size_t, Sizes[4]>, Backend)->Unit(benchmark::kMillisecond);
// clang-format on

template <typename T>
static std::string ToString(const T &obj)
{
    std::stringstream ss;
    ss << obj;
    return ss.str();
}

template <typename Expected, typename Actual>
void CheckResult(benchmark::State &state, const Expected &expected, const Actual &actual,
                 const std::string &member_name)
{
    if (expected != actual) {
        std::stringstream ss;
        ss << "Wrong result in " << member_name << ": expected " << ToString(expected) << ", got " << ToString(actual);
        state.SkipWithError(ss.str());
    }
}

#endif // BENCHMARKS_COMMON_H
