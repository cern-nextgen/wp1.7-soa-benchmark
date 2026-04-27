#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <format>
#include <sstream>
#include <string>
#include <cstddef>

#ifdef SOA_MANUAL
#define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ.MEMBER[INDEX]
#elif defined(AOS_MANUAL)
#define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ[INDEX].MEMBER
#else
#define MEMBER_ACCESS(OBJ, MEMBER, INDEX) OBJ[INDEX].MEMBER
#endif

using Vector3D = Eigen::Vector3d;
using Matrix3D = Eigen::Matrix3d;

constexpr std::size_t N[]       = {10, 100, 1000, 10000, 100000};
//constexpr std::size_t N_medium[] = {1<<12, 1<<16, 1<<20, 1<<24, 1<<28};
constexpr std::size_t N_Large[] = {10000, 100000, 1000000, 10000000, 100000000};
constexpr size_t Alignment = 128;

// clang-format off
#define INSTANTIATE_BENCHMARKS_F1(BM, Type, Sizes) \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM, Type, std::integral_constant<size_t, Sizes[0]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM, Type, std::integral_constant<size_t, Sizes[1]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM, Type, std::integral_constant<size_t, Sizes[2]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM, Type, std::integral_constant<size_t, Sizes[3]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM, Type, std::integral_constant<size_t, Sizes[4]>)->Unit(benchmark::kMillisecond);

#define INSTANTIATE_BENCHMARKS_F2(BM, Type1, Type2, Sizes) \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, BM, Type1, Type2, std::integral_constant<size_t, Sizes[0]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, BM, Type1, Type2, std::integral_constant<size_t, Sizes[1]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, BM, Type1, Type2, std::integral_constant<size_t, Sizes[2]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, BM, Type1, Type2, std::integral_constant<size_t, Sizes[3]>)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, BM, Type1, Type2, std::integral_constant<size_t, Sizes[4]>)->Unit(benchmark::kMillisecond);
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
        state.SkipWithError(
            std::format("Wrong result in {}: expected {}, got {}", member_name, ToString(expected), ToString(actual)));
    }
}

template <typename S, typename N>
class Fixture1;

template <typename T1, typename T2, typename N>
class Fixture2;
