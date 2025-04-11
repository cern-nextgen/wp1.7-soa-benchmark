#include <vector>

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
    F<double> x0, x1, x2;
    F<float> x3, x4, x5;
    F<int> x6, x7;
    F<Eigen::Vector3d> x8;
    F<Eigen::Matrix3d> x9;
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

constexpr std::size_t N0 = 10;
auto t0 = factory::default_wrapper<my_vector, S2, wrapper::layout::soa>(N0);
BENCHMARK_CAPTURE(BM_CPUEasyRW, wrapper, t0);

constexpr std::size_t N1 = 10;
auto t1 = factory::default_wrapper<my_vector, S2, wrapper::layout::soa>(N1);
BENCHMARK_CAPTURE(BM_CPUEasyCompute, wrapper, t1);

constexpr std::size_t N2 = 10e4;
auto t2 = factory::default_wrapper<my_vector, S64, wrapper::layout::soa>(N2);
BENCHMARK_CAPTURE(BM_CPUHardRW, wrapper, t2);

constexpr std::size_t N3 = 10e5;
auto t3 = factory::default_wrapper<my_vector, S10, wrapper::layout::soa>(N3);
BENCHMARK_CAPTURE(BM_CPURealRW, wrapper, t3);

BENCHMARK_MAIN();
