#include <vector>

#include "benchmark.h"
#include "wrapper/wrapper.h"

#include <Eigen/Core>

using Vector3D = Eigen::Vector3d;
using Matrix3D = Eigen::Matrix3d;

template <template <class> class F>
struct S2 {
    F<int> x0,  x1;
};

template <template <class> class F>
struct S10 {
    F<float> x0, x1;
    F<double> x2, x3;
    F<int> x4, x5;
    F<Vector3D> x6, x7;
    F<Matrix3D> x8, x9;
};

template <template <class> class F>
struct S64 {
    F<double> x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13;
    F<float> x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26;
    F<int> x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39
    F<Vector3D> x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51;
    F<Matrix3D> x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63;
};

template <class T>
using my_vector = std::vector<T>;

constexpr std::size_t BM_CPUEasyRW_nelem = 10;
auto t0 = factory::default_wrapper<my_vector, S2, wrapper::layout::soa>(BM_CPUEasyRW_nelem);
BENCHMARK_CAPTURE(BM_CPUEasyRW, wrapper, t0);

constexpr std::size_t BM_CPUEasyCompute_nelem = 10;
auto t1 = factory::default_wrapper<my_vector, S2, wrapper::layout::soa>(BM_CPUEasyCompute_nelem);
BENCHMARK_CAPTURE(BM_CPUEasyCompute, wrapper, t1);

/*constexpr std::size_t BM_CPUHardRW_nelem = 10e4;
auto t2 = factory::default_wrapper<my_vector, S64, wrapper::layout::soa>(BM_CPUHardRW_nelem);
BENCHMARK_CAPTURE(BM_CPUHardRW, wrapper, t2);*/

constexpr std::size_t BM_CPURealRW_nelem = 100000;
auto t3 = factory::default_wrapper<my_vector, S10, wrapper::layout::soa>(BM_CPURealRW_nelem);
BENCHMARK_CAPTURE(BM_CPURealRW, wrapper, t3);

BENCHMARK_MAIN();
