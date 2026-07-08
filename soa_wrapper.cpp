#include <type_traits>

#include "benchmark.h"
#include "memlayout/wrapper.h"

#include <Eigen/Core>

template <class T>
T* malloc_helper(std::size_t n) {
    return reinterpret_cast<T*>(std::malloc(n * sizeof(T)));
}

struct mallocator {
    std::size_t n;
    template <class ...Args>
    void operator()(Args*& ...args) const { ((args = malloc_helper<Args>(n)), ...); }
};

struct deallocator {
    template <class ...Args>
    void operator()(Args*& ...args) const { ((std::free(args), args = nullptr), ...); }
};

template <class ArrayType>
void allocate(ArrayType& w, std::size_t n) {
    if constexpr (ArrayType::layout_type == memlayout::Layout::aos) {
        using value_type = std::remove_pointer<typename ArrayType::Data>::type;
        w.data = malloc_helper<value_type>(n);
    } else {    
        w.apply(mallocator{n});
    }
}

template <class ArrayType>
void deallocate(ArrayType& w) {
    if constexpr (ArrayType::layout_type == memlayout::Layout::aos) {
        std::free(w.data);
    } else {    
        w.apply(deallocator{});
    }
};

template <template <class> class F>
struct S2 {
    MEMLAYOUT_APPLY(S2, x0, x1)
    F<int> x0, x1;
};

template <template <class> class F>
struct S10 {
    MEMLAYOUT_APPLY(S10, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)
    F<float> x0, x1;
    F<double> x2, x3;
    F<int> x4, x5;
    F<Eigen::Vector3d> x6, x7;
    F<Eigen::Matrix3d> x8, x9;
};

template <template <class> class F>
struct S32 {
    MEMLAYOUT_APPLY(S32, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31)
    F<float> x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31;
};

template <template <class> class F>
struct S64 {
    MEMLAYOUT_APPLY(S64, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,
        x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25,
        x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38,
        x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50,
        x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63
    )
    F<float> x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
    F<double> x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25;
    F<int> x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38;
    F<Eigen::Vector3d> x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50;
    F<Eigen::Matrix3d> x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63;
};

template <template <class> class F>
struct Snbody {
    MEMLAYOUT_APPLY(Snbody, x, y, z, vx, vy, vz)
    F<float> x, y, z, vx, vy, vz;
};

template <template <class> class F>
struct Sstencil {
    MEMLAYOUT_APPLY(Sstencil, src, dst, rhs)
    F<double> src, dst, rhs;
};

template <template <class> class F>
struct PxPyPzM {
    MEMLAYOUT_APPLY(PxPyPzM, x, y, z, M)
    F<double> x, y, z, M;
};

constexpr memlayout::Layout L = memlayout::Layout::soa;

/// Register Benchmarks ///
template <typename ArrayType, typename N>
class Fixture1 : public benchmark::Fixture {
 public:
    static constexpr auto n = N::value;
    ArrayType t;
    void SetUp(::benchmark::State &state) override { allocate<ArrayType>(t, n); }
    void TearDown(::benchmark::State &state) override { deallocate<ArrayType>(t); }
};

using S2ArrayType = memlayout::Wrapper<S2, memlayout::pointer, L>;
using S10ArrayType = memlayout::Wrapper<S10, memlayout::pointer, L>;
using S32ArrayType = memlayout::Wrapper<S32, memlayout::pointer, L>;
using S64ArrayType = memlayout::Wrapper<S64, memlayout::pointer, L>;
using SnbodyArrayType = memlayout::Wrapper<Snbody, memlayout::pointer, L>;
using SstencilArrayType = memlayout::Wrapper<Sstencil, memlayout::pointer, L>;
using PxPyPzMArrayType = memlayout::Wrapper<PxPyPzM, memlayout::pointer, L>;

INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyRW, S2ArrayType, N_Large);
INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyCompute, S2ArrayType, N);
INSTANTIATE_BENCHMARKS_F1(BM_CPURealRW, S10ArrayType, N);
INSTANTIATE_BENCHMARKS_F1(BM_CPUStrided, S32ArrayType, N_Large);
INSTANTIATE_BENCHMARKS_F1(BM_CPUHardRW, S64ArrayType, N);
INSTANTIATE_BENCHMARKS_F1(BM_nbody, SnbodyArrayType, N);
INSTANTIATE_BENCHMARKS_F1(BM_stencil, SstencilArrayType, N_Large);

template <typename ArrayType1, typename ArrayType2, typename N>
class Fixture2 : public benchmark::Fixture {
 public:
    static constexpr auto n = N::value;
    ArrayType1 t1;
    ArrayType2 t2;
    void SetUp(::benchmark::State &state) override {
        allocate<ArrayType1>(t1, n);
        allocate<ArrayType2>(t2, n);
    }
    void TearDown(::benchmark::State &state) override {
        deallocate<ArrayType1>(t1);
        deallocate<ArrayType2>(t2);
    }
};

INSTANTIATE_BENCHMARKS_F2(BM_InvariantMass, PxPyPzMArrayType, PxPyPzMArrayType, N_Large);

BENCHMARK_MAIN();
