#pragma once

#include <type_traits>
#include <Eigen/Core>
#include "benchmarks/common.h"
#include "wrapper.h"

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
}

template <template <class> class F>
struct S2 {
    MEMLAYOUT_APPLY_UNARY(x0, x1)
    MEMLAYOUT_APPLY_BINARY(S2, MEMLAYOUT_EXPAND(x0), MEMLAYOUT_EXPAND(x1))
    F<int> x0, x1;
};

template <template <class> class F>
struct S10 {
    MEMLAYOUT_APPLY_UNARY(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)
    MEMLAYOUT_APPLY_BINARY(S10, MEMLAYOUT_EXPAND(x0), MEMLAYOUT_EXPAND(x1), MEMLAYOUT_EXPAND(x2), MEMLAYOUT_EXPAND(x3), MEMLAYOUT_EXPAND(x4), MEMLAYOUT_EXPAND(x5), MEMLAYOUT_EXPAND(x6), MEMLAYOUT_EXPAND(x7), MEMLAYOUT_EXPAND(x8), MEMLAYOUT_EXPAND(x9))
    F<float> x0, x1;
    F<double> x2, x3;
    F<int> x4, x5;
    F<Eigen::Vector3d> x6, x7;
    F<Eigen::Matrix3d> x8, x9;
};

template <template <class> class F>
struct S32 {
    MEMLAYOUT_APPLY_UNARY(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31)
    MEMLAYOUT_APPLY_BINARY(S32, MEMLAYOUT_EXPAND(x0), MEMLAYOUT_EXPAND(x1), MEMLAYOUT_EXPAND(x2), MEMLAYOUT_EXPAND(x3), MEMLAYOUT_EXPAND(x4), MEMLAYOUT_EXPAND(x5), MEMLAYOUT_EXPAND(x6), MEMLAYOUT_EXPAND(x7), MEMLAYOUT_EXPAND(x8), MEMLAYOUT_EXPAND(x9), MEMLAYOUT_EXPAND(x10), MEMLAYOUT_EXPAND(x11), MEMLAYOUT_EXPAND(x12), MEMLAYOUT_EXPAND(x13), MEMLAYOUT_EXPAND(x14), MEMLAYOUT_EXPAND(x15), MEMLAYOUT_EXPAND(x16), MEMLAYOUT_EXPAND(x17), MEMLAYOUT_EXPAND(x18), MEMLAYOUT_EXPAND(x19), MEMLAYOUT_EXPAND(x20), MEMLAYOUT_EXPAND(x21), MEMLAYOUT_EXPAND(x22), MEMLAYOUT_EXPAND(x23), MEMLAYOUT_EXPAND(x24), MEMLAYOUT_EXPAND(x25), MEMLAYOUT_EXPAND(x26), MEMLAYOUT_EXPAND(x27), MEMLAYOUT_EXPAND(x28), MEMLAYOUT_EXPAND(x29), MEMLAYOUT_EXPAND(x30), MEMLAYOUT_EXPAND(x31))
    F<uint32_t> x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31;
};

template <template <class> class F>
struct S64 {
    MEMLAYOUT_APPLY_UNARY(
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,
        x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25,
        x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38,
        x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50,
        x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63
    )
    MEMLAYOUT_APPLY_BINARY(S64,
        MEMLAYOUT_EXPAND(x0), MEMLAYOUT_EXPAND(x1), MEMLAYOUT_EXPAND(x2), MEMLAYOUT_EXPAND(x3), MEMLAYOUT_EXPAND(x4), MEMLAYOUT_EXPAND(x5), MEMLAYOUT_EXPAND(x6), MEMLAYOUT_EXPAND(x7), MEMLAYOUT_EXPAND(x8), MEMLAYOUT_EXPAND(x9), MEMLAYOUT_EXPAND(x10), MEMLAYOUT_EXPAND(x11), MEMLAYOUT_EXPAND(x12),
        MEMLAYOUT_EXPAND(x13), MEMLAYOUT_EXPAND(x14), MEMLAYOUT_EXPAND(x15), MEMLAYOUT_EXPAND(x16), MEMLAYOUT_EXPAND(x17), MEMLAYOUT_EXPAND(x18), MEMLAYOUT_EXPAND(x19), MEMLAYOUT_EXPAND(x20), MEMLAYOUT_EXPAND(x21), MEMLAYOUT_EXPAND(x22), MEMLAYOUT_EXPAND(x23), MEMLAYOUT_EXPAND(x24), MEMLAYOUT_EXPAND(x25),
        MEMLAYOUT_EXPAND(x26), MEMLAYOUT_EXPAND(x27), MEMLAYOUT_EXPAND(x28), MEMLAYOUT_EXPAND(x29), MEMLAYOUT_EXPAND(x30), MEMLAYOUT_EXPAND(x31), MEMLAYOUT_EXPAND(x32), MEMLAYOUT_EXPAND(x33), MEMLAYOUT_EXPAND(x34), MEMLAYOUT_EXPAND(x35), MEMLAYOUT_EXPAND(x36), MEMLAYOUT_EXPAND(x37), MEMLAYOUT_EXPAND(x38),
        MEMLAYOUT_EXPAND(x39), MEMLAYOUT_EXPAND(x40), MEMLAYOUT_EXPAND(x41), MEMLAYOUT_EXPAND(x42), MEMLAYOUT_EXPAND(x43), MEMLAYOUT_EXPAND(x44), MEMLAYOUT_EXPAND(x45), MEMLAYOUT_EXPAND(x46), MEMLAYOUT_EXPAND(x47), MEMLAYOUT_EXPAND(x48), MEMLAYOUT_EXPAND(x49), MEMLAYOUT_EXPAND(x50),
        MEMLAYOUT_EXPAND(x51), MEMLAYOUT_EXPAND(x52), MEMLAYOUT_EXPAND(x53), MEMLAYOUT_EXPAND(x54), MEMLAYOUT_EXPAND(x55), MEMLAYOUT_EXPAND(x56), MEMLAYOUT_EXPAND(x57), MEMLAYOUT_EXPAND(x58), MEMLAYOUT_EXPAND(x59), MEMLAYOUT_EXPAND(x60), MEMLAYOUT_EXPAND(x61), MEMLAYOUT_EXPAND(x62), MEMLAYOUT_EXPAND(x63)
    )
    F<float> x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
    F<double> x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25;
    F<int> x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38;
    F<Eigen::Vector3d> x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50;
    F<Eigen::Matrix3d> x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63;
};

template <template <class> class F>
struct Snbody {
    MEMLAYOUT_APPLY_UNARY(x, y, z, vx, vy, vz)
    MEMLAYOUT_APPLY_BINARY(Snbody, MEMLAYOUT_EXPAND(x), MEMLAYOUT_EXPAND(y), MEMLAYOUT_EXPAND(z), MEMLAYOUT_EXPAND(vx), MEMLAYOUT_EXPAND(vy), MEMLAYOUT_EXPAND(vz))
    F<float> x, y, z, vx, vy, vz;
};

template <template <class> class F>
struct Sstencil {
    MEMLAYOUT_APPLY_UNARY(src, dst, rhs)
    MEMLAYOUT_APPLY_BINARY(Sstencil, MEMLAYOUT_EXPAND(src), MEMLAYOUT_EXPAND(dst), MEMLAYOUT_EXPAND(rhs))
    F<double> src, dst, rhs;
};

template <template <class> class F>
struct PxPyPzM {
    MEMLAYOUT_APPLY_UNARY(x, y, z, M)
    MEMLAYOUT_APPLY_BINARY(PxPyPzM, MEMLAYOUT_EXPAND(x), MEMLAYOUT_EXPAND(y), MEMLAYOUT_EXPAND(z), MEMLAYOUT_EXPAND(M))
    F<double> x, y, z, M;
};

template <typename ArrayType, typename N>
class Fixture1 : public benchmark::Fixture {
public:
    static constexpr auto n = N::value;
    static constexpr Backend backend = Backend::CPU;
    ArrayType t;

    void SetUp(benchmark::State &) override { allocate<ArrayType>(t, n); }
    void TearDown(benchmark::State &) override { deallocate<ArrayType>(t); }
};

template <typename ArrayType1, typename ArrayType2, typename N>
class Fixture2 : public benchmark::Fixture {
public:
    static constexpr auto n = N::value;
    static constexpr Backend backend = Backend::CPU;
    ArrayType1 t1;
    ArrayType2 t2;

    void SetUp(benchmark::State &) override {
        allocate<ArrayType1>(t1, n);
        allocate<ArrayType2>(t2, n);
    }
    void TearDown(benchmark::State &) override {
        deallocate<ArrayType1>(t1);
        deallocate<ArrayType2>(t2);
    }
};
