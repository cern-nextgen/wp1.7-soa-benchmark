#define SOA_MANUAL

#include <span>
#include <vector>
#include <Eigen/Core>
#include "benchmarks/common.h"

constexpr inline size_t align_size(size_t size)
{
    return ((size + Alignment - 1) / Alignment) * Alignment;
}

/// Data structures: SoA with raw pointers into a single aligned buffer ///

struct S2 {
    int *__restrict__ x0, *__restrict__ x1;

    S2(std::byte *buf, size_t n)
    {
        size_t offset = 0;
        x0 = reinterpret_cast<int *__restrict__>(buf);
        offset += align_size(n * sizeof(int));
        x1 = reinterpret_cast<int *__restrict__>(std::launder(buf + offset));
    }

    static size_t size_bytes(size_t n) { return align_size(sizeof(int[n])) * 2; }

    S2(std::nullptr_t, size_t) : x0(nullptr), x1(nullptr) {}
};

struct S10 {
    float *__restrict__ x0, *__restrict__ x1;
    double *__restrict__ x2, *__restrict__ x3;
    int *__restrict__ x4, *__restrict__ x5;
    Eigen::Vector3d *__restrict__ x6, *__restrict__ x7;
    Eigen::Matrix3d *__restrict__ x8, *__restrict__ x9;

    S10(std::byte *buf, size_t n)
    {
        size_t offset = 0;
        x0 = reinterpret_cast<float *__restrict__>(buf);
        offset += align_size(n * sizeof(float));
        x1 = reinterpret_cast<float *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        x2 = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x3 = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x4 = reinterpret_cast<int *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x5 = reinterpret_cast<int *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x6 = reinterpret_cast<Eigen::Vector3d *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x7 = reinterpret_cast<Eigen::Vector3d *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x8 = reinterpret_cast<Eigen::Matrix3d *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Matrix3d));
        x9 = reinterpret_cast<Eigen::Matrix3d *__restrict__>(buf + offset);
    }

    static size_t size_bytes(size_t n)
    {
        return align_size(sizeof(float[n])) * 2 + align_size(sizeof(double[n])) * 2 +
               align_size(sizeof(int[n])) * 2 + align_size(sizeof(Eigen::Vector3d[n])) * 2 +
               align_size(sizeof(Eigen::Matrix3d[n])) * 2;
    }

    S10(std::nullptr_t, size_t) : x0(nullptr), x1(nullptr), x2(nullptr), x3(nullptr),
        x4(nullptr), x5(nullptr), x6(nullptr), x7(nullptr), x8(nullptr), x9(nullptr) {}
};

struct S32 {
    float *__restrict__ x0,  *__restrict__ x1,  *__restrict__ x2,  *__restrict__ x3,
          *__restrict__ x4,  *__restrict__ x5,  *__restrict__ x6,  *__restrict__ x7,
          *__restrict__ x8,  *__restrict__ x9,  *__restrict__ x10, *__restrict__ x11,
          *__restrict__ x12, *__restrict__ x13, *__restrict__ x14, *__restrict__ x15,
          *__restrict__ x16, *__restrict__ x17, *__restrict__ x18, *__restrict__ x19,
          *__restrict__ x20, *__restrict__ x21, *__restrict__ x22, *__restrict__ x23,
          *__restrict__ x24, *__restrict__ x25, *__restrict__ x26, *__restrict__ x27,
          *__restrict__ x28, *__restrict__ x29, *__restrict__ x30, *__restrict__ x31;

    S32(std::byte *buf, size_t n)
    {
        auto *p = reinterpret_cast<float *__restrict__>(buf);
        auto next = [&]() -> float *__restrict__ {
            float *__restrict__ ptr = p;
            p = reinterpret_cast<float *__restrict__>(reinterpret_cast<std::byte *>(p) + align_size(n * sizeof(float)));
            return ptr;
        };
        x0  = next(); x1  = next(); x2  = next(); x3  = next();
        x4  = next(); x5  = next(); x6  = next(); x7  = next();
        x8  = next(); x9  = next(); x10 = next(); x11 = next();
        x12 = next(); x13 = next(); x14 = next(); x15 = next();
        x16 = next(); x17 = next(); x18 = next(); x19 = next();
        x20 = next(); x21 = next(); x22 = next(); x23 = next();
        x24 = next(); x25 = next(); x26 = next(); x27 = next();
        x28 = next(); x29 = next(); x30 = next(); x31 = next();
    }

    static size_t size_bytes(size_t n) { return align_size(sizeof(float[n])) * 32; }

    S32(std::nullptr_t, size_t) : x0(nullptr), x1(nullptr), x2(nullptr), x3(nullptr),
        x4(nullptr), x5(nullptr), x6(nullptr), x7(nullptr), x8(nullptr), x9(nullptr),
        x10(nullptr), x11(nullptr), x12(nullptr), x13(nullptr), x14(nullptr), x15(nullptr),
        x16(nullptr), x17(nullptr), x18(nullptr), x19(nullptr), x20(nullptr), x21(nullptr),
        x22(nullptr), x23(nullptr), x24(nullptr), x25(nullptr), x26(nullptr), x27(nullptr),
        x28(nullptr), x29(nullptr), x30(nullptr), x31(nullptr) {}
};

struct S64 {
    float   *__restrict__ x0,  *__restrict__ x1,  *__restrict__ x2,  *__restrict__ x3,
            *__restrict__ x4,  *__restrict__ x5,  *__restrict__ x6,  *__restrict__ x7,
            *__restrict__ x8,  *__restrict__ x9,  *__restrict__ x10, *__restrict__ x11,
            *__restrict__ x12;
    double  *__restrict__ x13, *__restrict__ x14, *__restrict__ x15, *__restrict__ x16,
            *__restrict__ x17, *__restrict__ x18, *__restrict__ x19, *__restrict__ x20,
            *__restrict__ x21, *__restrict__ x22, *__restrict__ x23, *__restrict__ x24,
            *__restrict__ x25;
    int     *__restrict__ x26, *__restrict__ x27, *__restrict__ x28, *__restrict__ x29,
            *__restrict__ x30, *__restrict__ x31, *__restrict__ x32, *__restrict__ x33,
            *__restrict__ x34, *__restrict__ x35, *__restrict__ x36, *__restrict__ x37,
            *__restrict__ x38;
    Eigen::Vector3d *__restrict__ x39, *__restrict__ x40, *__restrict__ x41, *__restrict__ x42,
                    *__restrict__ x43, *__restrict__ x44, *__restrict__ x45, *__restrict__ x46,
                    *__restrict__ x47, *__restrict__ x48, *__restrict__ x49, *__restrict__ x50;
    Eigen::Matrix3d *__restrict__ x51, *__restrict__ x52, *__restrict__ x53, *__restrict__ x54,
                    *__restrict__ x55, *__restrict__ x56, *__restrict__ x57, *__restrict__ x58,
                    *__restrict__ x59, *__restrict__ x60, *__restrict__ x61, *__restrict__ x62,
                    *__restrict__ x63;

    S64(std::byte *buf, size_t n)
    {
        size_t offset = 0;
        auto assign_float = [&](float *__restrict__ &ptr) {
            ptr = reinterpret_cast<float *__restrict__>(buf + offset);
            offset += align_size(n * sizeof(float));
        };
        auto assign_double = [&](double *__restrict__ &ptr) {
            ptr = reinterpret_cast<double *__restrict__>(buf + offset);
            offset += align_size(n * sizeof(double));
        };
        auto assign_int = [&](int *__restrict__ &ptr) {
            ptr = reinterpret_cast<int *__restrict__>(buf + offset);
            offset += align_size(n * sizeof(int));
        };
        auto assign_vec3 = [&](Eigen::Vector3d *__restrict__ &ptr) {
            ptr = reinterpret_cast<Eigen::Vector3d *__restrict__>(buf + offset);
            offset += align_size(n * sizeof(Eigen::Vector3d));
        };
        auto assign_mat3 = [&](Eigen::Matrix3d *__restrict__ &ptr) {
            ptr = reinterpret_cast<Eigen::Matrix3d *__restrict__>(buf + offset);
            offset += align_size(n * sizeof(Eigen::Matrix3d));
        };
        assign_float(x0);  assign_float(x1);  assign_float(x2);  assign_float(x3);
        assign_float(x4);  assign_float(x5);  assign_float(x6);  assign_float(x7);
        assign_float(x8);  assign_float(x9);  assign_float(x10); assign_float(x11);
        assign_float(x12);
        assign_double(x13); assign_double(x14); assign_double(x15); assign_double(x16);
        assign_double(x17); assign_double(x18); assign_double(x19); assign_double(x20);
        assign_double(x21); assign_double(x22); assign_double(x23); assign_double(x24);
        assign_double(x25);
        assign_int(x26); assign_int(x27); assign_int(x28); assign_int(x29); assign_int(x30);
        assign_int(x31); assign_int(x32); assign_int(x33); assign_int(x34); assign_int(x35);
        assign_int(x36); assign_int(x37); assign_int(x38);
        assign_vec3(x39); assign_vec3(x40); assign_vec3(x41); assign_vec3(x42);
        assign_vec3(x43); assign_vec3(x44); assign_vec3(x45); assign_vec3(x46);
        assign_vec3(x47); assign_vec3(x48); assign_vec3(x49); assign_vec3(x50);
        assign_mat3(x51); assign_mat3(x52); assign_mat3(x53); assign_mat3(x54);
        assign_mat3(x55); assign_mat3(x56); assign_mat3(x57); assign_mat3(x58);
        assign_mat3(x59); assign_mat3(x60); assign_mat3(x61); assign_mat3(x62);
        assign_mat3(x63);
    }

    static size_t size_bytes(size_t n)
    {
        return align_size(sizeof(float[n])) * 13 + align_size(sizeof(double[n])) * 13 +
               align_size(sizeof(int[n])) * 13 + align_size(sizeof(Eigen::Vector3d[n])) * 12 +
               align_size(sizeof(Eigen::Matrix3d[n])) * 13;
    }

    S64(std::nullptr_t, size_t) : x0(nullptr), x1(nullptr), x2(nullptr), x3(nullptr),
        x4(nullptr), x5(nullptr), x6(nullptr), x7(nullptr), x8(nullptr), x9(nullptr),
        x10(nullptr), x11(nullptr), x12(nullptr), x13(nullptr), x14(nullptr), x15(nullptr),
        x16(nullptr), x17(nullptr), x18(nullptr), x19(nullptr), x20(nullptr), x21(nullptr),
        x22(nullptr), x23(nullptr), x24(nullptr), x25(nullptr), x26(nullptr), x27(nullptr),
        x28(nullptr), x29(nullptr), x30(nullptr), x31(nullptr), x32(nullptr), x33(nullptr),
        x34(nullptr), x35(nullptr), x36(nullptr), x37(nullptr), x38(nullptr), x39(nullptr),
        x40(nullptr), x41(nullptr), x42(nullptr), x43(nullptr), x44(nullptr), x45(nullptr),
        x46(nullptr), x47(nullptr), x48(nullptr), x49(nullptr), x50(nullptr), x51(nullptr),
        x52(nullptr), x53(nullptr), x54(nullptr), x55(nullptr), x56(nullptr), x57(nullptr),
        x58(nullptr), x59(nullptr), x60(nullptr), x61(nullptr), x62(nullptr), x63(nullptr) {}
};

struct Snbody {
    float *__restrict__ x, *__restrict__ y, *__restrict__ z;
    float *__restrict__ vx, *__restrict__ vy, *__restrict__ vz;

    Snbody(std::byte *buf, size_t n)
    {
        size_t offset = 0;
        x  = reinterpret_cast<float *__restrict__>(buf);
        offset += align_size(n * sizeof(float));
        y  = reinterpret_cast<float *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        z  = reinterpret_cast<float *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        vx = reinterpret_cast<float *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        vy = reinterpret_cast<float *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        vz = reinterpret_cast<float *__restrict__>(buf + offset);
    }

    static size_t size_bytes(size_t n) { return align_size(sizeof(float[n])) * 6; }

    Snbody(std::nullptr_t, size_t) : x(nullptr), y(nullptr), z(nullptr),
        vx(nullptr), vy(nullptr), vz(nullptr) {}
};

struct Sstencil {
    double *__restrict__ src, *__restrict__ dst, *__restrict__ rhs;

    Sstencil(std::byte *buf, size_t n)
    {
        size_t offset = 0;
        src = reinterpret_cast<double *__restrict__>(buf);
        offset += align_size(n * sizeof(double));
        dst = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        rhs = reinterpret_cast<double *__restrict__>(buf + offset);
    }

    static size_t size_bytes(size_t n) { return align_size(sizeof(double[n])) * 3; }

    Sstencil(std::nullptr_t, size_t) : src(nullptr), dst(nullptr), rhs(nullptr) {}
};

struct PxPyPzM {
    double *__restrict__ x, *__restrict__ y, *__restrict__ z, *__restrict__ M;

    PxPyPzM(std::byte *buf, size_t n)
    {
        size_t offset = 0;
        x = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        y = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        z = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        M = reinterpret_cast<double *__restrict__>(buf + offset);
    }

    static size_t size_bytes(size_t n) { return align_size(sizeof(double[n])) * 4; }

    PxPyPzM(std::nullptr_t, size_t) : x(nullptr), y(nullptr), z(nullptr), M(nullptr) {}
};

/// Fixtures ///

template <typename S, typename N>
class Fixture1 : public benchmark::Fixture {
public:
    static constexpr auto n = N::value;
    std::byte *buffer = nullptr;
    S t;

    void SetUp(benchmark::State &) override
    {
        buffer = reinterpret_cast<std::byte *>(std::aligned_alloc(Alignment, S::size_bytes(n)));
        t = S(buffer, n);
    }

    void TearDown(benchmark::State &) override
    {
        std::free(buffer);
        buffer = nullptr;
    }

    Fixture1() : t(nullptr, 0) {}
};

template <typename S1, typename S2, typename N>
class Fixture2 : public benchmark::Fixture {
public:
    static constexpr auto n = N::value;
    std::byte *buffer1 = nullptr, *buffer2 = nullptr;
    S1 t1;
    S2 t2;

    void SetUp(benchmark::State &) override
    {
        buffer1 = reinterpret_cast<std::byte *>(std::aligned_alloc(Alignment, S1::size_bytes(n)));
        buffer2 = reinterpret_cast<std::byte *>(std::aligned_alloc(Alignment, S2::size_bytes(n)));
        t1 = S1(buffer1, n);
        t2 = S2(buffer2, n);
    }

    void TearDown(benchmark::State &) override
    {
        std::free(buffer1); buffer1 = nullptr;
        std::free(buffer2); buffer2 = nullptr;
    }

    Fixture2() : t1(nullptr, 0), t2(nullptr, 0) {}
};

/// Benchmarks ///

#include "benchmarks/bm_easy.h"
#include "benchmarks/bm_real.h"
#include "benchmarks/bm_strided.h"
#include "benchmarks/bm_hard.h"
#include "benchmarks/bm_nbody.h"
#include "benchmarks/bm_stencil.h"
#include "benchmarks/bm_invmass.h"

INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyRW,      S2,       N_Large);
INSTANTIATE_BENCHMARKS_F1(BM_CPUEasyCompute, S2,       N);
INSTANTIATE_BENCHMARKS_F1(BM_CPURealRW,      S10,      N);
INSTANTIATE_BENCHMARKS_F1(BM_CPUStrided,     S32,      N_Large);
INSTANTIATE_BENCHMARKS_F1(BM_CPUHardRW,      S64,      N);
INSTANTIATE_BENCHMARKS_F1(BM_nbody,          Snbody,   N);
INSTANTIATE_BENCHMARKS_F1(BM_stencil,        Sstencil, N_Large);

INSTANTIATE_BENCHMARKS_F2(BM_InvariantMass, PxPyPzM, PxPyPzM, N_Large);

BENCHMARK_MAIN();
