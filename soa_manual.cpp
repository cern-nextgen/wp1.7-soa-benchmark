#define SOA_MANUAL

#include <vector>
#include <span>

#include "benchmark.h"
#include <Eigen/Core>

#include <iostream>

constexpr size_t Alignment = 128;
constexpr inline size_t align_size(size_t size) { return ((size + Alignment - 1) / Alignment) * Alignment; }

#define ADDR_FMT
struct S2 {
    int* __restrict__ x0, *__restrict__ x1;

    S2(std::byte* buf, size_t n) {
        size_t offset = 0;

        x0 = reinterpret_cast<int* __restrict__>(buf);
        offset += align_size(n * sizeof(int));
        x1 = reinterpret_cast<int* __restrict__>(std::launder(buf + offset));
    }

    static size_t size_bytes(size_t n) {
        return align_size(sizeof(int[n])) * 2;
    }
};

struct S10 {
    float* __restrict__ x0, *__restrict__ x1;
    double* __restrict__ x2, *__restrict__ x3;
    int* __restrict__ x4, *__restrict__ x5;
    Eigen::Vector3d* __restrict__ x6, *__restrict__ x7;
    Eigen::Matrix3d* __restrict__ x8, *__restrict__ x9;

    S10(std::byte* buf, size_t n)  {
        size_t offset = 0;
        x0 = reinterpret_cast<float* __restrict__>(buf);
        offset += align_size(n * sizeof(float));
        x1 = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        x2 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x3 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x4 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x5 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x6 = reinterpret_cast<Eigen::Vector3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x7 = reinterpret_cast<Eigen::Vector3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x8 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Matrix3d));
        x9 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
    }

    static size_t size_bytes(size_t n) {
        return align_size(sizeof(float[n])) * 2 +
               align_size(sizeof(double[n])) * 2 +
               align_size(sizeof(int[n])) * 2 +
               align_size(sizeof(Eigen::Vector3d[n])) * 2 +
               align_size(sizeof(Eigen::Matrix3d[n])) * 2;
    }
};

struct S64 {
    std::vector<std::byte> storage;
    float *__restrict__ x0, *__restrict__ x1, *__restrict__ x2, *__restrict__ x3,
       *__restrict__ x4, *__restrict__ x5, *__restrict__ x6, *__restrict__ x7,
       *__restrict__ x8, *__restrict__ x9, *__restrict__ x10, *__restrict__ x11, *__restrict__ x12;
    double* __restrict__ x13, *__restrict__ x14, *__restrict__ x15, *__restrict__ x16,
           *__restrict__ x17, *__restrict__ x18, *__restrict__ x19, *__restrict__ x20,
           *__restrict__ x21, *__restrict__ x22, *__restrict__ x23, *__restrict__ x24, *__restrict__ x25;
    int* __restrict__ x26, *__restrict__ x27, *__restrict__ x28, *__restrict__ x29,
        *__restrict__ x30, *__restrict__ x31, *__restrict__ x32, *__restrict__ x33,
        *__restrict__ x34, *__restrict__ x35, *__restrict__ x36, *__restrict__ x37,
        *__restrict__ x38;
    Eigen::Vector3d* __restrict__ x39, *__restrict__ x40, *__restrict__ x41, *__restrict__ x42,
        *__restrict__ x43, *__restrict__ x44, *__restrict__ x45, *__restrict__ x46,
        *__restrict__ x47, *__restrict__ x48, *__restrict__ x49, *__restrict__ x50;
    Eigen::Matrix3d* __restrict__ x51, *__restrict__ x52, *__restrict__ x53, *__restrict__ x54,
        *__restrict__ x55, *__restrict__ x56, *__restrict__ x57, *__restrict__ x58,
        *__restrict__ x59, *__restrict__ x60, *__restrict__ x61, *__restrict__ x62, *__restrict__ x63;

    S64(std::byte* buf, size_t n) {
        size_t offset = 0;
        storage.resize(align_size(sizeof(float[n]) * 13 + sizeof(double[n]) * 13 + sizeof(int[n]) * 13 +
                                  sizeof(Eigen::Vector3d[n]) * 13 + sizeof(Eigen::Matrix3d[n]) * 13));
        x0 = reinterpret_cast<float* __restrict__>(buf);
        offset += align_size(n * sizeof(float));
        x1 = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        x2 = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        x3 = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        x4 = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        x5 = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        x6 = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        x7 = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        x8 = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        x9 = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        x10 = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        x11 = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        x12 = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        x13 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x14 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x15 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x16 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x17 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x18 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x19 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x20 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x21 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x22 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x23 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x24 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x25 = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        x26 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x27 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x28 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x29 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x30 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x31 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x32 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x33 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x34 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x35 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x36 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x37 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x38 = reinterpret_cast<int* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(int));
        x39 = reinterpret_cast<Eigen::Vector3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x40 = reinterpret_cast<Eigen::Vector3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x41 = reinterpret_cast<Eigen::Vector3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x42 = reinterpret_cast<Eigen::Vector3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x43 = reinterpret_cast<Eigen::Vector3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x44 = reinterpret_cast<Eigen::Vector3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x45 = reinterpret_cast<Eigen::Vector3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x46 = reinterpret_cast<Eigen::Vector3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x47 = reinterpret_cast<Eigen::Vector3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x48 = reinterpret_cast<Eigen::Vector3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x49 = reinterpret_cast<Eigen::Vector3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x50 = reinterpret_cast<Eigen::Vector3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Vector3d));
        x51 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Matrix3d));
        x52 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Matrix3d));
        x53 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Matrix3d));
        x54 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Matrix3d));
        x55 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Matrix3d));
        x56 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Matrix3d));
        x57 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Matrix3d));
        x58 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Matrix3d));
        x59 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Matrix3d));
        x60 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Matrix3d));
        x61 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Matrix3d));
        x62 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(Eigen::Matrix3d));
        x63 = reinterpret_cast<Eigen::Matrix3d* __restrict__>(buf + offset);
    }

    static size_t size_bytes(size_t n) {
        return align_size(sizeof(float[n])) * 13 + align_size(sizeof(double[n])) * 13 +
               align_size(sizeof(int[n])) * 13 + align_size(sizeof(Eigen::Vector3d[n])) * 12 +
               align_size(sizeof(Eigen::Matrix3d[n])) * 13;
    }
};

struct Snbody {
    float* __restrict__ x, *__restrict__ y, *__restrict__ z;
    float* __restrict__ vx, *__restrict__ vy, *__restrict__ vz;

    Snbody(std::byte* buf, size_t n) {
        size_t offset = 0;

        x = reinterpret_cast<float* __restrict__>(buf);
        offset += align_size(n * sizeof(float));
        y = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        z = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        vx = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        vy = reinterpret_cast<float* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        vz = reinterpret_cast<float* __restrict__>(buf + offset);
    }

    static size_t size_bytes(size_t n) {
        return align_size(sizeof(float[n])) * 6;
    }
};

struct Sstencil {
    double* __restrict__ src, *__restrict__ dst, *__restrict__ rhs;

    Sstencil(std::byte* buf, size_t n) {
        size_t offset = 0;

        src = reinterpret_cast<double* __restrict__>(buf);
        offset += align_size(n * sizeof(double));
        dst = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        rhs = reinterpret_cast<double* __restrict__>(buf + offset);
    }

    static size_t size_bytes(size_t n) {
        return align_size(sizeof(double[n])) * 3;
    }
};

struct PxPyPzM {
    double *x, *y, *z, *M;

    PxPyPzM(std::byte* buf, size_t n) {
        size_t offset = 0;
        x = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        y = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        z = reinterpret_cast<double* __restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        M = reinterpret_cast<double* __restrict__>(buf + offset);
    }

    static size_t size_bytes(size_t n) {
        return align_size(sizeof(double[n])) * 4;
    }
};

// Helper function to register benchmarks with one argument
template <typename S>
void RegisterBenchmarkHelper(const char* name, auto bm_func, auto &free_list) {
    for (auto n : N) {
        auto buffer = reinterpret_cast<std::byte * __restrict__>(std::aligned_alloc(Alignment, S::size_bytes(n)));
        S t(buffer, n);
        benchmark::RegisterBenchmark(name, bm_func, t)->Arg(n)->Unit(benchmark::kMillisecond);
        free_list.push_back(buffer);
    }
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    std::vector<void *> free_list;

    // Seperate loops to sort the output by benchmark.
    RegisterBenchmarkHelper<S2>("BM_CPUEasyRW", BM_CPUEasyRW<S2>, free_list);
    RegisterBenchmarkHelper<S2>("BM_CPUEasyCompute", BM_CPUEasyCompute<S2>, free_list);
    RegisterBenchmarkHelper<S10>("BM_CPURealRW", BM_CPURealRW<S10>, free_list);
    RegisterBenchmarkHelper<S64>("BM_CPUHardRW", BM_CPUHardRW<S64>, free_list);
    RegisterBenchmarkHelper<Snbody>("BM_nbody", BM_nbody<Snbody>, free_list);
    RegisterBenchmarkHelper<Sstencil>("BM_stencil", BM_stencil<Sstencil>, free_list);

    for (auto n : N) {
        auto buffer1 = reinterpret_cast<std::byte * __restrict__>(std::aligned_alloc(Alignment, PxPyPzM::size_bytes(n)));
        auto buffer2 = reinterpret_cast<std::byte * __restrict__>(std::aligned_alloc(Alignment, PxPyPzM::size_bytes(n)));
        PxPyPzM t1(buffer1, n);
        PxPyPzM t2(buffer2, n);
        benchmark::RegisterBenchmark("BM_InvariantMass", BM_InvariantMass<PxPyPzM, PxPyPzM>, t1, t2)->Arg(n)->Unit(benchmark::kMillisecond);
        free_list.push_back(buffer1);
        free_list.push_back(buffer2);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    for (auto buffer : free_list) {
        std::free(buffer);
    }
}