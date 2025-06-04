#define SOA_MANUAL

#include <vector>
#include <span>

#include "benchmark.h"
#include <Eigen/Core>

#include <iostream>

constexpr size_t Alignment = 64;
constexpr inline size_t align_size(size_t size) { return ((size + Alignment - 1) / Alignment) * Alignment; }

#define ADDR_FMT
struct S2 {
    std::span<int> x0, x1;

    S2(std::byte* buf, size_t n) {
        size_t offset = 0;

        x0 = std::span(reinterpret_cast<int*>(std::launder(new (buf) int[n])), n);
        offset += align_size(x0.size_bytes());
        x1 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
    }

    static size_t size_bytes(size_t n) {
        return align_size(sizeof(int[n])) * 2;
    }
};

struct S10 {
    std::span<float> x0, x1;
    std::span<double> x2, x3;
    std::span<int> x4, x5;
    std::span<Eigen::Vector3d> x6, x7;
    std::span<Eigen::Matrix3d> x8, x9;

    S10(std::byte* buf, size_t n)  {
        size_t offset = 0;
        x0 = std::span(reinterpret_cast<float*>(std::launder(new (buf) float[n])), n);
        offset += align_size(x0.size_bytes());
        x1 = std::span(reinterpret_cast<float*>(std::launder(new (buf + offset) float[n])), n);
        offset += align_size(x1.size_bytes());
        x2 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x2.size_bytes());
        x3 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x3.size_bytes());
        x4 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x4.size_bytes());
        x5 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x5.size_bytes());
        x6 = std::span(reinterpret_cast<Eigen::Vector3d*>(std::launder(new (buf + offset) Eigen::Vector3d[n])), n);
        offset += align_size(x6.size_bytes());
        x7 = std::span(reinterpret_cast<Eigen::Vector3d*>(std::launder(new (buf + offset) Eigen::Vector3d[n])), n);
        offset += align_size(x7.size_bytes());
        x8 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
        offset += align_size(x8.size_bytes());
        x9 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
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
    std::span<float> x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
    std::span<double> x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25;
    std::span<int> x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38;
    std::span<Eigen::Vector3d> x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50;
    std::span<Eigen::Matrix3d> x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63;

    S64(std::byte* buf, size_t n) {
        size_t offset = 0;
        storage.resize(align_size(sizeof(float[n]) * 13 + sizeof(double[n]) * 13 + sizeof(int[n]) * 13 +
                                  sizeof(Eigen::Vector3d[n]) * 13 + sizeof(Eigen::Matrix3d[n]) * 13));
        x0 = std::span(reinterpret_cast<float*>(std::launder(new (buf) float[n])), n);
        offset += align_size(x0.size_bytes());
        x1 = std::span(reinterpret_cast<float*>(std::launder(new (buf + offset) float[n])), n);
        offset += align_size(x1.size_bytes());
        x2 = std::span(reinterpret_cast<float*>(std::launder(new (buf + offset) float[n])), n);
        offset += align_size(x2.size_bytes());
        x3 = std::span(reinterpret_cast<float*>(std::launder(new (buf + offset) float[n])), n);
        offset += align_size(x3.size_bytes());
        x4 = std::span(reinterpret_cast<float*>(std::launder(new (buf + offset) float[n])), n);
        offset += align_size(x4.size_bytes());
        x5 = std::span(reinterpret_cast<float*>(std::launder(new (buf + offset) float[n])), n);
        offset += align_size(x5.size_bytes());
        x6 = std::span(reinterpret_cast<float*>(std::launder(new (buf + offset) float[n])), n);
        offset += align_size(x6.size_bytes());
        x7 = std::span(reinterpret_cast<float*>(std::launder(new (buf + offset) float[n])), n);
        offset += align_size(x7.size_bytes());
        x8 = std::span(reinterpret_cast<float*>(std::launder(new (buf + offset) float[n])), n);
        offset += align_size(x8.size_bytes());
        x9 = std::span(reinterpret_cast<float*>(std::launder(new (buf + offset) float[n])), n);
        offset += align_size(x9.size_bytes());
        x10 = std::span(reinterpret_cast<float*>(std::launder(new (buf + offset) float[n])), n);
        offset += align_size(x10.size_bytes());
        x11 = std::span(reinterpret_cast<float*>(std::launder(new (buf + offset) float[n])), n);
        offset += align_size(x11.size_bytes());
        x12 = std::span(reinterpret_cast<float*>(std::launder(new (buf + offset) float[n])), n);
        offset += align_size(x12.size_bytes());
        x13 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x13.size_bytes());
        x14 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x14.size_bytes());
        x15 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x15.size_bytes());
        x16 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x16.size_bytes());
        x17 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x17.size_bytes());
        x18 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x18.size_bytes());
        x19 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x19.size_bytes());
        x20 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x20.size_bytes());
        x21 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x21.size_bytes());
        x22 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x22.size_bytes());
        x23 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x23.size_bytes());
        x24 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x24.size_bytes());
        x25 = std::span(reinterpret_cast<double*>(std::launder(new (buf + offset) double[n])), n);
        offset += align_size(x25.size_bytes());
        x26 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x26.size_bytes());
        x27 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x27.size_bytes());
        x28 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x28.size_bytes());
        x29 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x29.size_bytes());
        x30 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x30.size_bytes());
        x31 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x31.size_bytes());
        x32 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x32.size_bytes());
        x33 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x33.size_bytes());
        x34 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x34.size_bytes());
        x35 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x35.size_bytes());
        x36 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x36.size_bytes());
        x37 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x37.size_bytes());
        x38 = std::span(reinterpret_cast<int*>(std::launder(new (buf + offset) int[n])), n);
        offset += align_size(x38.size_bytes());
        x39 = std::span(reinterpret_cast<Eigen::Vector3d*>(std::launder(new (buf + offset) Eigen::Vector3d[n])), n);
        offset += align_size(x39.size_bytes());
        x40 = std::span(reinterpret_cast<Eigen::Vector3d*>(std::launder(new (buf + offset) Eigen::Vector3d[n])), n);
        offset += align_size(x40.size_bytes());
        x41 = std::span(reinterpret_cast<Eigen::Vector3d*>(std::launder(new (buf + offset) Eigen::Vector3d[n])), n);
        offset += align_size(x41.size_bytes());
        x42 = std::span(reinterpret_cast<Eigen::Vector3d*>(std::launder(new (buf + offset) Eigen::Vector3d[n])), n);
        offset += align_size(x42.size_bytes());
        x43 = std::span(reinterpret_cast<Eigen::Vector3d*>(std::launder(new (buf + offset) Eigen::Vector3d[n])), n);
        offset += align_size(x43.size_bytes());
        x44 = std::span(reinterpret_cast<Eigen::Vector3d*>(std::launder(new (buf + offset) Eigen::Vector3d[n])), n);
        offset += align_size(x44.size_bytes());
        x45 = std::span(reinterpret_cast<Eigen::Vector3d*>(std::launder(new (buf + offset) Eigen::Vector3d[n])), n);
        offset += align_size(x45.size_bytes());
        x46 = std::span(reinterpret_cast<Eigen::Vector3d*>(std::launder(new (buf + offset) Eigen::Vector3d[n])), n);
        offset += align_size(x46.size_bytes());
        x47 = std::span(reinterpret_cast<Eigen::Vector3d*>(std::launder(new (buf + offset) Eigen::Vector3d[n])), n);
        offset += align_size(x47.size_bytes());
        x48 = std::span(reinterpret_cast<Eigen::Vector3d*>(std::launder(new (buf + offset) Eigen::Vector3d[n])), n);
        offset += align_size(x48.size_bytes());
        x49 = std::span(reinterpret_cast<Eigen::Vector3d*>(std::launder(new (buf + offset) Eigen::Vector3d[n])), n);
        offset += align_size(x49.size_bytes());
        x50 = std::span(reinterpret_cast<Eigen::Vector3d*>(std::launder(new (buf + offset) Eigen::Vector3d[n])), n);
        offset += align_size(x50.size_bytes());
        x51 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
        offset += align_size(x51.size_bytes());
        x52 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
        offset += align_size(x52.size_bytes());
        x53 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
        offset += align_size(x53.size_bytes());
        x54 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
        offset += align_size(x54.size_bytes());
        x55 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
        offset += align_size(x55.size_bytes());
        x56 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
        offset += align_size(x56.size_bytes());
        x57 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
        offset += align_size(x57.size_bytes());
        x58 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
        offset += align_size(x58.size_bytes());
        x59 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
        offset += align_size(x59.size_bytes());
        x60 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
        offset += align_size(x60.size_bytes());
        x61 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
        offset += align_size(x61.size_bytes());
        x62 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
        offset += align_size(x62.size_bytes());
        x63 = std::span(reinterpret_cast<Eigen::Matrix3d*>(std::launder(new (buf + offset) Eigen::Matrix3d[n])), n);
    }

    static size_t size_bytes(size_t n) {
        return align_size(sizeof(float[n])) * 13 + align_size(sizeof(double[n])) * 13 +
               align_size(sizeof(int[n])) * 13 + align_size(sizeof(Eigen::Vector3d[n])) * 12 +
               align_size(sizeof(Eigen::Matrix3d[n])) * 13;
    }
};

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    std::vector<void *> free_list;

    // Seperate loops to sort the output by benchmark.
    for (auto n : N) {
        auto buffer = reinterpret_cast<std::byte *>(std::aligned_alloc(Alignment, S2::size_bytes(n)));
        benchmark::RegisterBenchmark("BM_CPUEasyRW", BM_CPUEasyRW<S2>, S2(buffer,n))->Arg(n)->Unit(benchmark::kMillisecond);
        free_list.push_back(buffer);
    }

    for (auto n : N) {
        auto buffer = reinterpret_cast<std::byte *>(std::aligned_alloc(Alignment, S2::size_bytes(n)));
        S2 t2b(buffer, n);
        benchmark::RegisterBenchmark("BM_CPUEasyCompute", BM_CPUEasyCompute<S2>, t2b)->Arg(n)->Unit(benchmark::kMillisecond);
        free_list.push_back(buffer);
    }

    for (auto n : N) {
        auto buffer = reinterpret_cast<std::byte *>(std::aligned_alloc(Alignment, S10::size_bytes(n)));
        S10 t10(buffer, n);
        benchmark::RegisterBenchmark("BM_CPURealRW", BM_CPURealRW<S10>, t10)->Arg(n)->Unit(benchmark::kMillisecond);
        free_list.push_back(buffer);
    }

    for (auto n : N) {
        auto buffer = reinterpret_cast<std::byte *>(std::aligned_alloc(Alignment, S64::size_bytes(n)));
        S64 t64(buffer, n);
        benchmark::RegisterBenchmark("BM_CPUHardRW", BM_CPUHardRW<S64>, t64)->Arg(n)->Unit(benchmark::kMillisecond);
        free_list.push_back(buffer);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    for (auto buffer : free_list) {
        std::free(buffer);
    }
}