#define AOS_MANUAL

#include <Eigen/Core>

#include "benchmark.h"

struct S2 { int x0, x1; };

struct S10 {
    float x0, x1;
    double x2, x3;
    int x4, x5;
    Eigen::Vector3d x6, x7;
    Eigen::Matrix3d x8, x9;
};

struct S64 {
    float x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
    double x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25;
    int x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38;
    Eigen::Vector3d x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50;
    Eigen::Matrix3d x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63;
};

struct Snbody { float x, y, z, vx, vy, vz; };

struct Sstencil { double src, dst, rhs; };

struct PxPyPzM { double x, y, z, M; };

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    std::vector<S2*> BM_CPUEasyRW_ptrs;
    for (auto n : N) {
        S2* t = new S2[n];
        BM_CPUEasyRW_ptrs.push_back(t);
        benchmark::RegisterBenchmark("BM_CPUEasyRW", BM_CPUEasyRW<S2*>, t)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    std::vector<S2*> BM_CPUEasyCompute_ptrs;
    for (auto n : N) {
        S2* t = new S2[n];
        BM_CPUEasyCompute_ptrs.push_back(t);
        benchmark::RegisterBenchmark("BM_CPUEasyCompute", BM_CPUEasyCompute<S2*>, t)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    std::vector<S10*> BM_CPURealRW_ptrs;
    for (auto n : N) {
        S10* t = new S10[n];
        BM_CPURealRW_ptrs.push_back(t);
        benchmark::RegisterBenchmark("BM_CPURealRW", BM_CPURealRW<S10*>, t)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    std::vector<S64*> BM_CPUHardRW_ptrs;
    for (auto n : N) {
        S64* t = new S64[n];
        BM_CPUHardRW_ptrs.push_back(t);
        benchmark::RegisterBenchmark("BM_CPUHardRW", BM_CPUHardRW<S64*>, t)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    std::vector<Snbody*> BM_nbody_ptrs;
    for (auto n : N) {
        Snbody* t = new Snbody[n];
        BM_nbody_ptrs.push_back(t);
        benchmark::RegisterBenchmark("BM_nbody", BM_nbody<Snbody*>, t)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    std::vector<Sstencil*> BM_stencil_ptrs;
    for (auto n : N) {
        Sstencil* t = new Sstencil[n];
        BM_stencil_ptrs.push_back(t);
        benchmark::RegisterBenchmark("BM_stencil", BM_stencil<Sstencil*>, t)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    std::vector<PxPyPzM*> BM_InvariantMass_ptrs;
    for (std::size_t n : N) {
        PxPyPzM* t1 = new PxPyPzM[n];
        PxPyPzM* t2 = new PxPyPzM[n];
        BM_InvariantMass_ptrs.push_back(t1);
        BM_InvariantMass_ptrs.push_back(t2);
        benchmark::RegisterBenchmark("BM_InvariantMass", BM_InvariantMass<PxPyPzM*, PxPyPzM*>, t1, t2)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    for (S2* ptr : BM_CPUEasyRW_ptrs) delete[] ptr;
    for (S2* ptr : BM_CPUEasyCompute_ptrs) delete[] ptr;
    for (S10* ptr : BM_CPURealRW_ptrs) delete[] ptr;
    for (S64* ptr : BM_CPUHardRW_ptrs) delete[] ptr;
    for (Snbody* ptr : BM_nbody_ptrs) delete[] ptr;
    for (Sstencil* ptr : BM_stencil_ptrs) delete[] ptr;
    for (PxPyPzM* ptr : BM_InvariantMass_ptrs) delete[] ptr;

    return 0;
}