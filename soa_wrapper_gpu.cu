/*
Step 1
Every file should #include only the headers it needs (don't rely on #included includes)
Remove commented code (maybe there are exceptions)
Define the variables close to where you need them
Don't write comments that say what the code is doing (there might be rare exceptions)
Make the code simpler and nicer if you can =)
Change S3_3 etc to a meanding ful name describing the class

Step 2
Add kernels that do the following:

1)
data[i].x = data[i].x + data[i].y + data[i].z

template <template <class> class F>
struct Point {
    template<template <class> class F_new>
    operator Point<F_new>() { return {x, y, z}; }
    F<float> x;
    F<float> y;
    F<float> z;
};

2)
float h = 0.01f;  // small timestep
data[i].x = data[i].x + data[i].vx * h;
data[i].y = data[i].y + data[i].vy * h;
data[i].z = data[i].z + data[i].vz * h;

template <template <class> class F>
struct choose_me {
    template<template <class> class F_new>
    operator choose_me<F_new>() { return {x, y, z, vx, vy, vz}; }
    F<float> x;
    F<float> y;
    F<float> z;
    F<float> vx;
    F<float> vy;
    F<float> vz;
};
Further:
- Change order of member variables
- We could use Eigen::Vector3D for this.

3)
Checkout ot the BM_CPUHardRW example here and try to do something similar:
https://github.com/cern-nextgen/wp1.7-soa-benchmark/blob/main/benchmark.h
*/

#include <span>

#include <benchmark/benchmark.h>

#include "wrapper/decorator.h"
#include "wrapper/wrapper.h"

// #include "benchmark_gpu.h"
#include "benchmark_find_max_gpu.h"
#include "benchmark_estimate_pi_gpu.h"
#include "Sync_benchmark_add_gpu.h"
// #include "Sync_benchmark_64_add_gpu.h"
#include "Sync_benchmark_posvel_gpu.h"
#include "Sync_benchmark_posvel_shuf_gpu.h"
//#include "benchmark_bitonic_sort_gpu.h"


template <class T>
struct device_memory_array {
    device_memory_array(int N) : ptr(), N{N} { cudaMalloc((void**)&ptr, N * sizeof(T)); }
    ~device_memory_array() { if (ptr != nullptr) cudaFree(ptr); }
    __host__ operator std::span<T>() { return { ptr, ptr + N }; }
    constexpr T& operator[](int i) { return ptr[i]; }
    constexpr const T& operator[](int i) const { return ptr[i]; }
    T* ptr;
    int N;
};

template<wrapper::layout L>
struct CreateWrapperMax {
    wrapper::wrapper<s_max, device_memory_array, L> operator()(std::size_t n) {
        if constexpr (L == wrapper::layout::soa) return {n, n, n};
        else return {n};
    }
};

/*
template<wrapper::layout L>
struct CreateWrapper {
    wrapper::wrapper<S2, device_memory_array, L> operator()(int n) {
        if constexpr (L == wrapper::layout::soa) return {n, n};
        else return {{n}};
    }
};
*/

template<wrapper::layout L>
struct CreateWrapperCoor {
    wrapper::wrapper<s_coordinates, device_memory_array, L> operator()(std::size_t n) {
        if constexpr (L == wrapper::layout::soa) return {n, n};
        else return {n};
    }
};

template<wrapper::layout L>
struct CreateWrapperAdd {
    wrapper::wrapper<s_point, device_memory_array, L> operator()(std::size_t n) {
        if constexpr (L == wrapper::layout::soa) return {n, n, n};
        else return {n};
    }
};

template<wrapper::layout L>
struct CreateWrapperPosVel {
    wrapper::wrapper<s_posvel, device_memory_array, L> operator()(std::size_t n) {
        if constexpr (L == wrapper::layout::soa) return {n, n, n, n, n, n};
        else return {n};
    }
};

template<wrapper::layout L>
struct CreateWrapperPosVelShuf {
    wrapper::wrapper<s_posvel_shuf, device_memory_array, L> operator()(std::size_t n) {
        if constexpr (L == wrapper::layout::soa) return {n, n, n, n, n, n};
        else return {n};
    }
};
/*
template<wrapper::layout L>
struct CreateWrapper64Add {
    wrapper::wrapper<s_64_dim_point, device_memory_array, L> operator()(int n) {
        if constexpr (L == wrapper::layout::soa) return {n, n, n, n, n, n, n, n,
                                                         n, n, n, n, n, n, n, n,
                                                         n, n, n, n, n, n, n, n,
                                                         n, n, n, n, n, n, n, n,
                                                         n, n, n, n, n, n, n, n,   
                                                         n, n, n, n, n, n, n, n, 
                                                         n, n, n, n, n, n, n, n, 
                                                         n, n, n, n, n, n, n, n};
        else return {n};
    }
};
*/

int main(int argc, char** argv) {
    // constexpr int N[] = {1<<10, 1<<12, 1<<14, 1<<16, 1<<18, 1<<20};
    constexpr unsigned long long N_LONGLONG[] = {1ull<<16, 1ull<<18, 1ull<<20, 1ull<<22, 1ull<<24, 1ull<<26, 1ull<<28};

    /*
    for (int n : N) {
        using Create = CreateWrapper<wrapper::layout::soa>;
        using KernelInput = wrapper::wrapper<S2, std::span, wrapper::layout::soa>;
        benchmark::RegisterBenchmark("BM_GPUTest_SOA", BM_GPUTest<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    for (int n : N) {
        using Create = CreateWrapper<wrapper::layout::aos>;
        using KernelInput = wrapper::wrapper<S2, std::span, wrapper::layout::aos>;
        benchmark::RegisterBenchmark("BM_GPUTest_AOS", BM_GPUTest<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }
    */

    for (int n : N_LONGLONG) {
        using Create = CreateWrapperMax<wrapper::layout::soa>;
        using KernelInput = wrapper::wrapper<s_max, std::span, wrapper::layout::soa>;
        benchmark::RegisterBenchmark("MAX_GPUTest_SOA", MAX_GPUTest<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }
    
    for (int n : N_LONGLONG) {
        using Create = CreateWrapperMax<wrapper::layout::aos>;
        using KernelInput = wrapper::wrapper<s_max, std::span, wrapper::layout::aos>;
        benchmark::RegisterBenchmark("MAX_GPUTest_AOS", MAX_GPUTest<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    /*
    for (int n : N) {
        benchmark::RegisterBenchmark("BITONIC_Simp", BITONIC_Simp)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    } 
    */  

    for (int n : N_LONGLONG) {
        using Create = CreateWrapperCoor<wrapper::layout::soa>;
        using KernelInput = wrapper::wrapper<s_coordinates, std::span, wrapper::layout::soa>;
        benchmark::RegisterBenchmark("PiSimp_GPUTest_SOA", PiSimp_GPUTest<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    for (int n : N_LONGLONG) {
        using Create = CreateWrapperCoor<wrapper::layout::aos>;
        using KernelInput = wrapper::wrapper<s_coordinates, std::span, wrapper::layout::aos>;
        benchmark::RegisterBenchmark("PiSimp_GPUTest_AOS", PiSimp_GPUTest<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    for (unsigned long long n : N_LONGLONG) {
        using Create = CreateWrapperAdd<wrapper::layout::soa>;
        using KernelInput = wrapper::wrapper<s_point, std::span, wrapper::layout::soa>;
        benchmark::RegisterBenchmark("SYNC_GPUAdd_SOA", SYNC_GPUAdd<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }


    for (unsigned long long n : N_LONGLONG) {
        using Create = CreateWrapperAdd<wrapper::layout::aos>;
        using KernelInput = wrapper::wrapper<s_point, std::span, wrapper::layout::aos>;
        benchmark::RegisterBenchmark("SYNC_GPUAdd_AOS", SYNC_GPUAdd<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    /*
    for (unsigned long long n : N) {
        using Create = CreateWrapperAdd<wrapper::layout::soa>;
        using KernelInput = wrapper::wrapper<s_64_dim_point, std::span, wrapper::layout::soa>;
        benchmark::RegisterBenchmark("SYNC_GPU64Add_SOA", SYNC_GPU64Add<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }


    for (unsigned long long n : N) {
        using Create = CreateWrapperAdd<wrapper::layout::aos>;
        using KernelInput = wrapper::wrapper<s_64_dim_point, std::span, wrapper::layout::aos>;
        benchmark::RegisterBenchmark("SYNC_GPU64AddAOS", SYNC_GPU64Add<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }
    */

    for (unsigned long long n : N_LONGLONG) {
        using Create = CreateWrapperPosVel<wrapper::layout::soa>;
        using KernelInput = wrapper::wrapper<s_posvel, std::span, wrapper::layout::soa>;
        benchmark::RegisterBenchmark("SYNC_GPUPosVel_SOA", SYNC_GPUPosVel<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    for (unsigned long long n : N_LONGLONG) {
        using Create = CreateWrapperPosVel<wrapper::layout::aos>;
        using KernelInput = wrapper::wrapper<s_posvel, std::span, wrapper::layout::aos>;
        benchmark::RegisterBenchmark("SYNC_GPUPosVel_AOS", SYNC_GPUPosVel<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    for (int n : N_LONGLONG) {
        using Create = CreateWrapperPosVelShuf<wrapper::layout::soa>;
        using KernelInput = wrapper::wrapper<s_posvel_shuf, std::span, wrapper::layout::soa>;
        benchmark::RegisterBenchmark("SYNC_GPUPosVelShuf_SOA", SYNC_GPUPosVelShuf<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    for (int n : N_LONGLONG) {
        using Create = CreateWrapperPosVelShuf<wrapper::layout::aos>;
        using KernelInput = wrapper::wrapper<s_posvel_shuf, std::span, wrapper::layout::aos>;
        benchmark::RegisterBenchmark("SYNC_GPUPosVelShuf_AOS", SYNC_GPUPosVelShuf<Create, KernelInput>)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}