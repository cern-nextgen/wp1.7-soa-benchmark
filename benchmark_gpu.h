#ifndef BENCHMARK_GPU_H
#define BENCHMARK_GPU_H

#include <span>

#include <Eigen/Core>

#include "wrapper/wrapper.h"

#include <random>
#include <chrono>
#include <cfloat>

namespace benchmark { class State; }

template <template <class> class F>
struct S2 {
    template<template <class> class F_new>
    operator S2<F_new>() { return {x0, x1}; }
    F<int> x0, x1;
};

template <template <class> class F>
struct S3 {
    template<template <class> class F_new>
    operator S3<F_new>() { return {x0, x1, x2}; }
    F<float> x0;
    F<float> x1;
    F<int> x2;
};

template <template <class> class F>
struct S10 {
    template <template <class> class F_new>
    operator S10<F_new>() { return {x0, x1, x2, x3, x4, x5, x6, x7, x8, x9}; }
    F<float> x0, x1;
    F<double> x2, x3;
    F<int> x4, x5;
    F<Eigen::Vector3d> x6, x7;
    F<Eigen::Matrix3d> x8, x9;
};

template <template <class> class F>
struct S64 {
    template <template <class> class F_new>
    operator S64<F_new>() { return {
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,
        x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25,
        x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38,
        x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50,
        x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63
    }; }
    F<float> x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
    F<double> x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25;
    F<int> x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38;
    F<Eigen::Vector3d> x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50;
    F<Eigen::Matrix3d> x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63;
};

template <class T>
struct device_memory_array {
    device_memory_array(int N) : ptr(), N{N} { cudaMalloc((void**)&ptr, N * sizeof(T)); }
    ~device_memory_array() { cudaFree(ptr); }
    operator std::span<T>() { return { ptr, ptr + N }; }
    __device__ T& operator[](int i) { return *(ptr + i); }
    __device__ const T& operator[](int i) const { return *(ptr + i); }
    T* ptr;
    int N;
};

__global__ void add(wrapper::wrapper<S2, std::span, wrapper::layout::soa> data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx].x0 += 1;
        data[idx].x1 += 1;
    }
}

__global__ void arit(wrapper::wrapper<S2, std::span, wrapper::layout::soa> data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (data[idx].x0 > 5) data[idx].x0 += 1;
        else data[idx].x1 += 1;
    }
}

__device__ __forceinline__ void warp_reduce_max(float& val, int& idx) { 
    unsigned mask = 0xffffffff; 
    #pragma unroll 
    for (int offset = 16; offset > 0; offset >>= 1) { 
        float other_val = __shfl_xor_sync(mask, val, offset); 
        int other_idx = __shfl_xor_sync(mask, idx, offset); 

        if (other_val > val || (other_val == val && other_idx < idx)) { 
            val = other_val; 
            idx = other_idx; 
        } 
    } 
} 

//__global__ void arg_max(const float* input, float* max_out, int* idx_out, int N) { 
__global__ void arg_max(wrapper::wrapper<S3, std::span, wrapper::layout::soa> data, int N) {
    __shared__ float max_vals[32]; 
    __shared__ int max_idxs[32]; 

    int tid = threadIdx.x; 
    int lane_id = tid % 32; 
    int warp_id = tid / 32; 

    float local_max = -FLT_MAX; 
    int local_idx = -1; 

    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += gridDim.x * blockDim.x) { 
        float val = data.x0[i];
        if (val > local_max) { 
            local_max = val; 
            local_idx = i; 
        } 
    } 

    warp_reduce_max(local_max, local_idx); 

    if (lane_id == 0) { 
        max_vals[warp_id] = local_max; 
        max_idxs[warp_id] = local_idx; 
    }
    __syncthreads(); 

    if (warp_id == 0) { 
        float val = (tid < blockDim.x / 32) ? max_vals[lane_id] : -FLT_MAX; 
        int idx = (tid < blockDim.x / 32) ? max_idxs[lane_id] : -1; 

        warp_reduce_max(val, idx); 

        if (lane_id == 0) { 
            data.x1[tid] = val; 
            data.x2[tid] = idx; 
        } 
    } 
} 

void MAX_GPUTest(benchmark::State &state) {
    int n = state.range();
    wrapper::wrapper<S3, device_memory_array, wrapper::layout::soa> t = {n, n, n};

    // Set up randome input generation
    unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0,10);

    cudaMemset(t.x0.ptr, dist(rng), n * sizeof(int));
    cudaMemset(t.x1.ptr, dist(rng), n * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (auto _ : state) {
        cudaEventRecord(start, 0);

        arg_max<<<numBlocks, blockSize>>>(t, n);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
    }

    state.counters["n_elem"] = n;
}

void ARIT_GPUTest(benchmark::State &state) {
    int n = state.range();
    wrapper::wrapper<S2, device_memory_array, wrapper::layout::soa> t = {n, n};

    // Set up randome input generation
    unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0,10);

    cudaMemset(t.x0.ptr, dist(rng), n * sizeof(int));
    cudaMemset(t.x1.ptr, dist(rng), n * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (auto _ : state) {
        cudaEventRecord(start, 0);
        arit<<<numBlocks, blockSize>>>(t, n);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
    }

    state.counters["n_elem"] = n;
}

void BM_GPUTest(benchmark::State &state) {
    int n = state.range(0);

    wrapper::wrapper<S2, device_memory_array, wrapper::layout::soa> t = {n, n};
    cudaMemset(t.x0.ptr, 0, n * sizeof(int));
    cudaMemset(t.x1.ptr, 0, n * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (auto _ : state) {
        cudaEventRecord(start, 0);
        add<<<numBlocks, blockSize>>>(t, n);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
    }

    state.counters["n_elem"] = n;
}

#endif  // BENCHMARK_GPU_H