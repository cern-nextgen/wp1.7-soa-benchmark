#ifndef BENCHMARK_FIND_MAX_GPU_H
#define BENCHMARK_FIND_MAX_GPU_H

#include <span>

#include <Eigen/Core>

#include "wrapper/wrapper.h"

#include <random>
#include <chrono>
#include <cfloat>

namespace benchmark { class State; }

template <template <class> class F>
struct S3_1 {
    template<template <class> class F_new>
    operator S3_1<F_new>() { return {x0, x1, x2}; }
    F<float> x0;
    F<float> x1;
    F<int> x2;
};

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

__global__ void arg_max(wrapper::wrapper<S3_1, std::span, wrapper::layout::soa> data, int N) {
    __shared__ float max_vals[32]; 
    __shared__ int max_idxs[32]; 

    int tid = threadIdx.x; 
    int lane_id = tid % 32; 
    int warp_id = tid / 32; 

    float local_max = -FLT_MAX; 
    int local_idx = -1; 

    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += gridDim.x * blockDim.x) { 
        float val = data[i].x0;
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
            data[tid].x1 = val; 
            data[tid].x2 = idx; 
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

#endif