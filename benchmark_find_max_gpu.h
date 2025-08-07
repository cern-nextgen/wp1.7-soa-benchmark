#ifndef BENCHMARK_FIND_MAX_GPU_H
#define BENCHMARK_FIND_MAX_GPU_H

#include <random>
#include <cfloat>

namespace benchmark { class State; }

template <template <class> class F>
struct s_max {
    template<template <class> class F_new>
    constexpr operator s_max <F_new>() { return {x0, x1, x2}; }
    F<float> x0;
    F<float> x1;
    F<int> x2;
};

template <class KernelInput>
__global__ void initialize_max(KernelInput data, float *d_x0, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i].x0 = d_x0[i];
        data[i].x1 = 0.0f;
        data[i].x2 = 0;
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

template <class KernelInput>
__global__ void arg_max(KernelInput data, int N) {
    constexpr int warp_size = 32;
    __shared__ float max_vals[warp_size]; 
    __shared__ int max_idxs[warp_size]; 

    int tid = threadIdx.x; 
    int lane_id = tid % warp_size; 
    int warp_id = tid / warp_size; 

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
        float val = (tid < blockDim.x / warp_size) ? max_vals[lane_id] : -FLT_MAX; 
        int idx = (tid < blockDim.x / warp_size) ? max_idxs[lane_id] : -1; 

        warp_reduce_max(val, idx); 

        if (lane_id == 0) { 
            data[tid].x1 = val; 
            data[tid].x2 = idx; 
        }
    } 
} 

template <class Create, class KernelInput>
void MAX_GPUTest(benchmark::State &state) {
    int n = state.range();
    state.counters["n_elem"] = n;

    unsigned int seed = 0;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0, 10);
    std::vector<float> h_x0(n);
    for (int i = 0; i < n; i++) h_x0[i] = dist(rng);

    float * d_x0;
    cudaMalloc(&d_x0, n * sizeof(float));
    cudaMemcpy(d_x0, h_x0.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    auto t = Create()(n);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    initialize_max<KernelInput><<<numBlocks, blockSize>>>(t, d_x0, n);
    
    cudaDeviceSynchronize();
    cudaFree(d_x0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start, 0);

        arg_max<KernelInput><<<numBlocks, blockSize>>>(t, n);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
    }
}

#endif