#ifndef BENCHMARK_FIND_MAX_GPU_H
#define BENCHMARK_FIND_MAX_GPU_H

#include <vector>
#include <random>
#include <float.h>
#include <cuda_runtime.h>

namespace benchmark { class State; }

template <template <class> class F>
struct s_max {
    template<template <class> class F_new>
    constexpr operator s_max<F_new>() { return {x0, x1, x2}; }
    F<float>              x0;
    F<float>              x1;
    F<unsigned long long> x2;
};

template <class KernelInput>
__global__ void initialize_max(KernelInput data, const float *d_x0, unsigned long long N) {
    unsigned long long idx    = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (; idx < N; idx += stride) {
        data[idx].x0 = d_x0[idx];
        data[idx].x1 = 0.0f;
        data[idx].x2 = 0ull;
    }
}

__device__ __forceinline__
unsigned long long shfl_xor_ull(unsigned mask, unsigned long long v, int offset) {
    unsigned lo = (unsigned)(v & 0xFFFFFFFFull);
    unsigned hi = (unsigned)(v >> 32);
    lo = __shfl_xor_sync(mask, lo, offset);
    hi = __shfl_xor_sync(mask, hi, offset);
    return ((unsigned long long)hi << 32) | lo;
}

__device__ __forceinline__
void warp_reduce_max(float& val, unsigned long long& idx) {
    unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val              = __shfl_xor_sync(mask, val, offset);
        unsigned long long other_idx = shfl_xor_ull(mask, idx, offset);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
}

template <class KernelInput>
__global__ void arg_max(KernelInput data, unsigned long long N) {
    constexpr int warp_size = 32;
    __shared__ float              max_vals[warp_size];
    __shared__ unsigned long long max_idxs[warp_size];

    int tid     = threadIdx.x;
    int lane_id = tid % warp_size;
    int warp_id = tid / warp_size;

    float              local_max = -FLT_MAX;
    unsigned long long local_idx = 0ull;

    unsigned long long idx    = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;

    for (; idx < N; idx += stride) {
        float v = data[idx].x0;
        if (v > local_max) {
            local_max = v;
            local_idx = idx;
        }
    }

    warp_reduce_max(local_max, local_idx);

    if (lane_id == 0) {
        max_vals[warp_id] = local_max;
        max_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float              val = (tid < blockDim.x / warp_size) ? max_vals[lane_id] : -FLT_MAX;
        unsigned long long id  = (tid < blockDim.x / warp_size) ? max_idxs[lane_id] : 0ull;
        warp_reduce_max(val, id);
        if (lane_id == 0) {
            data[tid].x1 = val;
            data[tid].x2 = id;
        }
    }
}

template <class Create, class KernelInput>
void MAX_GPUTest(benchmark::State &state) {
    unsigned long long n = state.range();
    state.counters["n_elem"] = n;

    unsigned int seed = 0;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    std::vector<float> h_x0(n);
    for (unsigned long long i = 0; i < n; i++) h_x0[i] = dist(rng);

    float* d_x0 = nullptr;
    cudaMalloc(&d_x0, n * sizeof(float));
    cudaMemcpy(d_x0, h_x0.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    auto t = Create()(n);

    int blockSize = 256;
    int numBlocks = (int)((n + blockSize - 1ull) / blockSize);

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
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0f);
    }
}

#endif
