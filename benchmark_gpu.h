#ifndef BENCHMARK_GPU_H
#define BENCHMARK_GPU_H

#include <string>

namespace benchmark { class State; }

template <template <class> class F>
struct S2 {
    template<template <class> class F_new>
    operator S2<F_new>() { return {x0, x1}; }
    F<int> x0, x1;
};

template <class KernelInput>
__global__ void initialize(KernelInput data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i].x0 = 1;
        data[i].x1 = 1;
    }
}

template <class KernelInput>
__global__ void add(KernelInput data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx].x0  = data[idx].x0 + data[idx].x1;
}

template <class KernelInput>
__global__ void copy_x0(KernelInput data, int * d_x0, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) d_x0[i] = data[i].x0;
}

template <class Create, class KernelInput>
void BM_GPUTest(benchmark::State &state) {
    int n = state.range(0);
    state.counters["n_elem"] = n;

    int blockSize = 32;
    int numBlocks = (n + blockSize - 1) / blockSize;

    auto t = Create()(n);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        initialize<KernelInput><<<numBlocks, blockSize>>>(t, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start, 0);
        add<KernelInput><<<numBlocks, blockSize>>>(t, n);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
    }

    int * d_x0;
    cudaMalloc(&d_x0, n * sizeof(int));
    copy_x0<KernelInput><<<numBlocks, blockSize>>>(t, d_x0, n);
    cudaDeviceSynchronize();

    std::vector<int> h_x0(n);
    cudaMemcpy(h_x0.data(), d_x0, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_x0);

    for (int i = 0; i < n; i++) {
        if (h_x0[i] != 2) {
            std::string message = "Wrong result at index " + std::to_string(i) + ": expected 2, got " + std::to_string(h_x0[i]);
            state.SkipWithError(message);
        }
    }
}

#endif  // BENCHMARK_GPU_H