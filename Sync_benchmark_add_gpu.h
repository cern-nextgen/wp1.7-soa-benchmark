#ifndef SYNC_BENCHMARK_ADD_GPU_H
#define SYNC_BENCHMARK_ADD_GPU_H

#include <random>
#include <cfloat>

namespace benchmark { class State; }

template <template <class> class F>
struct s_point {
    template<template <class> class F_new>
    constexpr operator s_point<F_new>() { return {x, y, z}; }
    F<float> x;
    F<float> y;
    F<float> z;
};

template <class KernelInput>
__global__ void initialize_add(KernelInput data, float *d_x, const float *d_y, const float *d_z, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i].x = d_x[i];
        data[i].y = d_y[i];
        data[i].z = d_z[i];
    }
} 

template <class KernelInput>
__global__ void sync_test_add(KernelInput data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    data[idx].x = data[idx].x + data[idx].y + data[idx].z;
} 

template <class Create, class KernelInput>
void SYNC_GPUAdd(benchmark::State &state) {
    int n = state.range();
    state.counters["n_elem"] = n;

    unsigned int seed = 0;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0, 10);

    std::vector<float> h_x(n);
    std::vector<float> h_y(n);
    std::vector<float> h_z(n);

    for (int i = 0; i < n; i++) {
        h_x[i] = dist(rng);
        h_y[i] = dist(rng);
        h_z[i] = dist(rng);
    } 

    float *d_x, *d_y, *d_z;
    
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_z, n * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    auto t = Create()(n);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    initialize_add<KernelInput><<<numBlocks, blockSize>>>(t, d_x, d_y, d_z, n);

    cudaFree(d_y);
    cudaFree(d_z);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start, 0);

        sync_test_add<KernelInput><<<numBlocks, blockSize>>>(t, n);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
    }

    std::vector<float> h_x_copy(n);

    for (int i = 0; i < n; i++) {
        h_x_copy[i] = h_x[i];
    }

    cudaMemcpy(h_x.data(), t.x.ptr, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);

    for (int i = 0; i < n; i++) {
        if (h_x[i] = h_x[i] + h_y[i] + h_z[i]) {
            std::string message = "Wrong result at index " + std::to_string(i) + ": expected 2, got " + std::to_string(h_x[i]);
            state.SkipWithError(message);
        }
    }
}

#endif