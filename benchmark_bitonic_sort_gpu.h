///
#ifndef BENCHMARK_BITONIC_SORT_GPU_H
#define BENCHMARK_BITONIC_SORT_GPU_H

#include <span>

#include <Eigen/Core>

#include "wrapper/wrapper.h"

#include <random>
#include <chrono>
#include <cfloat>

/*
    PLACEHOLDER FOR TEST - IS NOT YET IMPLEMENTED    

    OBS - this version does only work using a single block
*/

namespace benchmark { class State; }

// OBS: Only uses the first parameter for sorting
template <template <class> class F>
struct S3_3 {
    template<template <class> class F_new>
    operator S3_3<F_new>() { return {x0, x1, x2}; }
    F<float> x0;
    F<float> x1;
    F<int> x2;
};

__global__ void bitonic_sort(wrapper::wrapper<S3_3, std::span, wrapper::layout::soa> data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    data[idx].x0 = data[idx].x1 * data[idx].x0;
} 

void BITONIC_Simp(benchmark::State &state) {
    int n = state.range();
    wrapper::wrapper<S3_3, device_memory_array, wrapper::layout::soa> t = {n, n, n};

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

        // bitonic_sort<<<numBlocks, blockSize>>>(t, n);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
    }

    state.counters["n_elem"] = n;
}

#endif