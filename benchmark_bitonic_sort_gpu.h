///
#ifndef BENCHMARK_BITONIC_SORT_GPU_H
#define BENCHMARK_BITONIC_SORT_GPU_H

#include <span>

#include <Eigen/Core>

#include "wrapper/wrapper.h"

#include <random>
#include <chrono>
#include <cfloat>
#include <iostream>



/* 
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

__global__ void basic_bitonic_sort(wrapper::wrapper<S3_3, std::span, wrapper::layout::soa> data, int n) {
    extern __shared__ int shared_data[];

    int tid = threadIdx.x;

    if (tid < n) {
        shared_data[tid] = data[tid].x2;
    }

    __syncthreads();

    for (int size = 2; size <= n; size *= 2) {
        for (int stride = size / 2; stride > 0; stride /= 2) {
            int pair_id = tid ^ stride;
            if (pair_id > tid && pair_id < n) {
                bool ascending = ((tid & size) == 0);
                int a = shared_data[tid];
                int b = shared_data[pair_id];

                if ((ascending && a > b) || (!ascending && a < b)) {
                    shared_data[tid] = b;
                    shared_data[pair_id] = a;
                }
            }
            __syncthreads();
        }
    }

    if (tid < n) {
        data[tid].x2 = shared_data[tid];
    }
}

void BITONIC_Simp(benchmark::State &state) {
    int n = state.range();
    n = std::min(n, 1024); // Since this version only allows for a single block!!!  => 1024 threads => max 1024 elememts

    wrapper::wrapper<S3_3, device_memory_array, wrapper::layout::soa> t = {n, n, n};

    // Set up randome input generation
    unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 10);

    std::vector<int> h_x(n);

    // Generate random values on host
    for (int i = 0; i < n; i++) {
        h_x[i] = dist(rng);
    }

    // Copy to device
    cudaMemcpy(t.x2.ptr, h_x.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = n;   // one thread per element
    int numBlocks = 1;   // single block
    size_t shared_mem_size = n * sizeof(int);

    for (auto _ : state) {
        cudaEventRecord(start, 0);

        basic_bitonic_sort<<<numBlocks, blockSize, shared_mem_size>>>(t, n);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
    }

    cudaMemcpy(h_x.data(), t.x2.ptr, n * sizeof(int), cudaMemcpyDeviceToHost);\

    // Check for result correctness
    for (int i = 1; i < n; ++i) {
        if (h_x[i] < h_x[i - 1]) {
            std::cerr << "Error: Array not sorted correctly!\n";
            return;
        }
    }

    state.counters["n_elem"] = n;

}

#endif