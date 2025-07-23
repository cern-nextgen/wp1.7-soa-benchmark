#ifndef BENCHMARK_ESTIMATE_PI_GPU_H
#define BENCHMARK_ESTIMATE_PI_GPU_H

#include <span>

#include <Eigen/Core>

#include "wrapper/wrapper.h"

#include <random>
#include <chrono>
#include <cfloat>

namespace benchmark { class State; }

template <template <class> class F>
struct S3_2 {
    template<template <class> class F_new>
    operator S3_2<F_new>() { return {x_axis, y_axis, pi_estimate}; }
    F<float> x_axis;
    F<float> y_axis;
    F<float> pi_estimate;
};


#define NUM_POINTS 1000000 // Number of random points to generate - Make higher for better accuarcy
// Define the block size for CUDA kernel execution
#define BLOCK_SIZE 256

__global__ void estimate_pi(float* x_axis, float* y_axis, float* pi_estimate, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        // Check if the point is inside the unit circle by using: x^2 + y^2 = r^2
        if (data[idx].x_axis * data[idx].x_axis + data[idx].y_axis * data[idx].y_axis <= 1.0f) {
            atomicAdd(pi_estimate, 1.0f);
        }
    }
}


void PiSimp_GPUTest(benchmark::State &state) {
    int n = state.range();
    wrapper::wrapper<S3_2, device_memory_array, wrapper::layout::soa> t = {n, n, n};

    // Set up randome input generation
    unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1, 1);

    cudaMemset(t.x_axis.ptr, dist(rng), n * sizeof(float));
    cudaMemset(t.y_axis.ptr, dist(rng), n * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (auto _ : state) {
        cudaEventRecord(start, 0);

        estimate_pi<<<numBlocks, blockSize>>>(t, n);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
    }

    // Calculate the final estimate of pi
    data[0].pi_estimate = (data[0].pi_estimate / NUM_POINTS) * 4.0f;            
    // std::cout << "Estimated value of Pi: " << data[0].pi_estimate << std::endl;

    state.counters["n_elem"] = n;
}

#endif