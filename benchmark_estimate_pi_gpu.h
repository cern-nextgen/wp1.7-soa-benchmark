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


#define NUM_POINTS 10000000 // Number of random points to generate - Make higher for better accuarcy
// Define the block size for CUDA kernel execution
#define BLOCK_SIZE 256


/*
__global__ void estimate_pi_kernel(wrapper::wrapper<S3_2, std::span, wrapper::layout::soa> data, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        // Check if the point is inside the unit circle by using: x^2 + y^2 = r^2
        if (data[idx].x_axis * data[idx].x_axis + data[idx].y_axis * data[idx].y_axis <= 1.0f) {
            atomicAdd(&data[idx].pi_estimate, 1.0f);
        }
    }
}
*/

__global__ void estimate_pi_kernel_shared(wrapper::wrapper<S3_2, std::span, wrapper::layout::soa> data, int num_points) {
    extern __shared__ float local_counts[]; // Shared memory for local counts
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // count local points in the unit circle
    int count = 0;
    if (idx < num_points) {
        float x = data[idx].x_axis;
        float y = data[idx].y_axis;
        if (x * x + y * y <= 1.0f) {
            count = 1;
        }
    }       
    
    // Store local count in shared memory
    local_counts[tid] = count;
    __syncthreads(); // Ensure all threads have written their counts

    // Block-wide parallel reduction to sum local counts
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            local_counts[tid] += local_counts[tid + stride];        
        }
        __syncthreads(); // Ensure all threads have completed their addition
    }  
    
    float temp_cnt = local_counts[0];

    // Calculate the total count of points in the unit circle for this block
    if (tid == 0) {
    atomicAdd(&data[0].pi_estimate, temp_cnt);
    }
       
    // Ensure all threads have completed before exiting the kernel
    __syncthreads();   

    // Note: The final pi estimate will be calculated in the host code after kernel execution

}


void PiSimp_GPUTest(benchmark::State &state) {
    int n = state.range();
    wrapper::wrapper<S3_2, device_memory_array, wrapper::layout::soa> t = {n, n, n};

    float* h_pi_estimate = 0;

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
    size_t shared_mem_size = blockSize * sizeof(float);

    for (auto _ : state) {
        cudaEventRecord(start, 0);

        estimate_pi_kernel_shared<<<numBlocks, blockSize, shared_mem_size>>>(t, n);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
    }

    cudaMemcpy(h_pi_estimate, t[0].pi_estimate.ptr, sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate the final estimate of pi
    h_pi_estimate[0] = (h_pi_estimate[0] / NUM_POINTS) * 4.0f;          
      
    printf("Estimated value of Pi: ", h_pi_estimate);

    state.counters["n_elem"] = n;
}

#endif