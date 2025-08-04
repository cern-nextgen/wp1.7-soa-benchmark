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
    operator S3_2<F_new>() { return {x_axis, y_axis, pi_counts}; }
    F<float> x_axis;
    F<float> y_axis;
    F<float> pi_counts;
};

template <class KernelInput>
__global__ void initialize_2(KernelInput data, float *d_x_axis, float *d_y_axis, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i].x_axis = d_x_axis[i];
        data[i].y_axis = d_y_axis[i];
        data[i].pi_counts = 0.0f;
    }
} 

template <class KernelInput>
__global__ void estimate_pi_kernel_shared(KernelInput data, int n) {
    extern __shared__ float local_counts[]; // Shared memory for local counts
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x; 

    // count local points in the unit circle
    int count = 0;
    if (idx < n) {
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
    atomicAdd(&data[0].pi_counts, temp_cnt);
    }
       
    // Ensure all threads have completed before exiting the kernel
    __syncthreads();   

    // Note: The final pi estimate will be calculated in the host code after kernel execution

}

template <class Create, class KernelInput>
void PiSimp_GPUTest(benchmark::State &state) {
    int n = state.range();
    state.counters["n_elem"] = n;
    // wrapper::wrapper<S3_2, device_memory_array, wrapper::layout::soa> t = {n, n, n};

    // Set up randome input generation
    unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1, 1);

    std::vector<float> h_x_axis(n);
    std::vector<float> h_y_axis(n);
    // std::vector<float> h_pi_counts(n, 0.0f); // host array to receive data
    float h_pi_counts = 0.0f;
    float h_pi_estimate = 0.0f;              // final estimate

    // Generate random values on host
    for (int i = 0; i < n; i++) {
        h_x_axis[i] = dist(rng);
        h_y_axis[i] = dist(rng);
    }

    float *d_x_axis, *d_y_axis; //, *d_pi_counts;
    cudaMalloc(&d_x_axis, n * sizeof(float));
    cudaMalloc(&d_y_axis, n * sizeof(float));
    // cudaMalloc(&d_pi_counts, n * sizeof(float));

    cudaMemcpy(d_x_axis, h_x_axis.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_axis, h_y_axis.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_pi_counts, h_pi_counts.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Copy to device
    // cudaMemcpy(t.x_axis.ptr, h_x_axis.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(t.y_axis.ptr, h_y_axis.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Print (from host)
    /*
    printf("First %d random x values:\n", n);
    for (int i = 0; i < n; i++) {
        printf("%f ", h_x[i]);
    }
    printf("\n");
    */
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto t = Create()(n);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    size_t shared_mem_size = blockSize * sizeof(float);
    initialize_2<KernelInput><<<numBlocks, blockSize>>>(t, d_x_axis, d_y_axis, n);

    for (auto _ : state) {
        cudaMemset(&t[0].pi_counts, 0, sizeof(float));
        // cudaMemset(h_pi_estimate.ptr, 0, sizeof(float));
        h_pi_estimate = 0.0f;

        cudaEventRecord(start, 0);

        estimate_pi_kernel_shared<KernelInput><<<numBlocks, blockSize, shared_mem_size>>>(t, n);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
    }

    cudaMemcpy(&h_pi_counts, &t[0].pi_counts, sizeof(float), cudaMemcpyDeviceToHost);

    h_pi_estimate = (h_pi_counts / n) * 4.0f;

    printf("Estimated value of Pi: %f \n", h_pi_estimate);

    cudaFree(d_x_axis);
    cudaFree(d_y_axis);

}

#endif