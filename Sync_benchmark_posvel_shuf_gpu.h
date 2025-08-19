#ifndef SYNC_BENCHMARK_POSVEL_SHUF_GPU_H
#define SYNC_BENCHMARK_POSVEL_SHUF_GPU_H

#include <random>
#include <cfloat>

namespace benchmark { class State; }

template <template <class> class F>
struct s_posvel_shuf {
    template<template <class> class F_new>
    constexpr operator s_posvel_shuf<F_new>() { return {x, y, z, vx, vy, vz}; }
    F<float> x;
    F<float> vx;
    F<float> y;
    F<float> vy;
    F<float> z;
    F<float> vz;
};

template <class KernelInput>
__global__ void initialize_posvel_shuf(KernelInput data, float *d_x, float *d_vx, float *d_y, float *d_vy, float *d_z, float *d_vz, unsigned long long N) {

    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride  = (unsigned long long)blockDim.x * gridDim.x;

    for (; idx < N; idx += stride) {
        data[idx].x = d_x[idx];
        data[idx].vx = d_vx[idx];
        data[idx].y = d_y[idx];
        data[idx].vy = d_vy[idx];
        data[idx].z = d_z[idx];
        data[idx].vz = d_vz[idx];
    }
} 

template <class KernelInput>
__global__ void return_posvel_shuf(KernelInput data, float* x_out, float* y_out, float* z_out, unsigned long long N) {

    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride  = (unsigned long long)blockDim.x * gridDim.x;

    for (; idx < N; idx += stride) {
        x_out[idx] = data[idx].x;
        y_out[idx] = data[idx].y;
        z_out[idx] = data[idx].z;
    } 
}

template <class KernelInput>
__global__ void sync_test_pos_vel_shuf(KernelInput data, int N) {

    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride  = (unsigned long long)blockDim.x * gridDim.x;

    float h = 0.1f;

    for (; idx < N; idx += stride) {
        data[idx].x = data[idx].x + data[idx].vx * h;
        data[idx].y = data[idx].y + data[idx].vy * h;
        data[idx].z = data[idx].z + data[idx].vz * h;
    }
} 

template <class Create, class KernelInput>
void SYNC_GPUPosVelShuf(benchmark::State &state) {
    unsigned long long n = state.range();
    state.counters["n_elem"] = n;

    unsigned int seed = 0;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0, 10);

    std::vector<float> h_x(n);
    std::vector<float> h_vx(n);

    std::vector<float> h_y(n);
    std::vector<float> h_vy(n);

    std::vector<float> h_z(n);
    std::vector<float> h_vz(n);

    for (unsigned long long i = 0; i < n; i++) {
        h_x[i] = dist(rng);
        h_vx[i] = dist(rng);

        h_y[i] = dist(rng);
        h_vy[i] = dist(rng);

        h_z[i] = dist(rng);
        h_vz[i] = dist(rng);
    } 

    float *d_x, *d_vx, *d_y, *d_vy, *d_z, *d_vz;
    
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_vx, n * sizeof(float));

    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_vy, n * sizeof(float));

    cudaMalloc(&d_z, n * sizeof(float));
    cudaMalloc(&d_vz, n * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_z, h_z.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    auto t = Create()(n);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    initialize_posvel_shuf<KernelInput><<<numBlocks, blockSize>>>(t, d_x, d_vx, d_y, d_vy, d_z, d_vz, n);

    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start, 0);

        sync_test_pos_vel_shuf<KernelInput><<<numBlocks, blockSize>>>(t, n);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
    }

    std::vector<float> h_x_copy(n);
    std::vector<float> h_y_copy(n);
    std::vector<float> h_z_copy(n);

    for (unsigned long long i = 0; i < n; i++) {
        h_x_copy[i] = h_x[i];
        h_y_copy[i] = h_y[i];
        h_z_copy[i] = h_z[i];
    }

    return_posvel_shuf<KernelInput><<<numBlocks, blockSize>>>(t, d_x, d_y, d_z, n);

    cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z.data(), d_z, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    // float h = 0.0f;

    /*
    for (unsigned long long i = 0; i < n; i++) {
        if (h_x_copy[i] == h_x[i] + h_vx[i] * h) {
            std::string message = "Wrong result at index " + std::to_string(i) + ": got " + std::to_string(h_x[i]);
            state.SkipWithError(message);
        }
        if (h_y_copy[i] == h_y[i] + h_vy[i] * h) {
            std::string message = "Wrong result at index " + std::to_string(i) + ": got " + std::to_string(h_y[i]);
            state.SkipWithError(message);
        }
        if (h_z_copy[i] == h_z[i] + h_vz[i] * h) {
            std::string message = "Wrong result at index " + std::to_string(i) + ": got " + std::to_string(h_z[i]);
            state.SkipWithError(message);
        }
    }
    */
}

#endif

