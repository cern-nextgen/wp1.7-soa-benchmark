#ifndef SYNC_BENCHMARK_POSVEL_GPU_H
#define SYNC_BENCHMARK_POSVEL_GPU_H

#include <random>
#include <cfloat>

namespace benchmark { class State; }

template <template <class> class F>
struct s_posvel {
    template<template <class> class F_new>
    constexpr operator s_posvel<F_new>() { return {x, y, z, vx, vy, vz}; }
    F<float> x;
    F<float> y;
    F<float> z;
    F<float> vx;
    F<float> vy;
    F<float> vz;
};

template <class KernelInput>
__global__ void initialize_posvel(KernelInput data, float *d_x, float *d_y, float *d_z, float *d_vx, float *d_vy, float *d_vz, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i].x = d_x[i];
        data[i].y = d_y[i];
        data[i].z = d_z[i];
        data[i].vx = d_vx[i];
        data[i].vy = d_vy[i];
        data[i].vz = d_vz[i];
    }
} 

template <class KernelInput>
__global__ void sync_test_pos_vel(KernelInput data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float h = 0.1f;

    data[idx].x = data[idx].x + data[idx].vx * h;
    data[idx].y = data[idx].y + data[idx].vy * h;
    data[idx].z = data[idx].z + data[idx].vz * h;
} 

template <class Create, class KernelInput>
void SYNC_GPUPosVel(benchmark::State &state) {
    int n = state.range();
    state.counters["n_elem"] = n;

    unsigned int seed = 0;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0, 10);

    std::vector<float> h_x(n);
    std::vector<float> h_y(n);
    std::vector<float> h_z(n);

    std::vector<float> h_vx(n);
    std::vector<float> h_vy(n);
    std::vector<float> h_vz(n);

    for (int i = 0; i < n; i++) {
        h_x[i] = dist(rng);
        h_y[i] = dist(rng);
        h_z[i] = dist(rng);

        h_vx[i] = dist(rng);
        h_vy[i] = dist(rng);
        h_vz[i] = dist(rng);
    } 

    float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
    
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_z, n * sizeof(float));

    cudaMalloc(&d_vx, n * sizeof(float));
    cudaMalloc(&d_vy, n * sizeof(float));
    cudaMalloc(&d_vz, n * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_vx, h_vx.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    auto t = Create()(n);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    initialize_posvel<KernelInput><<<numBlocks, blockSize>>>(t, d_x, d_y, d_z, d_vx, d_vy, d_vz, n);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start, 0);

        sync_test_pos_vel<KernelInput><<<numBlocks, blockSize>>>(t, n);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
    }
}

#endif