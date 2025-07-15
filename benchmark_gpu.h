#ifndef BENCHMARK_GPU_H
#define BENCHMARK_GPU_H

#include <benchmark/benchmark.h>

#include <iostream>

constexpr int N[] = {1<<10, 1<<12, 1<<14, 1<<16, 1<<18, 1<<20};

template<class T>
__global__ void add(T data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += 1;
}

template<class T>
void BM_GPUTest(benchmark::State &state, T t) {
    int n = state.range(0);
    
    cudaError_t err = cudaMemset(t, 0, n * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemset failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(t);
        return;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (auto _ : state) {
        cudaEventRecord(start, 0);
        add<<<numBlocks, blockSize>>>(t, n);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
    }

    state.counters["n_elem"] = n;
}

#endif  // BENCHMARK_GPU_H