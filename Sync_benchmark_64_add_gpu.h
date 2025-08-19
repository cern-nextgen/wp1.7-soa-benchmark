#ifndef SYNC_BENCHMARK_64_ADD_GPU_H
#define SYNC_BENCHMARK_64_ADD_GPU_H

#include <random>
#include <cfloat>

namespace benchmark { class State; }

constexpr int KCOMP = 64;

template <template <class> class F, int K = KCOMP>
struct s_64_dim_point {
    template<template <class> class F_new>
    constexpr operator s_64_dim_point<F_new, K>() const {
        s_64_dim_point<F_new, K> out;
        #pragma unroll
        for (int i = 0; i < K; ++i) out.v[i] = v[i];
        return out;
    }
    F<float> v[K];
};

template <template <class> class F>
using s_point64 = s_64_dim_point<F, KCOMP>;

template <class KernelInput, int K = KCOMP>
__global__ void initialize_64_add(KernelInput data,
                                  float* __restrict__ d_out0,
                                  float* const* __restrict__ d_in_all,
                                  unsigned long long N) {
    unsigned long long idx    = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;

    for (; idx < N; idx += stride) {
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            data[idx].v[k] = d_in_all[k][idx];
        }
        if (d_out0) d_out0[idx] = data[idx].v[0];
    }
}

template <class KernelInput, int K = KCOMP>
__global__ void return_64_add(KernelInput data, float* out, unsigned long long N) {
    unsigned long long idx    = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (; idx < N; idx += stride) {
        out[idx] = data[idx].v[0];
    }
}

template <class KernelInput, int K = KCOMP>
__global__ void sync_test_64_add(KernelInput data, unsigned long long N) {
    unsigned long long idx    = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;

    for (; idx < N; idx += stride) {
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            acc += data[idx].v[k];
        }
        data[idx].v[0] = acc;
    }
}

// NOTE: We only template on Create. KernelInput is deduced from the actual wrapper type.
template <class Create>
void SYNC_GPU64Add(benchmark::State &state) {
    unsigned long long n = state.range();
    state.counters["n_elem"] = n;

    unsigned int seed = 0;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0, 10);

    // Host inputs: KCOMP arrays of length n
    std::vector<std::vector<float>> h_a(KCOMP, std::vector<float>(n));
    for (int k = 0; k < KCOMP; ++k) {
        for (unsigned long long i = 0; i < n; ++i) {
            h_a[k][i] = dist(rng);
        }
    }

    // Device inputs: KCOMP device arrays
    std::vector<float*> d_a(KCOMP, nullptr);
    for (int k = 0; k < KCOMP; ++k) {
        cudaMalloc(&d_a[k], n * sizeof(float));
        cudaMemcpy(d_a[k], h_a[k].data(), n * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Device array-of-pointers to pass to init kernel
    float** d_a_all = nullptr;
    cudaMalloc(&d_a_all, KCOMP * sizeof(float*));
    cudaMemcpy(d_a_all, d_a.data(), KCOMP * sizeof(float*), cudaMemcpyHostToDevice);

    // Buffer to mirror v[0] when returning results
    float* d_a_buffer = nullptr;
    cudaMalloc(&d_a_buffer, n * sizeof(float));

    // Build wrapper and deduce KernelInput from its actual type
    auto t = Create()(n);
    using KernelInput = decltype(t);

    const int blockSize = 256;
    const int numBlocks = static_cast<int>((n + blockSize - 1) / blockSize);

    // Initialize wrapper data from K inputs
    initialize_64_add<KernelInput, KCOMP><<<numBlocks, blockSize>>>(t, d_a_buffer, d_a_all, n);

    // Pointer array no longer needed after init
    cudaFree(d_a_all);

    // Timing loop
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start, 0);
        sync_test_64_add<KernelInput, KCOMP><<<numBlocks, blockSize>>>(t, n);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0f);
    }

    // Fetch v[0] and validate
    std::vector<float> h_a_buffer(n);
    return_64_add<KernelInput, KCOMP><<<numBlocks, blockSize>>>(t, d_a_buffer, n);
    cudaMemcpy(h_a_buffer.data(), d_a_buffer, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (unsigned long long i = 0; i < n; ++i) {
        float expected = 0.0f;
        for (int k = 0; k < KCOMP; ++k) expected += h_a[k][i];
        float got = h_a_buffer[i];

        if (fabsf(got - expected) > 1e-5f * fmaxf(1.0f, fabsf(expected))) {
            std::string msg = "Wrong result at index " + std::to_string(i)
                            + ": got " + std::to_string(got)
                            + ", expected " + std::to_string(expected);
            state.SkipWithError(msg.c_str());
            break;
        }
    }

    // Cleanup
    for (int k = 0; k < KCOMP; ++k) cudaFree(d_a[k]);
    cudaFree(d_a_buffer);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

#endif
