#ifndef SYNC_BENCHMARK_64_ADD_GPU_H
#define SYNC_BENCHMARK_64_ADD_GPU_H

#include <random>
#include <cfloat>

namespace benchmark { class State; }

template <template <class> class F>
struct s_64_dim_point {
    template<template <class> class F_new>
    constexpr operator s_64_dim_point<F_new>() {  // match s_point: non-const
        return {
            x0,  x1,  x2,  x3,  x4,  x5,  x6,  x7,
            x8,  x9,  x10, x11, x12, x13, x14, x15,
            x16, x17, x18, x19, x20, x21, x22, x23,
            x24, x25, x26, x27, x28, x29, x30, x31,
            x32, x33, x34, x35, x36, x37, x38, x39,
            x40, x41, x42, x43, x44, x45, x46, x47,
            x48, x49, x50, x51, x52, x53, x54, x55,
            x56, x57, x58, x59, x60, x61, x62, x63
        };
    }

    F<float> x0;  F<float> x1;  F<float> x2;  F<float> x3;
    F<float> x4;  F<float> x5;  F<float> x6;  F<float> x7;
    F<float> x8;  F<float> x9;  F<float> x10; F<float> x11;
    F<float> x12; F<float> x13; F<float> x14; F<float> x15;
    F<float> x16; F<float> x17; F<float> x18; F<float> x19;
    F<float> x20; F<float> x21; F<float> x22; F<float> x23;
    F<float> x24; F<float> x25; F<float> x26; F<float> x27;
    F<float> x28; F<float> x29; F<float> x30; F<float> x31;
    F<float> x32; F<float> x33; F<float> x34; F<float> x35;
    F<float> x36; F<float> x37; F<float> x38; F<float> x39;
    F<float> x40; F<float> x41; F<float> x42; F<float> x43;
    F<float> x44; F<float> x45; F<float> x46; F<float> x47;
    F<float> x48; F<float> x49; F<float> x50; F<float> x51;
    F<float> x52; F<float> x53; F<float> x54; F<float> x55;
    F<float> x56; F<float> x57; F<float> x58; F<float> x59;
    F<float> x60; F<float> x61; F<float> x62; F<float> x63;
};

// 64 independent inputs (d_x0 .. d_x63), mirroring initialize_add style
template <class KernelInput>
__global__ void initialize_64add(
    KernelInput data,
    const float* __restrict__ d_x0,  const float* __restrict__ d_x1,
    const float* __restrict__ d_x2,  const float* __restrict__ d_x3,
    const float* __restrict__ d_x4,  const float* __restrict__ d_x5,
    const float* __restrict__ d_x6,  const float* __restrict__ d_x7,
    const float* __restrict__ d_x8,  const float* __restrict__ d_x9,
    const float* __restrict__ d_x10, const float* __restrict__ d_x11,
    const float* __restrict__ d_x12, const float* __restrict__ d_x13,
    const float* __restrict__ d_x14, const float* __restrict__ d_x15,
    const float* __restrict__ d_x16, const float* __restrict__ d_x17,
    const float* __restrict__ d_x18, const float* __restrict__ d_x19,
    const float* __restrict__ d_x20, const float* __restrict__ d_x21,
    const float* __restrict__ d_x22, const float* __restrict__ d_x23,
    const float* __restrict__ d_x24, const float* __restrict__ d_x25,
    const float* __restrict__ d_x26, const float* __restrict__ d_x27,
    const float* __restrict__ d_x28, const float* __restrict__ d_x29,
    const float* __restrict__ d_x30, const float* __restrict__ d_x31,
    const float* __restrict__ d_x32, const float* __restrict__ d_x33,
    const float* __restrict__ d_x34, const float* __restrict__ d_x35,
    const float* __restrict__ d_x36, const float* __restrict__ d_x37,
    const float* __restrict__ d_x38, const float* __restrict__ d_x39,
    const float* __restrict__ d_x40, const float* __restrict__ d_x41,
    const float* __restrict__ d_x42, const float* __restrict__ d_x43,
    const float* __restrict__ d_x44, const float* __restrict__ d_x45,
    const float* __restrict__ d_x46, const float* __restrict__ d_x47,
    const float* __restrict__ d_x48, const float* __restrict__ d_x49,
    const float* __restrict__ d_x50, const float* __restrict__ d_x51,
    const float* __restrict__ d_x52, const float* __restrict__ d_x53,
    const float* __restrict__ d_x54, const float* __restrict__ d_x55,
    const float* __restrict__ d_x56, const float* __restrict__ d_x57,
    const float* __restrict__ d_x58, const float* __restrict__ d_x59,
    const float* __restrict__ d_x60, const float* __restrict__ d_x61,
    const float* __restrict__ d_x62, const float* __restrict__ d_x63,
    unsigned long long N)
{
    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride  = (unsigned long long)blockDim.x * gridDim.x;

    for (; idx < N; idx += stride) {
        data[idx].x0  = d_x0[idx];  data[idx].x1  = d_x1[idx];
        data[idx].x2  = d_x2[idx];  data[idx].x3  = d_x3[idx];
        data[idx].x4  = d_x4[idx];  data[idx].x5  = d_x5[idx];
        data[idx].x6  = d_x6[idx];  data[idx].x7  = d_x7[idx];
        data[idx].x8  = d_x8[idx];  data[idx].x9  = d_x9[idx];
        data[idx].x10 = d_x10[idx]; data[idx].x11 = d_x11[idx];
        data[idx].x12 = d_x12[idx]; data[idx].x13 = d_x13[idx];
        data[idx].x14 = d_x14[idx]; data[idx].x15 = d_x15[idx];

        data[idx].x16 = d_x16[idx]; data[idx].x17 = d_x17[idx];
        data[idx].x18 = d_x18[idx]; data[idx].x19 = d_x19[idx];
        data[idx].x20 = d_x20[idx]; data[idx].x21 = d_x21[idx];
        data[idx].x22 = d_x22[idx]; data[idx].x23 = d_x23[idx];
        data[idx].x24 = d_x24[idx]; data[idx].x25 = d_x25[idx];
        data[idx].x26 = d_x26[idx]; data[idx].x27 = d_x27[idx];
        data[idx].x28 = d_x28[idx]; data[idx].x29 = d_x29[idx];
        data[idx].x30 = d_x30[idx]; data[idx].x31 = d_x31[idx];

        data[idx].x32 = d_x32[idx]; data[idx].x33 = d_x33[idx];
        data[idx].x34 = d_x34[idx]; data[idx].x35 = d_x35[idx];
        data[idx].x36 = d_x36[idx]; data[idx].x37 = d_x37[idx];
        data[idx].x38 = d_x38[idx]; data[idx].x39 = d_x39[idx];
        data[idx].x40 = d_x40[idx]; data[idx].x41 = d_x41[idx];
        data[idx].x42 = d_x42[idx]; data[idx].x43 = d_x43[idx];
        data[idx].x44 = d_x44[idx]; data[idx].x45 = d_x45[idx];
        data[idx].x46 = d_x46[idx]; data[idx].x47 = d_x47[idx];

        data[idx].x48 = d_x48[idx]; data[idx].x49 = d_x49[idx];
        data[idx].x50 = d_x50[idx]; data[idx].x51 = d_x51[idx];
        data[idx].x52 = d_x52[idx]; data[idx].x53 = d_x53[idx];
        data[idx].x54 = d_x54[idx]; data[idx].x55 = d_x55[idx];
        data[idx].x56 = d_x56[idx]; data[idx].x57 = d_x57[idx];
        data[idx].x58 = d_x58[idx]; data[idx].x59 = d_x59[idx];
        data[idx].x60 = d_x60[idx]; data[idx].x61 = d_x61[idx];
        data[idx].x62 = d_x62[idx]; data[idx].x63 = d_x63[idx];
    }
}

template <class KernelInput>
__global__ void return_64add(KernelInput data, float* out, unsigned long long N) {
    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride  = (unsigned long long)blockDim.x * gridDim.x;
    for (; idx < N; idx += stride) out[idx] = data[idx].x0;  // same idea: write one primary output
}

// Four disjoint 16-sums; store in x0, x16, x32, x48 (no element reused across sums)
template <class KernelInput>
__global__ void sync_test_64add(KernelInput data, unsigned long long N) {
    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride  = (unsigned long long)blockDim.x * gridDim.x;

    for (; idx < N; idx += stride) {
        float s0 =
            data[idx].x0 + data[idx].x1 + data[idx].x2 + data[idx].x3 +
            data[idx].x4 + data[idx].x5 + data[idx].x6 + data[idx].x7 +
            data[idx].x8 + data[idx].x9 + data[idx].x10 + data[idx].x11 +
            data[idx].x12 + data[idx].x13 + data[idx].x14 + data[idx].x15;
        float s1 =
            data[idx].x16 + data[idx].x17 + data[idx].x18 + data[idx].x19 +
            data[idx].x20 + data[idx].x21 + data[idx].x22 + data[idx].x23 +
            data[idx].x24 + data[idx].x25 + data[idx].x26 + data[idx].x27 +
            data[idx].x28 + data[idx].x29 + data[idx].x30 + data[idx].x31;
        float s2 =
            data[idx].x32 + data[idx].x33 + data[idx].x34 + data[idx].x35 +
            data[idx].x36 + data[idx].x37 + data[idx].x38 + data[idx].x39 +
            data[idx].x40 + data[idx].x41 + data[idx].x42 + data[idx].x43 +
            data[idx].x44 + data[idx].x45 + data[idx].x46 + data[idx].x47;
        float s3 =
            data[idx].x48 + data[idx].x49 + data[idx].x50 + data[idx].x51 +
            data[idx].x52 + data[idx].x53 + data[idx].x54 + data[idx].x55 +
            data[idx].x56 + data[idx].x57 + data[idx].x58 + data[idx].x59 +
            data[idx].x60 + data[idx].x61 + data[idx].x62 + data[idx].x63;

        data[idx].x0  = s0;
        data[idx].x16 = s1;
        data[idx].x32 = s2;
        data[idx].x48 = s3;
    }
}

// Host driver mirrors your 3-element SYNC_GPUAdd (alloc 64 inputs, copy, init, loop, return)
template <class Create, class KernelInput>
void SYNC_GPU64Add(benchmark::State &state) {
    unsigned long long n = state.range();
    state.counters["n_elem"] = n;

    unsigned int seed = 0;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0, 10);

    // 64 host arrays
    std::vector<float> h_x0(n),  h_x1(n),  h_x2(n),  h_x3(n),  h_x4(n),  h_x5(n),  h_x6(n),  h_x7(n),
                       h_x8(n),  h_x9(n),  h_x10(n), h_x11(n), h_x12(n), h_x13(n), h_x14(n), h_x15(n),
                       h_x16(n), h_x17(n), h_x18(n), h_x19(n), h_x20(n), h_x21(n), h_x22(n), h_x23(n),
                       h_x24(n), h_x25(n), h_x26(n), h_x27(n), h_x28(n), h_x29(n), h_x30(n), h_x31(n),
                       h_x32(n), h_x33(n), h_x34(n), h_x35(n), h_x36(n), h_x37(n), h_x38(n), h_x39(n),
                       h_x40(n), h_x41(n), h_x42(n), h_x43(n), h_x44(n), h_x45(n), h_x46(n), h_x47(n),
                       h_x48(n), h_x49(n), h_x50(n), h_x51(n), h_x52(n), h_x53(n), h_x54(n), h_x55(n),
                       h_x56(n), h_x57(n), h_x58(n), h_x59(n), h_x60(n), h_x61(n), h_x62(n), h_x63(n);

    for (unsigned long long i = 0; i < n; i++) {
        h_x0[i]=dist(rng); h_x1[i]=dist(rng); h_x2[i]=dist(rng); h_x3[i]=dist(rng);
        h_x4[i]=dist(rng); h_x5[i]=dist(rng); h_x6[i]=dist(rng); h_x7[i]=dist(rng);
        h_x8[i]=dist(rng); h_x9[i]=dist(rng); h_x10[i]=dist(rng); h_x11[i]=dist(rng);
        h_x12[i]=dist(rng); h_x13[i]=dist(rng); h_x14[i]=dist(rng); h_x15[i]=dist(rng);
        h_x16[i]=dist(rng); h_x17[i]=dist(rng); h_x18[i]=dist(rng); h_x19[i]=dist(rng);
        h_x20[i]=dist(rng); h_x21[i]=dist(rng); h_x22[i]=dist(rng); h_x23[i]=dist(rng);
        h_x24[i]=dist(rng); h_x25[i]=dist(rng); h_x26[i]=dist(rng); h_x27[i]=dist(rng);
        h_x28[i]=dist(rng); h_x29[i]=dist(rng); h_x30[i]=dist(rng); h_x31[i]=dist(rng);
        h_x32[i]=dist(rng); h_x33[i]=dist(rng); h_x34[i]=dist(rng); h_x35[i]=dist(rng);
        h_x36[i]=dist(rng); h_x37[i]=dist(rng); h_x38[i]=dist(rng); h_x39[i]=dist(rng);
        h_x40[i]=dist(rng); h_x41[i]=dist(rng); h_x42[i]=dist(rng); h_x43[i]=dist(rng);
        h_x44[i]=dist(rng); h_x45[i]=dist(rng); h_x46[i]=dist(rng); h_x47[i]=dist(rng);
        h_x48[i]=dist(rng); h_x49[i]=dist(rng); h_x50[i]=dist(rng); h_x51[i]=dist(rng);
        h_x52[i]=dist(rng); h_x53[i]=dist(rng); h_x54[i]=dist(rng); h_x55[i]=dist(rng);
        h_x56[i]=dist(rng); h_x57[i]=dist(rng); h_x58[i]=dist(rng); h_x59[i]=dist(rng);
        h_x60[i]=dist(rng); h_x61[i]=dist(rng); h_x62[i]=dist(rng); h_x63[i]=dist(rng);
    }

    // 64 device arrays
    float *d_x0,*d_x1,*d_x2,*d_x3,*d_x4,*d_x5,*d_x6,*d_x7,
          *d_x8,*d_x9,*d_x10,*d_x11,*d_x12,*d_x13,*d_x14,*d_x15,
          *d_x16,*d_x17,*d_x18,*d_x19,*d_x20,*d_x21,*d_x22,*d_x23,
          *d_x24,*d_x25,*d_x26,*d_x27,*d_x28,*d_x29,*d_x30,*d_x31,
          *d_x32,*d_x33,*d_x34,*d_x35,*d_x36,*d_x37,*d_x38,*d_x39,
          *d_x40,*d_x41,*d_x42,*d_x43,*d_x44,*d_x45,*d_x46,*d_x47,
          *d_x48,*d_x49,*d_x50,*d_x51,*d_x52,*d_x53,*d_x54,*d_x55,
          *d_x56,*d_x57,*d_x58,*d_x59,*d_x60,*d_x61,*d_x62,*d_x63;

    // allocate & copy (verbatim, like your style)
    #define MALLOC_AND_COPY(dx,hx) do{ cudaMalloc(&dx, n*sizeof(float)); cudaMemcpy(dx, hx.data(), n*sizeof(float), cudaMemcpyHostToDevice);}while(0)
    MALLOC_AND_COPY(d_x0,h_x0);   MALLOC_AND_COPY(d_x1,h_x1);   MALLOC_AND_COPY(d_x2,h_x2);   MALLOC_AND_COPY(d_x3,h_x3);
    MALLOC_AND_COPY(d_x4,h_x4);   MALLOC_AND_COPY(d_x5,h_x5);   MALLOC_AND_COPY(d_x6,h_x6);   MALLOC_AND_COPY(d_x7,h_x7);
    MALLOC_AND_COPY(d_x8,h_x8);   MALLOC_AND_COPY(d_x9,h_x9);   MALLOC_AND_COPY(d_x10,h_x10); MALLOC_AND_COPY(d_x11,h_x11);
    MALLOC_AND_COPY(d_x12,h_x12); MALLOC_AND_COPY(d_x13,h_x13); MALLOC_AND_COPY(d_x14,h_x14); MALLOC_AND_COPY(d_x15,h_x15);
    MALLOC_AND_COPY(d_x16,h_x16); MALLOC_AND_COPY(d_x17,h_x17); MALLOC_AND_COPY(d_x18,h_x18); MALLOC_AND_COPY(d_x19,h_x19);
    MALLOC_AND_COPY(d_x20,h_x20); MALLOC_AND_COPY(d_x21,h_x21); MALLOC_AND_COPY(d_x22,h_x22); MALLOC_AND_COPY(d_x23,h_x23);
    MALLOC_AND_COPY(d_x24,h_x24); MALLOC_AND_COPY(d_x25,h_x25); MALLOC_AND_COPY(d_x26,h_x26); MALLOC_AND_COPY(d_x27,h_x27);
    MALLOC_AND_COPY(d_x28,h_x28); MALLOC_AND_COPY(d_x29,h_x29); MALLOC_AND_COPY(d_x30,h_x30); MALLOC_AND_COPY(d_x31,h_x31);
    MALLOC_AND_COPY(d_x32,h_x32); MALLOC_AND_COPY(d_x33,h_x33); MALLOC_AND_COPY(d_x34,h_x34); MALLOC_AND_COPY(d_x35,h_x35);
    MALLOC_AND_COPY(d_x36,h_x36); MALLOC_AND_COPY(d_x37,h_x37); MALLOC_AND_COPY(d_x38,h_x38); MALLOC_AND_COPY(d_x39,h_x39);
    MALLOC_AND_COPY(d_x40,h_x40); MALLOC_AND_COPY(d_x41,h_x41); MALLOC_AND_COPY(d_x42,h_x42); MALLOC_AND_COPY(d_x43,h_x43);
    MALLOC_AND_COPY(d_x44,h_x44); MALLOC_AND_COPY(d_x45,h_x45); MALLOC_AND_COPY(d_x46,h_x46); MALLOC_AND_COPY(d_x47,h_x47);
    MALLOC_AND_COPY(d_x48,h_x48); MALLOC_AND_COPY(d_x49,h_x49); MALLOC_AND_COPY(d_x50,h_x50); MALLOC_AND_COPY(d_x51,h_x51);
    MALLOC_AND_COPY(d_x52,h_x52); MALLOC_AND_COPY(d_x53,h_x53); MALLOC_AND_COPY(d_x54,h_x54); MALLOC_AND_COPY(d_x55,h_x55);
    MALLOC_AND_COPY(d_x56,h_x56); MALLOC_AND_COPY(d_x57,h_x57); MALLOC_AND_COPY(d_x58,h_x58); MALLOC_AND_COPY(d_x59,h_x59);
    MALLOC_AND_COPY(d_x60,h_x60); MALLOC_AND_COPY(d_x61,h_x61); MALLOC_AND_COPY(d_x62,h_x62); MALLOC_AND_COPY(d_x63,h_x63);
    #undef MALLOC_AND_COPY

    auto t = Create()(n);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    initialize_64add<KernelInput><<<numBlocks, blockSize>>>(
        t,
        d_x0,d_x1,d_x2,d_x3,d_x4,d_x5,d_x6,d_x7,
        d_x8,d_x9,d_x10,d_x11,d_x12,d_x13,d_x14,d_x15,
        d_x16,d_x17,d_x18,d_x19,d_x20,d_x21,d_x22,d_x23,
        d_x24,d_x25,d_x26,d_x27,d_x28,d_x29,d_x30,d_x31,
        d_x32,d_x33,d_x34,d_x35,d_x36,d_x37,d_x38,d_x39,
        d_x40,d_x41,d_x42,d_x43,d_x44,d_x45,d_x46,d_x47,
        d_x48,d_x49,d_x50,d_x51,d_x52,d_x53,d_x54,d_x55,
        d_x56,d_x57,d_x58,d_x59,d_x60,d_x61,d_x62,d_x63,
        n);

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start, 0);
        sync_test_64add<KernelInput><<<numBlocks, blockSize>>>(t, n);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0f);
    }

    cudaFree(d_x0);  cudaFree(d_x1);  cudaFree(d_x2);  cudaFree(d_x3);
    cudaFree(d_x4);  cudaFree(d_x5);  cudaFree(d_x6);  cudaFree(d_x7);
    cudaFree(d_x8);  cudaFree(d_x9);  cudaFree(d_x10); cudaFree(d_x11);
    cudaFree(d_x12); cudaFree(d_x13); cudaFree(d_x14); cudaFree(d_x15);
    cudaFree(d_x16); cudaFree(d_x17); cudaFree(d_x18); cudaFree(d_x19);
    cudaFree(d_x20); cudaFree(d_x21); cudaFree(d_x22); cudaFree(d_x23);
    cudaFree(d_x24); cudaFree(d_x25); cudaFree(d_x26); cudaFree(d_x27);
    cudaFree(d_x28); cudaFree(d_x29); cudaFree(d_x30); cudaFree(d_x31);
    cudaFree(d_x32); cudaFree(d_x33); cudaFree(d_x34); cudaFree(d_x35);
    cudaFree(d_x36); cudaFree(d_x37); cudaFree(d_x38); cudaFree(d_x39);
    cudaFree(d_x40); cudaFree(d_x41); cudaFree(d_x42); cudaFree(d_x43);
    cudaFree(d_x44); cudaFree(d_x45); cudaFree(d_x46); cudaFree(d_x47);
    cudaFree(d_x48); cudaFree(d_x49); cudaFree(d_x50); cudaFree(d_x51);
    cudaFree(d_x52); cudaFree(d_x53); cudaFree(d_x54); cudaFree(d_x55);
    cudaFree(d_x56); cudaFree(d_x57); cudaFree(d_x58); cudaFree(d_x59);
    cudaFree(d_x60); cudaFree(d_x61); cudaFree(d_x62); cudaFree(d_x63);

}

#endif
