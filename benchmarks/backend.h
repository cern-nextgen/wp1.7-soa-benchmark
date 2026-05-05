#ifndef BENCHMARKS_BACKEND_H
#define BENCHMARKS_BACKEND_H

#include <cstddef>
#include <cstdlib>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define BACKEND_HOST_DEVICE __host__ __device__
#else
#define BACKEND_HOST_DEVICE
#endif

enum class Backend { CPU, GPU };

template <Backend B>
struct backend_allocator;

template <>
struct backend_allocator<Backend::CPU> {
    template <class T>
    static T* alloc(std::size_t n) {
        return static_cast<T*>(std::malloc(n * sizeof(T)));
    }
    template <class T>
    static void free(T* p) {
        std::free(p);
    }
    static void synchronize() {}
};

#ifdef __CUDACC__
template <>
struct backend_allocator<Backend::GPU> {
    template <class T>
    static T* alloc(std::size_t n) {
        T* p = nullptr;
        cudaMallocManaged(&p, n * sizeof(T));
        return p;
    }
    template <class T>
    static void free(T* p) {
        cudaFree(p);
    }
    static void synchronize() {
        cudaDeviceSynchronize();
    }
};
#endif

//////////////// parallel_for_n: serial loop on CPU, kernel launch on GPU

#ifdef __CUDACC__
template <class Body>
__global__ void parallel_for_n_kernel(std::size_t n, Body body) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) body(i);
}
#endif

template <Backend B, class Body>
inline void parallel_for_n(std::size_t n, Body body) {
    if constexpr (B == Backend::CPU) {
        for (std::size_t i = 0; i < n; ++i) body(i);
    } else {
#ifdef __CUDACC__
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        parallel_for_n_kernel<<<grid, block>>>(n, body);
        cudaDeviceSynchronize();
#endif
    }
}

#endif // BENCHMARKS_BACKEND_H
