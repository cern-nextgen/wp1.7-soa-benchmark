#include "kernel.h"

// #include <cuda/std/span>  // Should work out of the box


namespace kernel {

int cuda_malloc_managed(void** data, std::size_t size) { return cudaMallocManaged(data, size); }

int cuda_free(void* ptr) { return cudaFree(ptr); }

int cuda_malloc(void** d_data, std::size_t size) { return cudaMalloc(d_data, size); }

int cuda_memcpy(void* to, void* from, std::size_t size, cuda_memcpy_kind kind) {
    cudaError_t err;
    switch (kind) {
        case cuda_memcpy_kind::cudaMemcpyHostToDevice:
        err = cudaMemcpy(to, from, size, cudaMemcpyHostToDevice);
        break;
        case cuda_memcpy_kind::cudaMemcpyDeviceToHost:
        err = cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost);
        break;
        default:
        err = cudaError_t(-1);
    }
    return err;
}

__global__ void add(int* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += 1;
}

float apply(int* data, int N) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    add<<<1, 1>>>(data, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    return milliseconds;
}

}  // namespace kernel
