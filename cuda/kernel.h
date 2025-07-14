#ifndef KERNEL_H
#define KERNEL_H

namespace kernel {

int cuda_malloc_managed(void** data, std::size_t size);

int cuda_free(void* ptr);

int cuda_malloc(void** d_data, std::size_t size);

enum class cuda_memcpy_kind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };

int cuda_memcpy(void* d_data, void* h_data, std::size_t size, cuda_memcpy_kind kind);

float apply(int* data, int N);

}  // namespace kernel

#endif  // KERNEL_H