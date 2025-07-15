#include <span>
#include <iostream>
#include <vector>

#include "benchmark_gpu.h"
#include "wrapper/factory.h"
#include "wrapper/wrapper.h"

template <template <class> class F>
struct S2 {
    template<template <class> class F_new>
    operator S2<F_new>() { return {x0, x1}; }
    F<int> x0, x1;
};

int main(int argc, char** argv) {

    std::vector<int *> data_pointers;

    for (int n : N) {
        data_pointers.emplace_back(nullptr);
        int * data = data_pointers.back();
        cudaError_t err = cudaMalloc((void**)&data, n * sizeof(int));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        benchmark::RegisterBenchmark("BM_GPUTest", BM_GPUTest<int*>, data)->Arg(n)->UseManualTime()->Unit(benchmark::kMillisecond);
    }

    for (int * data : data_pointers) { cudaFree(data); }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}