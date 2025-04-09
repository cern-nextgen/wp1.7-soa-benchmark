#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <benchmark/benchmark.h>

// 2 data members, integers, 64 alignment, 10 elements
template <typename T>
void BM_CPUMemoryIntensive(benchmark::State &state, T &t) {
    constexpr auto n_elements = 10;
    constexpr auto repetitions = 1000000;

    for (auto _ : state) {
        for (size_t _; _ < repetitions; ++_) {
            for (int i = 0; i < n_elements; ++i) {
                t[i].x0 += 2;
                t[i].x1 += 2;
            }
        }
    }
}

#endif  // BENCHMARK_H