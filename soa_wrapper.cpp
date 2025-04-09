#include "benchmark.h"
#include "helper.h"
#include "wrapper.h"
#include "skeleton.h"

template <class T>
using my_vector = std::vector<T>;

wrapper::wrapper<my_vector, S, wrapper::layout::soa> w{ S<my_vector>{my_vector<int>(10), my_vector<int>(10) }};
BENCHMARK_CAPTURE(BM_CPUMemoryIntensive, wrapper, w);

BENCHMARK_MAIN();
