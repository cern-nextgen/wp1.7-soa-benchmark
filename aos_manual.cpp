#include <iostream>

#include <Eigen/Core>

#include "benchmark.h"

struct Point { int x0, x1; };
using S2 = Point*;


int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    std::vector<Point*> free_list;

    for (auto n : N) {
        S2 t = new Point[n];
        free_list.push_back(t);
        benchmark::RegisterBenchmark("BM_CPUEasyRW", BM_CPUEasyRW<S2>, t)->Arg(n)->Unit(benchmark::kMillisecond);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    for (Point* buffer : free_list) delete[] buffer;

    return 0;
}