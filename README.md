# AoS vs. SoA benchmarks
TODO: move this branch to standalone repository

## Description
This branch contains benchmarks that compare the performance of Array-of-Structures (AoS) and Structure-of-Arrays (SoA)
layout for three kernels: Invariant Mass, N-Body, and Poisson equation solver. These mini-applications are implemented in `benchmark.h`.

In `aos_manual.cpp` and `soa_manual.cpp` we define the data structures used in the three mini-apps with an AoS and SoA
layout, respectively. In `aos_manual.cpp` each mini-app is only run with objects in AoS layout and in
`soa_manual.cpp` each mini-app is run with SoA layout.

Using `python3 impact_benchmark.py`, we run both `aos_manual` and `soa_manual` with a range of automatic code modifications.
Currently, the script runs two experiments in the main method:
```python
665: experiment_nmembers("perf_output_nmembers_im.csv", "im", precompiled=False)
666: experiment_stride("perf_output_stride_im.csv", "im", precompiled=False, wrap=False)
```
The first experiment, `experiment_nmembers`, modifies the number of data members in the object used for the Invariant
Mass benchmark `im`. On lines 527 and 528, we specify that we add between 0-25 data members before and after the members
that are read in the mini-app. **Decrease this range if you want to experiment with fewer configurations**. After
modifying the code, the script runs a subprocess that recompiles the c++ scripts and runs them to benchmark the Invariant
Mass mini-app with both AoS and SoA layout. The results are written to `perf_output_nmembers_im.csv`. Note that
the script also gather various hardware counters while running the benchmarks, defined in the `events` array at
the top of `impact_benchmark.py`. **Modify this to only contain performance events supported by your CPU**.

The second experiment, `experiment_stride`, modifies the stride used to compute the next index in the
computational loop of the Invariant Mass benchmark. The range of strides are defined on line 491. **Decrease this range if you want to experiment with fewer configurations**. Here, the scripts also recompiles and runs the aos/soa benchmarks.
The results are written to `perf_output_stride_im.csv`. The parameter `wrap=False` indicates that the array size
increases as the stride increases, so there is no index wrap-around.

Using `python3 plot_impact.py <dir>` a heatmap image can be created for the nmembers experiment and a line plot for the stride
experiment, using the previously gathered results stored in the `<dir>` directory. The resulting images are
also stored in this directory.

## Build
Prerequisite: the [Google Benchmark](https://github.com/google/benchmark) library is needed to build this repository.

To build the C++ files, you can use the following commands:
```shell
git clone https://github.com/cern-nextgen/wp1.7-soa-benchmark.git
cd wp1.7-soa-benchmark
git checkout jolly_branch
cmake . -Dbenchmark_DIR=</path/to/google/benchmark/>
```

## Docker Container
The following container contains the dependencies to build and run the benchmark.
Pull and run the container, then follow the steps of the section [Build and Run](#build-and-run).
```
docker pull registry.cern.ch/ngt-wp1.7/wp1.7-soa-benchmark:latest
docker run -it --rm registry.cern.ch/ngt-wp1.7/wp1.7-soa-benchmark:latest bash
```
The corresponding Dockerfile can be found here: [wp1.7-soa-benchmark-image](https://github.com/cern-nextgen/wp1.7-soa-benchmark-image)
