cmake_minimum_required(VERSION 3.24)
project(SoABenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Allow "std::span" on the device
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# For clang
set(CMAKE_CXX_FLAGS "-O3 -ftree-vectorize -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -march=native -funroll-loops -ffast-math -funsafe-math-optimizations")

# For gcc
# set(CMAKE_CXX_FLAGS "-O3 -ftree-vectorize -fopt-info-vec -march=native -funroll-loops -ffast-math -funsafe-math-optimizations")

find_package(benchmark REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
link_libraries(benchmark::benchmark Eigen3::Eigen)

add_executable(soa_wrapper soa_wrapper.cpp)
target_include_directories(soa_wrapper PUBLIC wrapper)

add_executable(soa_manual soa_manual.cpp)

# CUDA ----------------------------------------
add_executable(kernel kernel.cu)
