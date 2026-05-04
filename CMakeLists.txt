cmake_minimum_required(VERSION 4.0)

project(SoABenchmark)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_GPU "Build CUDA GPU benchmark targets" ON)

set(CMAKE_CXX_FLAGS "-O3 -ftree-vectorize -march=native -funroll-loops -ffast-math") # -fopt-info-vec-missed -ftime-report -fno-tree-vectorize

if (NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER /opt/rh/gcc-toolset-14/root/usr/bin/g++)
endif()
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90)  # H100
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

find_package(benchmark REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
link_libraries(benchmark::benchmark Eigen3::Eigen)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(manual)
add_subdirectory(memlayout)
add_subdirectory(reflections)
