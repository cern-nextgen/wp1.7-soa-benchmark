cmake_minimum_required(VERSION 4.0)

project(SoABenchmark)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-O3 -ftree-vectorize -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize \
-Rpass-missed=loop-vectorize -march=native -funroll-loops -ffast-math -funsafe-math-optimizations -ftime-report")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-O3 -ftree-vectorize -march=native -funroll-loops -ffast-math") # -fopt-info-vec-missed -ftime-report
endif()

find_package(benchmark REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
link_libraries(benchmark::benchmark Eigen3::Eigen)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(manual)
add_subdirectory(memlayout)
add_subdirectory(reflections)
