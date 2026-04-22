cmake_minimum_required(VERSION 4.0)

project(SoABenchmark)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-O3 -ftree-vectorize -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize \
-Rpass-missed=loop-vectorize -march=native -funroll-loops -ffast-math -funsafe-math-optimizations -ftime-report")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-O3 -ftree-vectorize -fopt-info-vec-missed -march=native -funroll-loops -ffast-math -ftime-report")
endif()

find_package(benchmark REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
link_libraries(benchmark::benchmark Eigen3::Eigen)


add_executable(soa_wrapper soa_wrapper.cpp)
target_include_directories(soa_wrapper PUBLIC ${CMAKE_SOURCE_DIR}/memlayout)

add_executable(soa_manual soa_manual.cpp)

add_executable(aos_manual aos_manual.cpp)

#add_subdirectory(refl)