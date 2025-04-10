cmake_minimum_required(VERSION 3.10)

project(SoABenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(benchmark REQUIRED)
find_package(Eigen REQUIRED)

add_executable(soa_wrapper soa_wrapper.cpp)
add_executable(soa_boost soa_boost.cpp)
# add_executable(soa_refl soa_refl.cpp)

link_libraries(
    benchmark::benchmark
    Eigen::Core
    Eigen::Dense
)
