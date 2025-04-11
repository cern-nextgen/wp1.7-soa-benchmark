cmake_minimum_required(VERSION 3.10)

project(SoABenchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(benchmark REQUIRED)
find_package(Boost REQUIRED)

add_executable(soa_wrapper soa_wrapper.cpp)
target_include_directories(soa_wrapper PUBLIC wrapper /opt/eigen /workspace/benchmark)
target_link_libraries(soa_wrapper benchmark::benchmark)

add_executable(soa_boost soa_boost.cpp)
target_include_directories(soa_boost PUBLIC boost /opt/eigen /workspace/benchmark)
target_link_libraries(soa_boost benchmark::benchmark)

# add_executable(soa_refl soa_refl.cpp)

link_libraries(
    benchmark::benchmark
    Eigen::Core
    Boost::boost
)
