cmake_minimum_required(VERSION 3.10)

project(SoABenchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(benchmark REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(soa_wrapper soa_wrapper.cpp)
target_include_directories(soa_wrapper PUBLIC wrapper)
target_link_libraries(soa_wrapper benchmark::benchmark Eigen3::Eigen)

add_executable(soa_boost soa_boost.cpp)
target_include_directories(soa_boost PUBLIC boost)
target_link_libraries(soa_boost benchmark::benchmark Boost::boost Eigen3::Eigen)

# add_executable(soa_refl soa_refl.cpp)
