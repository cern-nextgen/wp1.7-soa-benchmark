cmake_minimum_required(VERSION 3.10)

project(SoABenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(benchmark REQUIRED)

add_executable(soa_wrapper soa_wrapper.cpp)
target_link_libraries(soa_wrapper PRIVATE benchmark::benchmark)

# add_executable(soa_cms soa_cms.cpp)
# target_link_libraries(soa_cms PRIVATE benchmark::benchmark)

# add_executable(soa_refl soa_refl.cpp)
# target_link_libraries(soa_refl PRIVATE benchmark::benchmark)
