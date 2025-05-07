cmake_minimum_required(VERSION 3.10)

project(SoABenchmark)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3")

find_package(benchmark REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
link_libraries(benchmark::benchmark Eigen3::Eigen)

include(CheckIncludeFileCXX)
check_include_file_cxx("format" CXX_FORMAT_SUPPORT)
if (NOT CXX_FORMAT_SUPPORT)
    find_package(fmt REQUIRED)
    link_libraries(fmt::fmt)
    add_compile_definitions(USE_FMTLIB_POLYFILL)
endif()

add_executable(soa_wrapper soa_wrapper.cpp)
target_include_directories(soa_wrapper PUBLIC wrapper)

add_executable(soa_boost soa_boost.cpp)
target_include_directories(soa_boost PUBLIC boost)
target_link_libraries(soa_boost Boost::boost)

add_executable(soa_manual soa_manual.cpp)

# add_executable(soa_refl soa_refl.cpp)
