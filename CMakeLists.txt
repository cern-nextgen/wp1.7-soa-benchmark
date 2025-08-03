cmake_minimum_required(VERSION 3.10)
project(SoABenchmark)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect if build dir contains "vtune"
get_filename_component(BUILD_DIR_NAME ${CMAKE_BINARY_DIR} NAME)
string(FIND "${BUILD_DIR_NAME}" "vtune" VTUNE_DIR_POS)

if (VTUNE_DIR_POS GREATER -1)
    message(STATUS "VTune build detected (debug symbols enabled)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -ftree-vectorize -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -march=native -funroll-loops -ffast-math -funsafe-math-optimizations")
else()
    message(STATUS "Release build (no debug symbols)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -march=native -funroll-loops -ffast-math -funsafe-math-optimizations")
endif()

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

