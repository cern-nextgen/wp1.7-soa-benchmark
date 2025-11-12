cmake_minimum_required(VERSION 3.13)

project(SoABenchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-O3 -ftree-vectorize -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize \
-Rpass-missed=loop-vectorize -march=native -funroll-loops -ffast-math -funsafe-math-optimizations -ftime-report")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-O3 -ftree-vectorize -ffast-math -march=native -funroll-loops -g1")
endif()

find_package(benchmark REQUIRED)
find_package(likwid CONFIG REQUIRED COMPONENTS marker)

include(CheckIncludeFileCXX)
check_include_file_cxx("format" CXX_FORMAT_SUPPORT)
if (NOT CXX_FORMAT_SUPPORT)
    find_package(fmt REQUIRED)
    link_libraries(fmt::fmt)
    add_compile_definitions(USE_FMTLIB_POLYFILL)
endif()

link_libraries(benchmark::benchmark)
link_libraries(likwid::likwid)
include_directories(${LIKWID_INCLUDE_DIRS}) # if not set in likwid-config.cmake

add_executable(soa_manual soa_manual.cpp)
add_executable(aos_manual aos_manual.cpp)
