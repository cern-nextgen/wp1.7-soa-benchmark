cmake_minimum_required(VERSION 3.13)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

project(SoABenchmark LANGUAGES C CXX)

option(ENABLE_EDG "Build using the EDG frontend" OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build configuration
# ---------------------------------------------------------------------------

get_filename_component(BUILD_DIR_NAME "${CMAKE_BINARY_DIR}" NAME)
string(FIND "${BUILD_DIR_NAME}" "vtune" VTUNE_DIR_POS)

if(VTUNE_DIR_POS GREATER -1)
    message(STATUS "VTune build detected: debug symbols enabled")
    set(ENABLE_DEBUG_SYMBOLS ON)
else()
    message(STATUS "Release build: no debug symbols")
    set(ENABLE_DEBUG_SYMBOLS OFF)
endif()

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------

set(COMMON_OPT_FLAGS
    -O3
    -ftree-vectorize
    -march=native
    -funroll-loops
    -ffast-math
    -ftime-report
)

set(CLANG_OPT_FLAGS
    -Rpass-analysis=loop-vectorize
    -Rpass=loop-vectorize
    -Rpass-missed=loop-vectorize
    -funsafe-math-optimizations
)

set(GNU_OPT_FLAGS
    -fopt-info-vec-missed
)

# EDG generates intermediate C code that is compiled using GCC.
# Therefore, only GCC-compatible flags may be passed through
# CMAKE_CXX_FLAGS when ENABLE_EDG is enabled.
if(ENABLE_EDG)
    if(
        NOT DEFINED ENV{EDG_BASE}
        OR "$ENV{EDG_BASE}" STREQUAL ""
        OR NOT EXISTS "$ENV{EDG_BASE}/lib/g++_incl_paths"
    )
        message(
            FATAL_ERROR
            "ENABLE_EDG=ON requires a valid EDG installation.\n"
            "Set EDG_BASE to the EDG root directory so that:\n"
            "  $EDG_BASE/lib/g++_incl_paths\n"
            "exists."
        )
    endif()

    message(STATUS "EDG enabled")
    message(STATUS "EDG_BASE=$ENV{EDG_BASE}")

    set(EDG_BACKEND_FLAGS
        ${COMMON_OPT_FLAGS}
        ${GNU_OPT_FLAGS}
        -flto
    )

    if(ENABLE_DEBUG_SYMBOLS)
        list(APPEND EDG_BACKEND_FLAGS -g)
    endif()

    # edg_config.cmake currently uses CMAKE_CXX_FLAGS when invoking
    # the GCC backend, so convert the list into a space-separated string.
    list(JOIN EDG_BACKEND_FLAGS " " CMAKE_CXX_FLAGS)

    include("${CMAKE_CURRENT_SOURCE_DIR}/edg_config.cmake")

else()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(STATUS "Configuring native Clang build")

        add_compile_options(
            ${COMMON_OPT_FLAGS}
            ${CLANG_OPT_FLAGS}
        )

    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "Configuring native GNU build")

        add_compile_options(
            ${COMMON_OPT_FLAGS}
            ${GNU_OPT_FLAGS}
        )

    else()
        message(
            WARNING
            "Compiler '${CMAKE_CXX_COMPILER_ID}' has no optimized flag set"
        )

        add_compile_options(-O3)
    endif()

    if(ENABLE_DEBUG_SYMBOLS)
        add_compile_options(-g)
    endif()
endif()

# ---------------------------------------------------------------------------
# Benchmark target helper
# ---------------------------------------------------------------------------

function(add_benchmark_executable target)
    if(ENABLE_EDG)
        add_edg_executable(${target} ${ARGN})
    else()
        add_executable(${target} ${ARGN})
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

find_package(benchmark REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

link_libraries(
    benchmark::benchmark
    Eigen3::Eigen
)

include(CheckIncludeFileCXX)
check_include_file_cxx("format" CXX_FORMAT_SUPPORT)

if(NOT CXX_FORMAT_SUPPORT)
    find_package(fmt REQUIRED)
    link_libraries(fmt::fmt)
    add_compile_definitions(USE_FMTLIB_POLYFILL)
endif()

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------

add_benchmark_executable(aos_boost aos_boost.cpp)
target_include_directories(
    aos_boost
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/boost"
)
target_link_libraries(aos_boost Boost::boost)

add_benchmark_executable(soa_wrapper soa_wrapper.cpp)
target_include_directories(
    soa_wrapper
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/memlayout"
)

add_benchmark_executable(soa_boost soa_boost.cpp)
target_include_directories(
    soa_boost
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/boost"
)
target_link_libraries(soa_boost Boost::boost)

add_benchmark_executable(soa_manual soa_manual.cpp)
add_benchmark_executable(aos_manual aos_manual.cpp)

# ---------------------------------------------------------------------------
# Reflection support
# ---------------------------------------------------------------------------

if(ENABLE_EDG)
    add_subdirectory(refl)
endif()