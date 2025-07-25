# soa_refl needs to be in a separate directory because it requires a different
# compiler (EDG) and different CMake settings. Changing CMAKE_CXX_COMPILER
# for a specific target in the main CMakeLists.txt is not supported.

set(CMAKE_CXX_COMPILER ${EDG_CXX}) # Set the EDG C++ compiler eccp
message(STATUS "Compiling EDG versions with ${EDG_CXX} (EDG_CXX)")

set(CMAKE_DEPFILE_FLAGS_CXX "")

# These should be set in the build of eccp
set(CMAKE_CXX_FLAGS "")
unset(CMAKE_CXX_STANDARD)

# Replaces the default -isystem with --sys_include
set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "--sys_include=")

add_executable(soa_refl soa_refl.cpp)

target_include_directories(soa_refl PUBLIC ${PROJECT_SOURCE_DIR} reflmempp/include reflmempp/include/kokkos/include/mdspan)
target_compile_options(soa_refl PUBLIC -g --driver_debug --no_char8_t  --displ)
# target_compile_options(soa_refl PUBLIC -error-limit=0)

# For some reason this directory is not automatically included and
# target_include_directories/include_directories do not work
target_link_options(soa_refl PUBLIC -L/usr/lib/x86_64-linux-gnu/)
