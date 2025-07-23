# soa_refl needs to be in a separate directory because it requires a different
# compiler (EDG) and different CMake settings. Changing CMAKE_CXX_COMPILER
# for a specific target in the main CMakeLists.txt is not supported.

set(CMAKE_CXX_COMPILER $ENV{EDG_BASE}/bin/eccp) # Set the EDG C++ compiler eccp
message(STATUS "Compiling EDG versions with ${CMAKE_CXX_COMPILER}")

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

# For some reason -lm is not automatically linked and it cannot be added
# to target_link_options because it is needs to be at the end
target_link_libraries(soa_refl /usr/lib/x86_64-linux-gnu/libm.so.6)

# Do not add -Wl, rpath flags to the linker command line (edg_prelink does not support them)
set(CMAKE_SKIP_RPATH TRUE)
# Do not add -Wl, --dependency-file flag to the linker command line
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)


