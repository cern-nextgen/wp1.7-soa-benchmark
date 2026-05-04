add_executable(aos_manual aos_manual.cpp)
add_executable(soa_manual soa_manual.cpp)

if (BUILD_GPU)
    add_executable(aos_manual_gpu aos_manual_gpu.cu)
endif()
