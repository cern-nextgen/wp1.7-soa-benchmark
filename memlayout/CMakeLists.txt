add_executable(soa_wrapper_cpu soa_wrapper_cpu.cpp)
add_executable(aos_wrapper_cpu aos_wrapper_cpu.cpp)

if (BUILD_GPU)
    add_executable(aos_wrapper_gpu aos_wrapper_gpu.cu)
    add_executable(soa_wrapper_gpu soa_wrapper_gpu.cu)
endif()
