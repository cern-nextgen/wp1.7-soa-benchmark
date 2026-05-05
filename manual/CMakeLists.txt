add_executable(aos_manual_cpu aos_manual_cpu.cpp)
add_executable(soa_manual_cpu soa_manual_cpu.cpp)

if (BUILD_GPU)
    add_executable(aos_manual_gpu aos_manual_gpu.cu)
    add_executable(soa_manual_gpu soa_manual_gpu.cu)
endif()
