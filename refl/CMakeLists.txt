if (ENABLE_EDG)
    add_benchmark_executable(soa_token_injection soa_token_injection.cpp)
    target_include_directories(soa_token_injection PUBLIC reflmempp/edg/include reflmempp/edg/include/kokkos/include/mdspan )
endif()

if (ENABLE_CLANG_P2996)
    add_benchmark_executable(soa_wrapper_refl soa_wrapper_refl.cpp)
    target_include_directories(soa_wrapper_refl PUBLIC reflmempp/clang-p2996 ../)

    add_benchmark_executable(soa_struct_splitter soa_struct_splitter.cpp)
    target_include_directories(soa_struct_splitter PUBLIC reflmempp/clang-p2996 ../)
endif()