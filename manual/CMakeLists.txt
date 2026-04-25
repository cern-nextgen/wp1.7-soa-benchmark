add_executable(aos_manual aos_manual.cpp)
add_executable(soa_manual soa_manual.cpp)
target_compile_options(soa_manual PRIVATE -freflection)
