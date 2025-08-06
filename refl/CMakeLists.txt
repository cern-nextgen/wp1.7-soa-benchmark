# soa_refl needs to be in a separate directory because it requires a different
# compiler (EDG) and different CMake settings. Changing CMAKE_CXX_COMPILER
# for a specific target in the main CMakeLists.txt is not supported.
cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_COMPILER $ENV{EDG_BASE}/bin/edgcpfe)
set(EDG_LINKER $ENV{EDG_BASE}/lib/edg_prelink)
set(EDG_GCC_TO_USE gcc)
message(STATUS "Compiling EDG versions with ${CMAKE_CXX_COMPILER} and ${EDG_GCC_TO_USE}")
message(STATUS "Linking EDG versions with ${EDG_LINKER}")

set(CMAKE_DEPFILE_FLAGS_CXX "")
message(STATUS "CMAKE_DEPFILE_FLAGS_CXX: ${CMAKE_DEPFILE_FLAGS_CXX}")

set(EDG_CPFE_DEFAULT_OPTIONS --g++ --set_flag=reflection --c++26 --gnu 140200 -tlocal --no_strict_gnu --no_char8_t
                             --extern_inline)
set(EDG_DEFAULT_DEFINES __CHAR_BIT__=8  _POSIX_SOURCE)
set(EDG_C_TO_OBJ_DEFAULT_OPTIONS -Dsetjmp=_setjmp -Dva_copy=__va_copy -falign-functions=4 -march=skylake
                                 ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} ${CMAKE_CXX_FLAGS}
                                 -finline-functions)
separate_arguments(EDG_C_TO_OBJ_DEFAULT_OPTIONS)
set(EDG_C_TO_OBJ_LIBRARIES -shared-libgcc -lstdc++ -lgcc_s -lpthread -lm)

#### Get paths to includes and libraries ####

# Get the path to libgcc.a
execute_process(COMMAND ${EDG_GCC_TO_USE} -print-libgcc-file-name OUTPUT_VARIABLE LIBGCC_FILE
                OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND dirname ${LIBGCC_FILE} OUTPUT_VARIABLE LIBGCC_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)

# Get the path to libstdc++.so
execute_process(COMMAND ${EDG_GCC_TO_USE} -print-file-name=libstdc++.so OUTPUT_VARIABLE LIBSTDC_FILE
                OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND dirname ${LIBSTDC_FILE} OUTPUT_VARIABLE LIBSTDC_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)

# Get paths to g++ headers/libraries
file(READ $ENV{EDG_BASE}/lib/g++_incl_paths sys_incl_dirs)
string(REPLACE ":" ";" sys_incl_dirs ${sys_incl_dirs})
string(STRIP "${sys_incl_dirs}" sys_incl_dirs) # Remove trailing newline
separate_arguments(sys_incl_dirs)

#### Options that are set differently for edgcpfe than for g++ ####
unset(CMAKE_CXX_STANDARD)
set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} "")
set(CMAKE_CXX_FLAGS "")

# Replaces the default -isystem with --sys_include
set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "--sys_include=")

# Default: <CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>
set(CMAKE_CXX_COMPILE_OBJECT " <CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> <SOURCE>")

# Default: <CMAKE_CXX_COMPILER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>
set(CMAKE_CXX_LINK_EXECUTABLE
    "${EDG_LINKER} -c 'nm -og' <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <LINK_LIBRARIES> <OBJECTS> --")

# Do not ignore /usr/include paths when added to target_include_directories
unset(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
unset(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES)

# Do not add -Wl, rpath flags to the linker command line (edg_prelink does not support them)
set(CMAKE_SKIP_RPATH TRUE)
# Do not add -Wl, --dependency-file flag to the edg_prelink command
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)

###########

include(CMakePrintHelpers)

function(add_edg_executable target sources)
    add_executable(${target} ${sources})
    target_include_directories(${target} SYSTEM AFTER PUBLIC ${sys_incl_dirs} $ENV{EDG_BASE}/include)
    target_include_directories(${target} PUBLIC ${PROJECT_SOURCE_DIR} reflmempp/include
                                                reflmempp/include/kokkos/include/mdspan)

    target_compile_options(${target} PUBLIC ${EDG_CPFE_DEFAULT_OPTIONS})
    # target_compile_options(${target} PUBLIC --error_limit=1) # for debugging

    # By products
    target_compile_options(${target} PUBLIC --gen_c_file_name=CMakeFiles/${target}.dir/${target}.int.c
                                            --ii_file=CMakeFiles/${target}.dir/${target}.cpp.ii
                                            --template_info_file=CMakeFiles/${target}.dir/${target}.cpp.ti
                                            -LCMakeFiles/${target}.dir/${target}.diagnostics)

    target_compile_definitions(${target} PUBLIC ${EDG_DEFAULT_DEFINES})

    add_custom_command(
        TARGET ${target}
        PRE_BUILD
        COMMAND echo "CMakeFiles/${target}.dir/${target}.cpp.o: \\\\" > CMakeFiles/${target}.dir/${target}.cpp.o.d
        COMMAND ${CMAKE_CXX_COMPILER} $<TARGET_PROPERTY:${target},COMPILE_OPTIONS>
                "-D$<JOIN:$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>,;-D>"
                "-I$<JOIN:$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>,;-I>"
                --dependencies "${CMAKE_CURRENT_SOURCE_DIR}/$<JOIN:${sources},;${CMAKE_CURRENT_SOURCE_DIR}/>"
                | sed "s/^${target}.o://" | sed "'s/$$/ \\\\/'" >> CMakeFiles/${target}.dir/${target}.cpp.o.d
        COMMENT "Add dependencies to CMakeFiles/${target}.dir/${target}.cpp.o.d"
        BYPRODUCTS CMakeFiles/${target}.dir/${target}.cpp.o.d
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS
    )

    add_custom_command(
        TARGET ${target}
        PRE_LINK
        COMMAND ${EDG_GCC_TO_USE} ${EDG_C_TO_OBJ_DEFAULT_OPTIONS} -c CMakeFiles/${target}.dir/${target}.int.c
        COMMAND mv -f ${target}.int.o CMakeFiles/${target}.dir/${target}.cpp.o
        BYPRODUCTS CMakeFiles/${target}.dir/${target}.int.c CMakeFiles/${target}.dir/${target}.cpp.ii
                   CMakeFiles/${target}.dir/${target}.cpp.ti CMakeFiles/${target}.dir/${target}.cpp.o
                   CMakeFiles/${target}.dir/${target}.diagnostics
        COMMENT "Building CMakeFiles/${target}.dir/${target}.cpp.o"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    target_link_directories(${target} PUBLIC ${LIBGCC_PATH} ${LIBSTDC_PATH} $ENV{EDG_BASE}/lib)

    # target_link_directories isn't working for these directories
    target_link_options(${target} PUBLIC -L/usr/lib -L/lib -L/usr/lib/x86_64-linux-gnu)

    get_target_property(link_dirs ${target} LINK_DIRECTORIES)

    get_target_property(link_libs ${target} LINK_LIBRARIES)
    foreach(link_lib ${link_libs})
        # $<TARGET_FILE:${link_lib}>       --- full path to the tgt binary file.
        get_target_property(target_type ${link_lib} TYPE)
        if (target_type STREQUAL "STATIC_LIBRARY" OR target_type STREQUAL "SHARED_LIBRARY")
            list(APPEND link_lib_paths "$<TARGET_FILE:${link_lib}>")
        endif()
    endforeach()

    add_custom_command(
        TARGET ${target}
        POST_BUILD
        COMMAND ${EDG_GCC_TO_USE} ${EDG_C_TO_OBJ_DEFAULT_OPTIONS} "$<$<BOOL:${link_dirs}>:-L$<JOIN:${link_dirs},;-L>>"
                -Wl,--dependency-file=CMakeFiles/${target}.dir/link.d -o ${target}
                CMakeFiles/${target}.dir/${target}.cpp.o ${link_lib_paths} ${EDG_C_TO_OBJ_LIBRARIES} -lrt -lC
        COMMENT "Complete linking"
        BYPRODUCTS ${target} CMakeFiles/${target}.dir/link.d
        COMMAND_EXPAND_LISTS
    )
endfunction()

add_edg_executable(soa_refl soa_refl.cpp)
