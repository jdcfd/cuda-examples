cmake_minimum_required(VERSION 3.18)
project(dot_product LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)  # adjust for your GPU

find_package(CUDAToolkit REQUIRED)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Library
add_library(dot_lib
    src/dot.cu
)

target_include_directories(dot_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(dot_lib PUBLIC
    CUDA::cudart
)

set_target_properties(dot_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Example
add_executable(dot_example
    examples/main.cu
)

target_include_directories(dot_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dot_example PRIVATE
    dot_lib
    CUDA::cublas
)

get_target_property(CUBLAS_LIB CUDA::cublas IMPORTED_LOCATION)
get_filename_component(CUBLAS_LIB_DIR "${CUBLAS_LIB}" DIRECTORY)

set_target_properties(dot_example PROPERTIES
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR};${CUBLAS_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# For helper_cuda.h (from CUDA samples) if needed
target_include_directories(dot_example PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)


# Profiling target
add_custom_target(profile
    COMMAND ncu --set full -k regex:dot_kernel_atomic -o dot_atomic_profile -c 1 -s 1 -f ./dot_example
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS dot_example
    COMMENT "Profiling dot_kernel_atomic with Nsight Compute"
)

message(STATUS "Build with: cmake -B build -S . && cmake --build build")
message(STATUS "Run with: ./build/dot_example")
message(STATUS "Profile with: cmake --build build --target profile")

