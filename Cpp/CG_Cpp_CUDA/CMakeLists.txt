cmake_minimum_required(VERSION 3.18)
project(cg_solver LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)  # adjust for your GPU

find_package(CUDAToolkit REQUIRED)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Import the SpMV project
# We use absolute paths to point to the sibling directory and a local binary dir.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../SpMV_Cpp_CUDA ${CMAKE_CURRENT_BINARY_DIR}/spmv)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Dot_Cpp_CUDA ${CMAKE_CURRENT_BINARY_DIR}/dot)

# CG Library
add_library(cg
    src/cg.cu
)

target_include_directories(cg PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cg PUBLIC
    sparse
    dot_lib
)

set_target_properties(cg PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Example / benchmark
add_executable(cg_benchmark
    examples/main.cu
    # Re-use the MTX reader from SpMV
    ${CMAKE_CURRENT_SOURCE_DIR}/../SpMV_Cpp_CUDA/src/mmio_reader.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../SpMV_Cpp_CUDA/third_party/mmio/mmio.c
)

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../SpMV_Cpp_CUDA/third_party/mmio/mmio.c PROPERTIES LANGUAGE C)

target_include_directories(cg_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../SpMV_Cpp_CUDA/third_party/mmio
    ${CMAKE_CURRENT_SOURCE_DIR}/../SpMV_Cpp_CUDA/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cg_benchmark PRIVATE
    cg
    sparse
    CUDA::cusparse
    CUDA::cublas
)

set_target_properties(cg_benchmark PROPERTIES
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

message(STATUS "Build with: cmake -B build -S . && cmake --build build")
message(STATUS "Run with: ./build/cg_benchmark ../SpMV_Cpp_CUDA/data/your_matrix.mtx")
