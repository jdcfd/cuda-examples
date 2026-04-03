cmake_minimum_required(VERSION 3.18)
project(sparse LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)  # adjust for your GPU

find_package(CUDAToolkit REQUIRED)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Library
add_library(sparse
    src/csr_matrix.cu
    src/dense_vector.cu
    src/kernels.cu
    src/spmv.cu
)

target_include_directories(sparse PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(sparse PUBLIC
    CUDA::cudart
)

set_target_properties(sparse PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Example / benchmark
add_executable(spmv_benchmark
    examples/benchmark/main.cu
    src/mmio_reader.cu
    third_party/mmio/mmio.c
)

set_source_files_properties(third_party/mmio/mmio.c PROPERTIES LANGUAGE C)

target_include_directories(spmv_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mmio
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(spmv_benchmark PRIVATE
    sparse
    CUDA::cusparse
)

set_target_properties(spmv_benchmark PROPERTIES
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# For helper_cuda.h (from CUDA samples)
target_include_directories(spmv_benchmark PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Optional: install the library
install(TARGETS sparse
    EXPORT sparseTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/sparse/ DESTINATION include/sparse)

message(STATUS "Build with: cmake -B build -S . && cmake --build build")
message(STATUS "Run with: ./build/spmv_benchmark data/your_matrix.mtx")

