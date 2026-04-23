# CG Library

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

# Integration: Pres_Poisson matrix, UF collection RHS, reference from SciPy/solution.txt
add_test(
    NAME cg_pres_poisson
    COMMAND $<TARGET_FILE:cg_benchmark>
        ${CMAKE_SOURCE_DIR}/SpMV_Cpp_CUDA/data/Pres_Poisson/Pres_Poisson.mtx
        ${CMAKE_SOURCE_DIR}/SpMV_Cpp_CUDA/data/Pres_Poisson/Pres_Poisson_b.mtx
        ${CMAKE_SOURCE_DIR}/SpMV_Cpp_CUDA/data/Pres_Poisson/solution.txt
)
set_tests_properties(cg_pres_poisson PROPERTIES
    TIMEOUT 3600
    LABELS "cg;gpu;integration"
)

