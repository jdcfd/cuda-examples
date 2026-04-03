# cuda-examples
Examples of GPU acceleration with CUDA and OpenACC.

This repository primarily focuses on a High-performance CSR Sparse Matrix - Dense Vector multiplication (SpMV) in CUDA located in `Cpp/SpMV_Cpp_CUDA`.

Older examples (Fortran, Managed Memory) have been moved to the `archive/` directory.

## Requirements
- NVIDIA HPC SDK: These examples depend on different modules and libraries like CUDA, OpenACC, cuSPARSE and Thrust.

## Active Projects
- **[SpMV_Cpp_CUDA](Cpp/SpMV_Cpp_CUDA/README.md)**: High-performance CSR Sparse Matrix - Dense Vector multiplication (SpMV) in CUDA. It features a templated CSR SpMV kernel with optional shared-memory reduction and aims to be competitive with `cuSPARSE`. This forms the foundation for our upcoming GPU-accelerated Finite Element Method (FEM) linear solver.

## Archived Projects
Located in `archive/`:
- **SpMV_Fortran_OACC**: Implementation of sparse matrix-vector multiplication using OpenACC directives in Fortran.
- **Thrust_interop**: Shows how to couple Fortran using OpenACC directives with the Thrust library via CUDA Fortran interfaces.
- **cuSPARSE_Fortran_example**: Sparse matrix multiplication on Fortran arrays ported to GPU using OpenACC, wrapping the cuSPARSE library.
- **SpMV_Cpp_CUDA_ManagedMemory**: Earlier SpMV implementation using unified managed memory.
