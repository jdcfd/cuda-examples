# cuda-examples
Examples of GPU acceleration with CUDA and OpenACC in C++ and Fortran. 

## Requirements
- NVIDIA HPC SDK 23.1: These examples depend on different modules and libraries like CUDA, OpenACC, cuSPARSE and Thrust. These are included in the NVIDIA HPC SDKs and are easy to include/link with NVIDIA compilers.

## Description of examples
- __SpMV_Fortran_OACC__: UNDER DEVELOPMENT... will be an implementation of sparse matrix-vector multiplication using OpenACC directives in Fortran. TODO: Implement CUDA kernel to compute SpMV.
- __Thrust_interop__: Shows how to couple Fortran using with OpenACC directives with Thrust library, by using interfaces with CUDA Fortran.
- __cuSPARES_Fortran_example__: Shows how to perform sprase matrix multiplication on Fortran arrays ported to GPU using OpenACC, by creating a CUDA Fortran wrapper to cuSPARSE library.
