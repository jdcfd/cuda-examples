# Sparse Matrix Library (`sparse`)
High-performance CSR Sparse Matrix - Dense Vector multiplication (SpMV) in CUDA.

This library is a cleaned-up and refactored version of an implementation originally inspired by the [AMD Lab Notes SpMV blog post](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-spmv-docs-spmv_part1/). It is competitive with (and in some cases faster than) `cuSPARSE`'s `cusparseSpMV`.

**Project Roadmap**: This module is serving as the foundation for an upcoming production-quality, GPU-accelerated Finite Element Method (FEM) linear solver.

## Features

- Clean, reusable C++ library under the `sparse` namespace
- CSR SpMV kernels in `float` and `double` (`sparse::multiply` overloads; types `CSRMatrix` / `DenseVector` vs `CSRMatrixF` / `DenseVectorF`)
- Optional shared-memory reduction
- Simple high-level API: `sparse::multiply(A, x, y)` (matrix and vector must use the same scalar type)
- Separate benchmark/example from the core library
- Modern CMake build system
- Easy to integrate into other CUDA projects

## Build commands

This project is part of a unified suite. To build everything at once, run from the `Cpp/` root:
```bash
cmake -B build -S .
cmake --build build -j
```

`CSRMatrixReader` can fill either `CSRMatrix` or `CSRMatrixF` from Matrix Market files (values are read as `double` from the file, then cast to `float` for the single-precision matrix).

Alternatively, you can still build it standalone:
```bash
cmake -B build -S .
cmake --build build --config Release
```


