#pragma once
#include <cuda_runtime.h>
#include <sparse/csr_matrix.cuh>
#include <sparse/dense_vector.cuh>

namespace linsolvers {

/**
 * Conjugate Gradient on the GPU. Matrix A must be symmetric positive-definite.
 * Overloads are provided for double (`CSRMatrix` / `DenseVector`) and single
 * (`CSRMatrixF` / `DenseVectorF`) precision. Tolerance is always `double`.
 */
int cg_solve(const sparse::CSRMatrix& A,
             const sparse::DenseVector& b,
             sparse::DenseVector& x,
             double tol = 1e-6,
             int max_iters = 1000,
             cudaStream_t stream = nullptr);

int cg_solve(const sparse::CSRMatrixF& A,
             const sparse::DenseVectorF& b,
             sparse::DenseVectorF& x,
             double tol = 1e-6,
             int max_iters = 1000,
             cudaStream_t stream = nullptr);

int pcg_solve(const sparse::CSRMatrix& A,
              const sparse::DenseVector& b,
              sparse::DenseVector& x,
              double tol = 1e-6,
              int max_iters = 1000,
              cudaStream_t stream = nullptr);

int pcg_solve(const sparse::CSRMatrixF& A,
              const sparse::DenseVectorF& b,
              sparse::DenseVectorF& x,
              double tol = 1e-6,
              int max_iters = 1000,
              cudaStream_t stream = nullptr);

} // namespace linsolvers
