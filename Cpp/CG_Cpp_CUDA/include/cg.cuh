#pragma once
#include <cuda_runtime.h>
#include <sparse/csr_matrix.cuh>
#include <sparse/dense_vector.cuh>

namespace linsolvers {

/**
 * Solve Ax = b using the Conjugate Gradient method on the GPU.
 * The matrix A must be symmetric positive-definite.
 *
 * @param A         The sparse matrix (CSR format, data must be on device).
 * @param b         The right-hand side vector (data must be on device).
 * @param x         The solution vector (initial guess on entry, solution on exit).
 * @param tol       The relative error tolerance for convergence.
 * @param max_iters The maximum number of iterations allowed.
 * @return          The number of iterations performed, or -1 if the solver
 *                  failed to converge within the maximum iterations.
 */
int cg_solve(const sparse::CSRMatrix& A, 
             const sparse::DenseVector& b, 
             sparse::DenseVector& x, 
             double tol = 1e-6, 
             int max_iters = 1000,
             cudaStream_t stream = nullptr);

int pcg_solve(const sparse::CSRMatrix& A, 
              const sparse::DenseVector& b, 
              sparse::DenseVector& x, 
              double tol = 1e-6, 
              int max_iters = 1000,
              cudaStream_t stream = nullptr);

} // namespace linsolvers
