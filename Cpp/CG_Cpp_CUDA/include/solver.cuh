#pragma once
#include <cuda_runtime.h>
#include <sparse/csr_matrix.cuh>
#include <sparse/dense_vector.cuh>
#include "cg.cuh"

namespace linsolvers {

enum class SolverType {
    CG,
    PCG_JACOBI
};

inline int solve(const sparse::CSRMatrix& A, 
                 const sparse::DenseVector& b, 
                 sparse::DenseVector& x, 
                 SolverType type,
                 double tol = 1e-6, 
                 int max_iters = 1000,
                 cudaStream_t stream = nullptr) {
    if (type == SolverType::CG) {
        return cg_solve(A, b, x, tol, max_iters, stream);
    } else if (type == SolverType::PCG_JACOBI) {
        return pcg_solve(A, b, x, tol, max_iters, stream);
    }
    return -1;
}

} // namespace linsolvers
