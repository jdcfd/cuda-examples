#pragma once
#include "csr_matrix.cuh"
#include "dense_vector.cuh"

namespace sparse {

/**
 * Sparse Matrix - Dense Vector multiplication (CSR format)
 *
 * @param A input matrix in CSR format (device memory)
 * @param x input vector (device memory)
 * @param y output vector (device memory, will be overwritten)
 * @param block_size number of threads per row (power of 2, typically 4-32)
 * @param use_shared_memory whether to use shared memory for reduction
 */
void multiply(const CSRMatrix& A,
              const DenseVector& x,
              DenseVector& y,
              int block_size = 32,
              bool use_shared_memory = false);

} // namespace sparse
