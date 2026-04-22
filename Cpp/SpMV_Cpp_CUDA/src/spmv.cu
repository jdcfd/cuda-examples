#include "sparse/spmv.cuh"
#include "sparse/kernels.cuh"
#include <iostream>

namespace sparse {

namespace {

template <int BLOCK_SIZE>
void launch_spmv(const CSRMatrix& A, const DenseVector& x, DenseVector& y,
                 bool use_shared, dim3 blocks, dim3 threads, size_t shmem, cudaStream_t stream)
{
    if (use_shared) {
        sparse_mvm_shared<BLOCK_SIZE><<<blocks, threads, shmem, stream>>>(
            A.d_rows, A.d_cols, A.d_values, x.d_val, y.d_val,
            A.nrows, A.ncols);
    } else {
        sparse_mvm<BLOCK_SIZE><<<blocks, threads, 0, stream>>>(
            A.d_rows, A.d_cols, A.d_values, x.d_val, y.d_val,
            A.nrows, A.ncols);
    }
}

} // anonymous namespace

void multiply(const CSRMatrix& A, const DenseVector& x, DenseVector& y,
              int block_size, bool use_shared_memory, cudaStream_t stream)
{
    if (A.nrows == 0) return;

    int rows_per_block = 256 / block_size;
    int num_blocks = (A.nrows + rows_per_block - 1) / rows_per_block;

    dim3 blocks(num_blocks, 1, 1);
    dim3 threads(block_size, rows_per_block, 1);
    size_t shmem = rows_per_block * block_size * sizeof(double);

    switch (block_size) {
        case 128: launch_spmv<128>(A, x, y, use_shared_memory, blocks, threads, shmem, stream); break;
        case 64:  launch_spmv<64> (A, x, y, use_shared_memory, blocks, threads, shmem, stream); break;
        case 32:  launch_spmv<32> (A, x, y, use_shared_memory, blocks, threads, shmem, stream); break;
        case 16:  launch_spmv<16> (A, x, y, use_shared_memory, blocks, threads, shmem, stream); break;
        case 8:   launch_spmv<8>  (A, x, y, use_shared_memory, blocks, threads, shmem, stream); break;
        case 4:   launch_spmv<4>  (A, x, y, use_shared_memory, blocks, threads, shmem, stream); break;
        case 2:   launch_spmv<2>  (A, x, y, use_shared_memory, blocks, threads, 0, stream); break;
        default:  launch_spmv<1>  (A, x, y, use_shared_memory, blocks, threads, 0, stream); break;
    }
}

} // namespace sparse
