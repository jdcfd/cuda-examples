#ifndef SPARSE_KERNELS_CUH
#define SPARSE_KERNELS_CUH

namespace sparse {

template <int block_size, typename T>
__global__ void sparse_mvm(int* rows, int* cols, T* vals, T* vec, T* res, int nrows, int ncols);

template <int block_size, typename T>
__device__ void warpReduce(volatile T* sdata, unsigned int tid);

template <int block_size, typename T>
__global__ void sparse_mvm_shared(int* rows, int* cols, T* vals, T* vec, T* res, int nrows, int ncols);

} // namespace sparse

#endif // SPARSE_KERNELS_CUH
