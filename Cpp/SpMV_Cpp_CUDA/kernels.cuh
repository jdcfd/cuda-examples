#ifndef KERNELS_CUH
#define KERNELS_CUH

template <int block_size>
__global__ void sparse_mvm(int * rows, int * cols, double * vals, double * vec, double * res, int nrows, int ncols);

template <int block_size>
__device__ void warpReduce(volatile double *sdata, unsigned int tid);

template <int block_size>
__global__ void sparse_mvm_shared(int * rows, int * cols, double * vals, double * vec, double * res, int nrows, int ncols);

#endif // KERNELS_CUH
