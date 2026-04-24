#include "sparse/kernels.cuh"
#include <cuda_runtime.h>

namespace sparse {

template <int block_size, typename T>
__global__ void sparse_mvm(int* rows, int* cols, T* vals, T* vec, T* res, int nrows, int ncols)
{
    int row = threadIdx.y + blockDim.y * blockIdx.x;
    if (row < nrows) {
        int start{rows[row]};
        int end{rows[row + 1]};
        T sum = T(0);

        for (int icol = threadIdx.x + start; icol < end; icol += block_size) {
            sum += vals[icol] * vec[cols[icol]];
        }

#pragma unroll
        for (int i = block_size >> 1; i > 0; i >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, i, i * 2);

        if (!threadIdx.x) {
            res[row] = sum;
        }
    }
}

template <int block_size, typename T>
__device__ void warpReduce(volatile T* sdata, unsigned int tid)
{
    if constexpr (block_size >= 64) sdata[tid] += sdata[tid + 32];
    if constexpr (block_size >= 32) sdata[tid] += sdata[tid + 16];
    if constexpr (block_size >= 16) sdata[tid] += sdata[tid + 8];
    if constexpr (block_size >= 8) sdata[tid] += sdata[tid + 4];
    if constexpr (block_size >= 4) sdata[tid] += sdata[tid + 2];
    if constexpr (block_size >= 2) sdata[tid] += sdata[tid + 1];
}

template <int block_size, typename T>
__global__ void sparse_mvm_shared(int* rows, int* cols, T* vals, T* vec, T* res, int nrows, int ncols)
{
    int row = threadIdx.y + blockDim.y * blockIdx.x;
    extern __shared__ unsigned char shared_raw[];
    T* sum = reinterpret_cast<T*>(shared_raw);
    unsigned int tid = threadIdx.x;

    if (row < nrows) {
        int start{rows[row]};
        int end{rows[row + 1]};
        int icol = tid + start;

        T vcval = vec[cols[icol]];

        sum[threadIdx.y * (block_size) + tid] = (icol < end) ? vals[icol] * vcval : T(0);

        for (icol = icol + block_size; icol < end; icol += block_size) {
            vcval = vec[cols[icol]];
            sum[threadIdx.y * (block_size) + tid] += vals[icol] * vcval;
        }
        __syncthreads();

        if (block_size >= 512) {
            if (tid < 256) {
                sum[threadIdx.y * (block_size) + tid] += sum[threadIdx.y * (block_size) + tid + 256];
            }
            __syncthreads();
        }
        if (block_size >= 256) {
            if (tid < 128) {
                sum[threadIdx.y * (block_size) + tid] += sum[threadIdx.y * (block_size) + tid + 128];
            }
            __syncthreads();
        }
        if (block_size >= 128) {
            if (tid < 64) {
                sum[threadIdx.y * (block_size) + tid] += sum[threadIdx.y * (block_size) + tid + 64];
            }
            __syncthreads();
        }

        if (tid < 32) warpReduce<block_size, T>(&(sum[threadIdx.y * (block_size)]), tid);

        if (!tid) {
            res[row] = sum[threadIdx.y * (block_size) + tid];
        }
    }
}

#define SPARSE_INST_MVM(BS, Ty) \
    template __global__ void sparse::sparse_mvm<BS, Ty>(int*, int*, Ty*, Ty*, Ty*, int, int);

#define SPARSE_INST_WARP(BS, Ty) \
    template __device__ void sparse::warpReduce<BS, Ty>(volatile Ty*, unsigned int);

#define SPARSE_INST_SHARED(BS, Ty) \
    template __global__ void sparse::sparse_mvm_shared<BS, Ty>(int*, int*, Ty*, Ty*, Ty*, int, int);

#define SPARSE_INST_ALL_BS(Ty) \
    SPARSE_INST_MVM(128, Ty) SPARSE_INST_MVM(64, Ty) SPARSE_INST_MVM(32, Ty) SPARSE_INST_MVM(16, Ty) \
        SPARSE_INST_MVM(8, Ty) SPARSE_INST_MVM(4, Ty) SPARSE_INST_MVM(2, Ty) SPARSE_INST_MVM(1, Ty) \
            SPARSE_INST_WARP(128, Ty) SPARSE_INST_WARP(64, Ty) SPARSE_INST_WARP(32, Ty) SPARSE_INST_WARP(16, Ty) \
                SPARSE_INST_WARP(8, Ty) SPARSE_INST_WARP(4, Ty) SPARSE_INST_WARP(2, Ty) SPARSE_INST_WARP(1, Ty) \
                    SPARSE_INST_SHARED(128, Ty) SPARSE_INST_SHARED(64, Ty) SPARSE_INST_SHARED(32, Ty) \
                        SPARSE_INST_SHARED(16, Ty) SPARSE_INST_SHARED(8, Ty) SPARSE_INST_SHARED(4, Ty) \
                            SPARSE_INST_SHARED(2, Ty) SPARSE_INST_SHARED(1, Ty)

SPARSE_INST_ALL_BS(double)
SPARSE_INST_ALL_BS(float)

#undef SPARSE_INST_ALL_BS
#undef SPARSE_INST_SHARED
#undef SPARSE_INST_WARP
#undef SPARSE_INST_MVM

} // namespace sparse
