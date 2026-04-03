#include "sparse/kernels.cuh"
#include <cuda_runtime.h>

namespace sparse {

template <int block_size>
__global__ void sparse_mvm(int * rows, int * cols, double * vals, double * vec, double * res, int nrows, int ncols)
{
    // Block index
    int row = threadIdx.y + blockDim.y*blockIdx.x;
    if(row < nrows){
        int start {rows[row]};
        int end {rows[row+1]}; 
        double sum = 0.0;

        for(int icol = threadIdx.x + start; icol < end; icol += block_size ){
            sum += vals[icol] * vec[cols[icol]];
        }

        // Need to use templated block size to unroll loop
#pragma unroll
        for (int i = block_size >> 1; i > 0; i >>= 1)
            sum += __shfl_down_sync(0xffffffff,sum, i, i*2);

        if(!threadIdx.x){ res[row] = sum; } // write only with first thread        
    }
}


template <int block_size>
__device__ void warpReduce(volatile double *sdata, unsigned int tid) {
    if (block_size >= 64) sdata[tid] += sdata[tid + 32];
    if (block_size >= 32) sdata[tid] += sdata[tid + 16];
    if (block_size >= 16) sdata[tid] += sdata[tid + 8];
    if (block_size >= 8) sdata[tid] += sdata[tid + 4];
    if (block_size >= 4) sdata[tid] += sdata[tid + 2];
    if (block_size >= 2) sdata[tid] += sdata[tid + 1];
}

template <int block_size>
__global__ void sparse_mvm_shared(int * rows, int * cols, double * vals, double * vec, double * res, int nrows, int ncols)
{
     // Block index
    int row = threadIdx.y + blockDim.y*blockIdx.x;
    extern __shared__ double sum[];
    unsigned int tid = threadIdx.x;

    if(row < nrows){
        int start {rows[row]};
        int end {rows[row+1]}; 
        int icol = tid + start;

        double vcval = vec[cols[icol]];

        sum[threadIdx.y*(block_size) + tid] = (icol < end) ? vals[icol] * vcval : 0.0;

        for( icol = icol + block_size; icol < end; icol+= block_size ){
            vcval = vec[cols[icol]];  
            sum[threadIdx.y*(block_size) + tid] += vals[icol] * vcval;
        }
        __syncthreads();

        if (block_size >= 512) { if (tid < 256) { 
            sum[threadIdx.y*(block_size) + tid] += sum[threadIdx.y*(block_size) + tid + 256]; } 
            __syncthreads(); }
        if (block_size >= 256) { if (tid < 128) { 
            sum[threadIdx.y*(block_size) + tid] += sum[threadIdx.y*(block_size) + tid + 128]; } 
            __syncthreads(); }
        if (block_size >= 128) { if (tid < 64) { 
            sum[threadIdx.y*(block_size) + tid] += sum[threadIdx.y*(block_size) + tid + 64]; } 
            __syncthreads(); }

        if (tid < 32) warpReduce<block_size>(&(sum[threadIdx.y*(block_size)]), tid);

        if(!tid){ res[row] = sum[threadIdx.y*(block_size) + tid]; } // write only with first thread        
    }
}

// Explicit instantiations (outside namespace for CUDA visibility)
template __global__ void sparse::sparse_mvm<128>(int*, int*, double*, double*, double*, int, int);
template __global__ void sparse::sparse_mvm<64>(int*, int*, double*, double*, double*, int, int);
template __global__ void sparse::sparse_mvm<32>(int*, int*, double*, double*, double*, int, int);
template __global__ void sparse::sparse_mvm<16>(int*, int*, double*, double*, double*, int, int);
template __global__ void sparse::sparse_mvm<8>(int*, int*, double*, double*, double*, int, int);
template __global__ void sparse::sparse_mvm<4>(int*, int*, double*, double*, double*, int, int);
template __global__ void sparse::sparse_mvm<2>(int*, int*, double*, double*, double*, int, int);
template __global__ void sparse::sparse_mvm<1>(int*, int*, double*, double*, double*, int, int);

template __device__ void sparse::warpReduce<128>(volatile double*, unsigned int);
template __device__ void sparse::warpReduce<64>(volatile double*, unsigned int);
template __device__ void sparse::warpReduce<32>(volatile double*, unsigned int);
template __device__ void sparse::warpReduce<16>(volatile double*, unsigned int);
template __device__ void sparse::warpReduce<8>(volatile double*, unsigned int);
template __device__ void sparse::warpReduce<4>(volatile double*, unsigned int);
template __device__ void sparse::warpReduce<2>(volatile double*, unsigned int);
template __device__ void sparse::warpReduce<1>(volatile double*, unsigned int);

template __global__ void sparse::sparse_mvm_shared<128>(int*, int*, double*, double*, double*, int, int);
template __global__ void sparse::sparse_mvm_shared<64>(int*, int*, double*, double*, double*, int, int);
template __global__ void sparse::sparse_mvm_shared<32>(int*, int*, double*, double*, double*, int, int);
template __global__ void sparse::sparse_mvm_shared<16>(int*, int*, double*, double*, double*, int, int);
template __global__ void sparse::sparse_mvm_shared<8>(int*, int*, double*, double*, double*, int, int);
template __global__ void sparse::sparse_mvm_shared<4>(int*, int*, double*, double*, double*, int, int);
template __global__ void sparse::sparse_mvm_shared<2>(int*, int*, double*, double*, double*, int, int);
template __global__ void sparse::sparse_mvm_shared<1>(int*, int*, double*, double*, double*, int, int);

} // namespace sparse
