#include "dot.cuh"
#include <cuda_runtime.h>


__global__ void dot_kernel_atomic(double* x, double* y, double* result, int N) {
    // Shared memory for block reduction
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    double temp = 0.0;
    // Iterate if the grid is smaller than the array size (grid-stride loop)
    for (int i = global_id; i < N/2; i += blockDim.x * gridDim.x) {
        double2 x2 = reinterpret_cast<double2*>(x)[i];
        double2 y2 = reinterpret_cast<double2*>(y)[i];
        temp += x2.x * y2.x + x2.y * y2.y;
    }
    if (global_id == N/2 && N%2==1) {
        temp += x[N-1] * y[N-1];
    }
    sdata[tid] = temp;
    __syncthreads();

    unsigned int s = blockDim.x / 2;
    for (s; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    for (s; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
    }

    // Thread 0 adds the block sum to the global result using atomicAdd
    if (!tid) {
        atomicAdd(result, sdata[0]);
    }
}

void dot_product(double* x, double* y, double* result, int N, cudaStream_t stream) {
    int blockSize = 64;
    // Allocate shared memory size: blockSize * sizeof(double)
    size_t sharedMemSize = blockSize * sizeof(double);

    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    int maxBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, dot_kernel_atomic, blockSize, sharedMemSize);
    
    // Wave size is the total number of blocks that can execute concurrently
    int waveSize = prop.multiProcessorCount * maxBlocksPerSM;
    
    // Set maxBlocks to a multiple of waveSize (e.g., 2 full waves)
    int maxBlocks = waveSize * 2;
    // Calculate the number of blocks needed, up to a maximum
    int numBlocks = (N / 2 + blockSize - 1) / blockSize;
    if (numBlocks > maxBlocks) {
        numBlocks = maxBlocks;
    }
    dot_kernel_atomic<<<numBlocks, blockSize, sharedMemSize, stream>>>(x, y, result, N);
    // cudaDeviceSynchronize();
}
