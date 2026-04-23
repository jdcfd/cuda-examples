#include "dot.cuh"
#include <cuda_runtime.h>

__global__ void dot_kernel_atomic(double* x, double* y, double* result, int N)
{
    extern __shared__ unsigned char dot_smem[];
    double* sdata = reinterpret_cast<double*>(dot_smem);

    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    double temp = 0.0;
    for (int i = global_id; i < N / 2; i += blockDim.x * gridDim.x) {
        double2 x2 = reinterpret_cast<double2*>(x)[i];
        double2 y2 = reinterpret_cast<double2*>(y)[i];
        temp += x2.x * y2.x + x2.y * y2.y;
    }
    if (global_id == N / 2 && N % 2 == 1) {
        temp += x[N - 1] * y[N - 1];
    }
    sdata[tid] = temp;
    __syncthreads();

    unsigned int s = blockDim.x / 2;
    for (; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    for (; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
    }

    if (!tid) {
        atomicAdd(result, sdata[0]);
    }
}

__global__ void dot_kernel_atomic_float(float* x, float* y, float* result, int N)
{
    extern __shared__ unsigned char dot_smem[];
    float* sdata = reinterpret_cast<float*>(dot_smem);

    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0.f;
    const int vec4_count = N / 4;
    for (int i = global_id; i < vec4_count; i += blockDim.x * gridDim.x) {
        float4 x4 = reinterpret_cast<float4*>(x)[i];
        float4 y4 = reinterpret_cast<float4*>(y)[i];
        temp += x4.x * y4.x + x4.y * y4.y + x4.z * y4.z + x4.w * y4.w;
    }
    const int tail_start = vec4_count * 4;
    for (int j = global_id + tail_start; j < N; j += blockDim.x * gridDim.x) {
        temp += x[j] * y[j];
    }

    sdata[tid] = temp;
    __syncthreads();

    unsigned int s = blockDim.x / 2;
    for (; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    for (; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
    }

    if (!tid) {
        atomicAdd(result, sdata[0]);
    }
}

void dot_product(double* x, double* y, double* result, int N, cudaStream_t stream)
{
    int blockSize = 64;
    size_t sharedMemSize = blockSize * sizeof(double);

    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, dot_kernel_atomic, blockSize, sharedMemSize);

    int waveSize = prop.multiProcessorCount * maxBlocksPerSM;
    int maxBlocks = waveSize * 2;
    int numBlocks = (N / 2 + blockSize - 1) / blockSize;
    if (numBlocks > maxBlocks) {
        numBlocks = maxBlocks;
    }
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    dot_kernel_atomic<<<numBlocks, blockSize, sharedMemSize, stream>>>(x, y, result, N);
}

void dot_product(float* x, float* y, float* result, int N, cudaStream_t stream)
{
    int blockSize = 64;
    size_t sharedMemSize = blockSize * sizeof(float);

    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM, dot_kernel_atomic_float, blockSize, sharedMemSize);

    int waveSize = prop.multiProcessorCount * maxBlocksPerSM;
    int maxBlocks = waveSize * 2;
    int numBlocks = (N / 4 + blockSize - 1) / blockSize;
    if (numBlocks > maxBlocks) {
        numBlocks = maxBlocks;
    }
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    dot_kernel_atomic_float<<<numBlocks, blockSize, sharedMemSize, stream>>>(x, y, result, N);
}
