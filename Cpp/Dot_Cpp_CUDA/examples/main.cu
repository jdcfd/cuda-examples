#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
#include <cublas_v2.h>
#include "dot.cuh"

// Helper macro for error checking
#define CHECK_CUDA(call)                                                 \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)       \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }

int main() {
    int N = 10000000 + 1; // 1 << 24; // divisible by 4
    int num_runs = 10;
    
    // Allocate Host Memory
    std::vector<double> h_x(N, 1.5);
    std::vector<double> h_y(N, 2.0);
    double h_result = 0.0;

    // Allocate Device Memory
    double *d_x, *d_y, *d_result;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(double)));

    // Copy data from Host to Device
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice));

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Custom Dot Product ---
    // Warmup
    h_result = 0.0;
    CHECK_CUDA(cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice));
    dot_product(d_x, d_y, d_result, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    float custom_ms, total_custom_ms = 0.0f;
    for (int i = 0; i < num_runs; i++) {
        h_result = 0.0;
        CHECK_CUDA(cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice));
        cudaEventRecord(start);
        dot_product(d_x, d_y, d_result, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&custom_ms, start, stop);
        total_custom_ms += custom_ms;
    }
    float custom_mean_ms = total_custom_ms / num_runs;
    
    // Get custom result
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    // --- Thrust Dot Product ---
    thrust::device_ptr<double> ptr_x(d_x);
    thrust::device_ptr<double> ptr_y(d_y);

    // Warmup
    double thrust_result = thrust::inner_product(ptr_x, ptr_x + N, ptr_y, 0.0);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    float thrust_ms, total_thrust_ms = 0.0f;
    for (int i = 0; i < num_runs; i++) {
        cudaEventRecord(start);
        thrust_result = thrust::inner_product(ptr_x, ptr_x + N, ptr_y, 0.0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&thrust_ms, start, stop);
        total_thrust_ms += thrust_ms;
    }
    float thrust_mean_ms = total_thrust_ms / num_runs;

    // --- cuBLAS Dot Product ---
    cublasHandle_t handle;
    cublasCreate(&handle);
    double cublas_result = 0.0;

    // Warmup
    cublasDdot(handle, N, d_x, 1, d_y, 1, &cublas_result);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    float cublas_ms, total_cublas_ms = 0.0f;
    for (int i = 0; i < num_runs; i++) {
        cudaEventRecord(start);
        cublasDdot(handle, N, d_x, 1, d_y, 1, &cublas_result);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cublas_ms, start, stop);
        total_cublas_ms += cublas_ms;
    }
    float cublas_mean_ms = total_cublas_ms / num_runs;

    // --- Output Results ---
    std::cout << "Array Size (N): " << N << std::endl;
    std::cout << "Runs per benchmark: " << num_runs << std::endl;
    std::cout << "Expected Dot Product: " << (1.5 * 2.0 * N) << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Custom Dot Product Result: " << h_result << std::endl;
    std::cout << "Custom Mean Time: " << custom_mean_ms << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Thrust Dot Product Result: " << thrust_result << std::endl;
    std::cout << "Thrust Mean Time: " << thrust_mean_ms << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "cuBLAS Dot Product Result: " << cublas_result << std::endl;
    std::cout << "cuBLAS Mean Time: " << cublas_mean_ms << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    // Free resources
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_result));

    return 0;
}
