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

template <typename T>
cublasStatus_t cublas_dot(cublasHandle_t handle, int n, const T* x, const T* y, T* result);

template <>
cublasStatus_t cublas_dot<float>(cublasHandle_t handle, int n, const float* x, const float* y, float* result) {
    return cublasSdot(handle, n, x, 1, y, 1, result);
}

template <>
cublasStatus_t cublas_dot<double>(cublasHandle_t handle, int n, const double* x, const double* y, double* result) {
    return cublasDdot(handle, n, x, 1, y, 1, result);
}

template <typename T>
void run_precision_example(const char* precision_name, int n, int num_runs, T x_value, T y_value) {
    std::vector<T> h_x(n, x_value);
    std::vector<T> h_y(n, y_value);
    T h_result = static_cast<T>(0);

    T* d_x = nullptr;
    T* d_y = nullptr;
    T* d_result = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_result, &h_result, sizeof(T), cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // --- Custom Dot Product ---
    h_result = static_cast<T>(0);
    CHECK_CUDA(cudaMemcpyAsync(d_result, &h_result, sizeof(T), cudaMemcpyHostToDevice, stream));
    dot_product(d_x, d_y, d_result, n, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float custom_ms = 0.0f;
    float total_custom_ms = 0.0f;
    for (int i = 0; i < num_runs; i++) {
        h_result = static_cast<T>(0);
        CHECK_CUDA(cudaMemcpyAsync(d_result, &h_result, sizeof(T), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaEventRecord(start, stream));
        dot_product(d_x, d_y, d_result, n, stream);
        CHECK_CUDA(cudaMemcpyAsync(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&custom_ms, start, stop));
        total_custom_ms += custom_ms;
    }
    const float custom_mean_ms = total_custom_ms / num_runs;
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost));

    // --- Thrust Dot Product ---
    thrust::device_ptr<T> ptr_x(d_x);
    thrust::device_ptr<T> ptr_y(d_y);
    T thrust_result = thrust::inner_product(ptr_x, ptr_x + n, ptr_y, static_cast<T>(0));
    CHECK_CUDA(cudaDeviceSynchronize());

    float thrust_ms = 0.0f;
    float total_thrust_ms = 0.0f;
    for (int i = 0; i < num_runs; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        thrust_result = thrust::inner_product(ptr_x, ptr_x + n, ptr_y, static_cast<T>(0));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&thrust_ms, start, stop));
        total_thrust_ms += thrust_ms;
    }
    const float thrust_mean_ms = total_thrust_ms / num_runs;

    // --- cuBLAS Dot Product ---
    cublasHandle_t handle;
    cublasStatus_t cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: cublasCreate failed\n";
        exit(EXIT_FAILURE);
    }
    cublas_status = cublasSetStream(handle, stream);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: cublasSetStream failed\n";
        exit(EXIT_FAILURE);
    }
    T cublas_result = static_cast<T>(0);

    cublas_status = cublas_dot<T>(handle, n, d_x, d_y, &cublas_result);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: warmup dot failed\n";
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float cublas_ms = 0.0f;
    float total_cublas_ms = 0.0f;
    for (int i = 0; i < num_runs; i++) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        cublas_status = cublas_dot<T>(handle, n, d_x, d_y, &cublas_result);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS Error: benchmark dot failed\n";
            exit(EXIT_FAILURE);
        }
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&cublas_ms, start, stop));
        total_cublas_ms += cublas_ms;
    }
    const float cublas_mean_ms = total_cublas_ms / num_runs;

    const double expected = static_cast<double>(x_value) * static_cast<double>(y_value) * static_cast<double>(n);

    std::cout << "Precision: " << precision_name << std::endl;
    std::cout << "Array Size (N): " << n << std::endl;
    std::cout << "Runs per benchmark: " << num_runs << std::endl;
    std::cout << "Expected Dot Product: " << expected << std::endl;
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
    std::cout << std::endl;

    cublasDestroy(handle);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_result));
}

int main() {
    const int n = 10000000 + 1;
    const int num_runs = 10;

    run_precision_example<float>("single (float)", n, num_runs, 1.5f, 2.0f);
    run_precision_example<double>("double", n, num_runs, 1.5, 2.0);

    return 0;
}
