#pragma once

#include <cuda_runtime.h>

// Dot product of two arrays on the device. Caller must zero *result before launch.
void dot_product(double* x, double* y, double* result, int N, cudaStream_t stream = nullptr);

void dot_product(float* x, float* y, float* result, int N, cudaStream_t stream = nullptr);
