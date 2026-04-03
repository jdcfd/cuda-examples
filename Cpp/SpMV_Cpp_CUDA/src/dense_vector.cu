#include "sparse/dense_vector.cuh"
#include <helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <iostream>

namespace sparse {

DenseVector::DenseVector(int n) : size(n) {
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_val), sizeof(double) * n));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_val), sizeof(double) * n));
    fill(0.0);
}

DenseVector::~DenseVector() {
    if (h_val) checkCudaErrors(cudaFreeHost(h_val));
    if (d_val) checkCudaErrors(cudaFree(d_val));
    h_val = nullptr;
    d_val = nullptr;
    size = 0;
}

void DenseVector::generate() {
    thrust::default_random_engine rng(1337);
    thrust::uniform_real_distribution<double> dist;
    // generate on host
    thrust::generate(h_val, h_val + size, [&]() { return dist(rng); });
    update_device();
}

void DenseVector::print() const {
    for (int i = 0; i < size; i++) {
        std::cout << "v[" << i << "] = " << h_val[i] << std::endl;
    }
    std::cout << std::endl;
}

void DenseVector::fill(double v) {
    thrust::device_ptr<double> first = thrust::device_pointer_cast(d_val);
    thrust::device_ptr<double> last  = thrust::device_pointer_cast(d_val + size);
    thrust::fill(first, last, v);
    update_host();
}

void DenseVector::update_host() {
    checkCudaErrors(cudaMemcpy(h_val, d_val, sizeof(double)*size, cudaMemcpyDeviceToHost));
}

void DenseVector::update_device() {
    checkCudaErrors(cudaMemcpy(d_val, h_val, sizeof(double)*size, cudaMemcpyHostToDevice));
}

} // namespace sparse
