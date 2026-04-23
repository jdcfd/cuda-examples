#include "sparse/dense_vector.cuh"
#include <helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <iostream>

namespace sparse {

template <typename T>
DenseVectorT<T>::DenseVectorT(int n) : size(n) {
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_val), sizeof(T) * n));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_val), sizeof(T) * n));
    fill(static_cast<T>(0));
}

template <typename T>
DenseVectorT<T>::~DenseVectorT() {
    if (h_val) checkCudaErrors(cudaFreeHost(h_val));
    if (d_val) checkCudaErrors(cudaFree(d_val));
    h_val = nullptr;
    d_val = nullptr;
    size = 0;
}

template <typename T>
void DenseVectorT<T>::generate() {
    thrust::default_random_engine rng(1337);
    thrust::uniform_real_distribution<T> dist;
    thrust::generate(h_val, h_val + size, [&]() { return dist(rng); });
    update_device();
}

template <typename T>
void DenseVectorT<T>::print() const {
    for (int i = 0; i < size; i++) {
        std::cout << "v[" << i << "] = " << h_val[i] << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void DenseVectorT<T>::fill(T v) {
    thrust::device_ptr<T> first = thrust::device_pointer_cast(d_val);
    thrust::device_ptr<T> last  = thrust::device_pointer_cast(d_val + size);
    thrust::fill(first, last, v);
    update_host();
}

template <typename T>
void DenseVectorT<T>::update_host() {
    checkCudaErrors(cudaMemcpy(h_val, d_val, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template <typename T>
void DenseVectorT<T>::update_device() {
    checkCudaErrors(cudaMemcpy(d_val, h_val, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template class DenseVectorT<double>;
template class DenseVectorT<float>;

} // namespace sparse
