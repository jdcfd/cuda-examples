#include <helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <vector_dense.cuh>

DenseVector::DenseVector(int n){
    this->size = n;
    this->h_v = new double [n] {};
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **> (&(this->d_v)),sizeof(double)*(n)));
    thrust::device_ptr<double> first = thrust::device_pointer_cast(this->d_v);
    thrust::device_ptr<double> last = thrust::device_pointer_cast(this->d_v + n);
    thrust::fill(first, last, 0.0);
}

DenseVector::~DenseVector(){
    delete [] this->h_v; this->h_v = nullptr;
    checkCudaErrors(cudaFree(this->d_v));
    this->d_v = nullptr;
}

void DenseVector::generate(){
    thrust::default_random_engine rng(1337);
    thrust::uniform_real_distribution<double> dist;
    thrust::generate( this->h_v, this->h_v + this->size, [&] { return dist(rng); });
    thrust::device_ptr<double> first = thrust::device_pointer_cast(this->d_v);
    thrust::copy(this->h_v, this->h_v + this->size, first);
}

void DenseVector::update_device(){
    size_t size_vec {sizeof(int)*(this->size)};
    checkCudaErrors(cudaMemcpy(this->d_v , this->h_v  , size_vec, cudaMemcpyHostToDevice));
}

void DenseVector::update_host(){
    size_t size_vec {sizeof(int)*(this->size)};
    checkCudaErrors(cudaMemcpy(this->h_v , this->d_v  , size_vec, cudaMemcpyDeviceToHost));
}