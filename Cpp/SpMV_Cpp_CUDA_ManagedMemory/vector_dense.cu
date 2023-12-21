#include <helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <vector_dense.cuh>

DenseVector::DenseVector(int n){
    this->size = n;
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **> (&(this->val)),sizeof(double)*(n)));
    this->fill(0.0);
}

DenseVector::~DenseVector(){
    checkCudaErrors(cudaFree(this->val));
    this->val = nullptr;
}

void DenseVector::generate(){
    thrust::default_random_engine rng(1337);
    thrust::uniform_real_distribution<double> dist;
    // This is done on host, Managed memory should know to update memory on device before using it
    thrust::generate( this->val, this->val + this->size, [&] { return dist(rng); });
}

void DenseVector::print(){
    for(int i {}; i < this->size; i++){
        std::cout << "v[" << i << "] = " << this->val[i] << std::endl;
    }
    std::cout << std::endl;
}

void DenseVector::fill(double v){
    thrust::device_ptr<double> first = thrust::device_pointer_cast(this->val);
    thrust::device_ptr<double> last = thrust::device_pointer_cast(this->val + this->size);
    thrust::fill(first, last, v);
}
