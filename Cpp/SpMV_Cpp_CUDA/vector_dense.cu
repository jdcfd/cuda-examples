#include <helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <vector_dense.cuh>

DenseVector::DenseVector(int n){
    this->size = n;
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **> (&(this->h_val)),sizeof(double)*(n)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **> (&(this->d_val)),sizeof(double)*(n)));
    this->fill(0.0);
}

DenseVector::~DenseVector(){
    checkCudaErrors(cudaFreeHost(this->h_val));
    checkCudaErrors(cudaFree(this->d_val));
    this->h_val = nullptr;
    this->d_val = nullptr;
    this->size = 0;
}

void DenseVector::generate(){
    thrust::default_random_engine rng(1337);
    thrust::uniform_real_distribution<double> dist;
    // This is done on host.
    thrust::generate( this->h_val, this->h_val + this->size, [&] { return dist(rng); });
    this->update_device();
}

void DenseVector::print(){
    // Update before calling print
    // this->update_host; 
    for(int i {}; i < this->size; i++){
        std::cout << "v[" << i << "] = " << this->h_val[i] << std::endl;
    }
    std::cout << std::endl;
}

void DenseVector::fill(double v){
    thrust::device_ptr<double> first = thrust::device_pointer_cast(this->d_val);
    thrust::device_ptr<double> last = thrust::device_pointer_cast(this->d_val + this->size);
    thrust::fill(first, last, v);
    this->update_host();
}

void DenseVector::update_host(){
    checkCudaErrors(cudaMemcpy(this->h_val,this->d_val,sizeof(double)*this->size,cudaMemcpyDeviceToHost));
}
void DenseVector::update_device(){
    checkCudaErrors(cudaMemcpy(this->d_val,this->h_val,sizeof(double)*this->size,cudaMemcpyHostToDevice));
}
