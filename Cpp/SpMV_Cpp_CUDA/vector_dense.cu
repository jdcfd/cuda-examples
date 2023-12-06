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
    size_t size_vec {sizeof(double)*(this->size)};
    checkCudaErrors(cudaMemcpy(this->d_v , this->h_v  , size_vec, cudaMemcpyHostToDevice));
}

void DenseVector::update_host(){
    size_t size_vec {sizeof(double)*(this->size)};
    checkCudaErrors(cudaMemcpy(this->h_v , this->d_v  , size_vec, cudaMemcpyDeviceToHost));
}

void DenseVector::print(){
    this->update_host();
    for(int i {}; i < this->size; i++){
        std::cout << "v[" << i << "] = " << this->h_v[i] << std::endl;
    }
    std::cout << std::endl;
}

void DenseVector::fill(double v){
    thrust::device_ptr<double> first = thrust::device_pointer_cast(this->d_v);
    thrust::device_ptr<double> last = thrust::device_pointer_cast(this->d_v + this->size);
    thrust::fill(first, last, v);
}

/*
void DenseVector::operator=(const DenseVector& dv){
    if(this->size != dv.size){        
        throw std::invalid_argument("DenseVector has different size.");
    }
    cudaPointerAttributes pin, pa;
    // Check if pointer is initialized or not
    checkCudaErrors(cudaPointerGetAttributes(&pa, this->d_v));
    checkCudaErrors(cudaPointerGetAttributes(&pin, dv.d_v));
    if(pa.devicePointer && pin.devicePointer) // if not null
    {
        thrust::device_ptr<double> dst = thrust::device_pointer_cast(this->d_v);
        thrust::device_ptr<double> src = thrust::device_pointer_cast(dv.d_v);
        thrust::copy(src, src + dv.size, dst);
    }
    else{
        throw std::invalid_argument("DenseVector not initialized on the device.");
    }
}
*/