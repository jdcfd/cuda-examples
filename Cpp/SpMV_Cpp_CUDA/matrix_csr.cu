#include <matrix_csr.cuh>
#include <iostream>
#include <helper_cuda.h>

CSRMatrix::CSRMatrix(int nr, int nc, int nnz) {
    if( nnz > ((long)nr)*nc ){
        throw std::invalid_argument( "received nnz > nrows * ncols" );
    }
    this->nnz = nnz;
    this->nrows = nr;
    this->ncols = nc;

    this->alloc_mem();
}

void CSRMatrix::alloc_mem(){
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **> (&(this->h_rows)),sizeof(int)*(this->nrows+1)));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **> (&(this->h_cols)),sizeof(int)*(this->nnz)));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **> (&(this->h_values)),sizeof(double)*(this->nnz)));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **> (&(this->d_rows)),sizeof(int)*(this->nrows+1)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **> (&(this->d_cols)),sizeof(int)*(this->nnz)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **> (&(this->d_values)),sizeof(double)*(this->nnz)));
}

void CSRMatrix::free_mem(){
    checkCudaErrors(cudaFreeHost(this->h_rows));
    checkCudaErrors(cudaFreeHost(this->h_cols));
    checkCudaErrors(cudaFreeHost(this->h_values));
    this->h_rows = nullptr;
    this->h_cols = nullptr;
    this->h_values = nullptr;

    checkCudaErrors(cudaFree(this->d_rows));
    checkCudaErrors(cudaFree(this->d_cols));
    checkCudaErrors(cudaFree(this->d_values));
    this->d_rows = nullptr;
    this->d_cols = nullptr;
    this->d_values = nullptr;
}

CSRMatrix::~CSRMatrix(){
    nnz = 0;
    nrows = 0;
    ncols = 0;
    this->free_mem();
}

void CSRMatrix::update_host(){
    checkCudaErrors(cudaMemcpyAsync(this->h_values,this->d_values,sizeof(double)*this->nnz,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpyAsync(this->h_cols,this->d_cols,sizeof(int)*this->nnz,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpyAsync(this->h_rows,this->d_rows,sizeof(int)*(this->nrows+1),cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}

void CSRMatrix::update_device(){
    checkCudaErrors(cudaMemcpyAsync(this->d_values,this->h_values,sizeof(double)*this->nnz,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(this->d_cols,this->h_cols,sizeof(int)*this->nnz,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(this->d_rows,this->h_rows,sizeof(int)*(this->nrows+1),cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
}

void CSRMatrix::print(){

    // Call update host before callign this
    if(this->nrows > 0 && this->nnz > 0){
        std::cout << "Nrows: " << this->nrows << " Ncols: " << this->ncols << std::endl;
        std::cout << "Nnz: "   << this->nnz << std::endl;
        for(int i {}; i < this->nrows + 1; i++){
            std::cout << "rows[" << i << "] = " << this->h_rows[i] << std::endl;
        }
        for(int i {}; i < this->nnz; i++){
            std::cout << "cols[" << i << "]= " << this->h_cols[i] << ", val[" << i << "]= " << this->h_values[i] << std::endl;
        }
        std::cout << std::endl;
    }else{
        std::cout << "Matrix has not been initialized." << std::endl;
    }
}
