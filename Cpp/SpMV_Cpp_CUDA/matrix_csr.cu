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
    this->h_rows = new int[this->nrows+1];
    this->h_cols = new int[this->nnz];
    this->h_values = new double[this->nnz];

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **> (&(this->d_rows)),sizeof(int)*(this->nrows+1)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **> (&(this->d_cols)),sizeof(int)*(this->nnz)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **> (&(this->d_values)),sizeof(double)*(this->nnz)));
}

void CSRMatrix::free_mem(){
    delete this->h_rows;
    delete this->h_cols;
    delete this->h_values;

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
    checkCudaErrors(cudaMemcpy(this->h_values,this->d_values,sizeof(double)*this->nnz,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(this->h_rows,this->d_rows,sizeof(int)*(this->nrows+1),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(this->h_cols,this->d_cols,sizeof(int)*this->nnz,cudaMemcpyDeviceToHost));
}

void CSRMatrix::update_device(){
    checkCudaErrors(cudaMemcpy(this->d_values,this->h_values,sizeof(double)*this->nnz,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->d_rows,this->h_rows,sizeof(int)*(this->nrows+1),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->d_cols,this->h_cols,sizeof(int)*this->nnz,cudaMemcpyHostToDevice));
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
