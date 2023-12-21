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
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **> (&(this->rows)),sizeof(int)*(this->nrows+1)));
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **> (&(this->cols)),sizeof(int)*(this->nnz+1)));
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **> (&(this->values)),sizeof(double)*(this->nnz+1)));
}

void CSRMatrix::free_mem(){
    checkCudaErrors(cudaFree(this->rows));
    checkCudaErrors(cudaFree(this->cols));
    checkCudaErrors(cudaFree(this->values));
    this->rows = nullptr;
    this->cols = nullptr;
    this->values = nullptr;
}

CSRMatrix::~CSRMatrix(){
    this->free_mem();
}

void CSRMatrix::print(){

    if(this->nrows > 0 && this->nnz > 0){
        std::cout << "Nrows: " << this->nrows << " Ncols: " << this->ncols << std::endl;
        std::cout << "Nnz: "   << this->nnz << std::endl;
        for(int i {}; i < this->nrows + 1; i++){
            std::cout << "rows[" << i << "] = " << this->rows[i] << std::endl;
        }
        for(int i {}; i < this->nnz; i++){
            std::cout << "cols[" << i << "]= " << this->cols[i] << ", val[" << i << "]= " << this->values[i] << std::endl;
        }
        std::cout << std::endl;
    }else{
        std::cout << "Matrix has not been set." << std::endl;
    }
}