#include <matrix_csr.cuh>
#include <iostream>

CSRMatrix::CSRMatrix(int nr, int nc, int nnz) : MatrixBase(nr,nc){
    if( nnz > nr*nc ){
        throw std::invalid_argument( "received nnz > nrows * ncols" );
    }
    this->nnz = nnz;
    this->rows = new int[nr+1]{};
    this->cols = new int[nnz]{};
    this->values = new double[nnz]{};

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **> (&(this->d_rows)),sizeof(int)*(nr+1)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **> (&(this->d_cols)),sizeof(int)*(nnz+1)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **> (&(this->d_values)),sizeof(double)*(nnz+1)));
}

CSRMatrix::~CSRMatrix(){
    delete [] this->rows; this->rows = nullptr;
    delete [] this->cols; this->cols = nullptr;
    delete [] this->values; this->values = nullptr;
    checkCudaErrors(cudaFree(this->d_rows));
    checkCudaErrors(cudaFree(this->d_cols));
    checkCudaErrors(cudaFree(this->d_values));
    this->d_rows = nullptr;
    this->d_cols = nullptr;
    this->d_values = nullptr;
}

void CSRMatrix::update_device(){
    size_t size_rows {sizeof(int)*(this->nrows+1)};
    size_t size_cols {sizeof(int)*(this->nnz)};
    size_t size_vals {sizeof(double)*(this->nnz)};

    checkCudaErrors(cudaMemcpy(this->d_rows  , this->rows  , size_rows, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->d_cols  , this->cols  , size_cols, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->d_values, this->values, size_vals, cudaMemcpyHostToDevice));
}

void CSRMatrix::update_host(){
    size_t size_rows {sizeof(int)*(this->nrows+1)};
    size_t size_cols {sizeof(int)*(this->nnz)};
    size_t size_vals {sizeof(double)*(this->nnz)};

    checkCudaErrors(cudaMemcpy(this->rows  , this->d_rows  , size_rows, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(this->cols  , this->d_cols  , size_cols, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(this->values, this->d_values, size_vals, cudaMemcpyDeviceToHost));
}