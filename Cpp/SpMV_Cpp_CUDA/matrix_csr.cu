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
}

CSRMatrix::~CSRMatrix(){
    delete [] this->rows; this->rows = nullptr;
    delete [] this->cols; this->cols = nullptr;
    delete [] this->values; this->values = nullptr;
}