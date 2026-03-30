/*
Author: Juan D. Colmenares F.
User  : jdcfd@github.com

Sparse Matrix-Vector multiplication in CUDA

Reads in Sparse matrix in MatrixMarket COO format and multiplies
it by a dense vector with random values.

*/
#include <helper_cuda.h>
#include "matrix.hpp"
#include "matrix_csr.cuh"
#include "mmio_reader.cuh"
#include "vector_dense.cuh"
#include "kernels.cuh"
#include <cusparse.h> 

#include <chrono>

#define TIME_KERNEL(func)                                                   \
{                                                                           \
    const int NRUNS = 10;                                                   \
    double total = 0.0;                                                     \
    for(int irun = 0; irun < NRUNS; irun++) {                              \
        auto t0 = std::chrono::high_resolution_clock::now();                \
        (func);                                                             \
        cudaDeviceSynchronize();                                            \
        auto t1 = std::chrono::high_resolution_clock::now();                \
        total += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1.e-6; \
    }                                                                       \
    std::cout << "-- Kernel duration (avg of " << NRUNS << " runs): " << (total / NRUNS) << " ms" << std::endl; \
}

#define EPS 1e-08

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

using namespace std;

void compare_values(DenseVector *Y, DenseVector *Yref)
{
    bool issame {true};

    for( int i {}; i < Y->size; i++ ){
        double error = fabs(Y->h_val[i] - Yref->h_val[i]);
        issame = issame && ((error < fabs(EPS*Yref->h_val[i])) ? true : (printf("|%e[%d] - %e[%d]| = %e\n",Y->h_val[i],i,Yref->h_val[i],i,error), false));
    }

    if(issame){
        cout << "-- Results are correct!" << endl;
    } else {
        cout << "-- Results are Wrong!" << endl;

        /*********
        for(int i = 0; i < Y->size ; i++){
            if( fabs(Y->h_val[i] - Yref->h_val[i]) >= EPS )
                cout << i << ", Y: " << Y->h_val[i] << ",  Ycsp: " << Yref->h_val[i] << endl;
        }
        *********/
    }
}

void run_test(int block_size, std::unique_ptr<CSRMatrix>& mymat, DenseVector *X, DenseVector *Y, int mnnzpr, bool shared = false){
    // limit the number of threads per row to be no larger than the warp size

    int rows_per_block = 256 / block_size;
    int num_blocks = (mymat->nrows + rows_per_block - 1) / rows_per_block;
    
    dim3 blocks(num_blocks, 1, 1);
    dim3 threads(block_size, rows_per_block, 1);
    size_t shms = rows_per_block*(block_size)*sizeof(double);

    cout << "Running test with block_size=" << block_size << " and shared=" << (shared ? "true" : "false") << endl;

    switch (block_size)
    {
    case 128:
        if(shared){
            TIME_KERNEL((sparse_mvm_shared<128><<<blocks,threads,shms>>>(mymat->d_rows, mymat->d_cols, mymat->d_values,X->d_val, Y->d_val, mymat->nrows, mymat->ncols)))
        }else{
            TIME_KERNEL((sparse_mvm<128><<<blocks,threads>>>(mymat->d_rows, mymat->d_cols, mymat->d_values,X->d_val, Y->d_val, mymat->nrows, mymat->ncols)))
        }
        break;
    case 64:
        if(shared){
            TIME_KERNEL((sparse_mvm_shared<64><<<blocks,threads,shms>>>(mymat->d_rows, mymat->d_cols, mymat->d_values,X->d_val, Y->d_val, mymat->nrows, mymat->ncols)))
        }else{
            TIME_KERNEL((sparse_mvm<64><<<blocks,threads>>>(mymat->d_rows, mymat->d_cols, mymat->d_values,X->d_val, Y->d_val, mymat->nrows, mymat->ncols)))
        }
        break;
    case 32:
        if(shared){
            TIME_KERNEL((sparse_mvm_shared<32><<<blocks,threads,shms>>>(mymat->d_rows, mymat->d_cols, mymat->d_values,X->d_val, Y->d_val, mymat->nrows, mymat->ncols)))
        }else{
            TIME_KERNEL((sparse_mvm<32><<<blocks,threads>>>(mymat->d_rows, mymat->d_cols, mymat->d_values,X->d_val, Y->d_val, mymat->nrows, mymat->ncols)))
        }
        break;
    case 16:
        if(shared){
            TIME_KERNEL((sparse_mvm_shared<16><<<blocks,threads,shms>>>(mymat->d_rows, mymat->d_cols, mymat->d_values,X->d_val, Y->d_val, mymat->nrows, mymat->ncols)))
        }else{
            TIME_KERNEL((sparse_mvm<16><<<blocks,threads>>>(mymat->d_rows, mymat->d_cols, mymat->d_values,X->d_val, Y->d_val, mymat->nrows, mymat->ncols)))
        }
        break;
    case 8:
        if(shared){
            TIME_KERNEL((sparse_mvm_shared<8><<<blocks,threads,shms>>>(mymat->d_rows, mymat->d_cols, mymat->d_values,X->d_val, Y->d_val, mymat->nrows, mymat->ncols)))
        }else{
            TIME_KERNEL((sparse_mvm<8><<<blocks,threads>>>(mymat->d_rows, mymat->d_cols, mymat->d_values,X->d_val, Y->d_val, mymat->nrows, mymat->ncols)))
        }
        break;
    case 4:
        if(shared){
            TIME_KERNEL((sparse_mvm_shared<4><<<blocks,threads,shms>>>(mymat->d_rows, mymat->d_cols, mymat->d_values,X->d_val, Y->d_val, mymat->nrows, mymat->ncols)))
        }else{
            TIME_KERNEL((sparse_mvm<4><<<blocks,threads>>>(mymat->d_rows, mymat->d_cols, mymat->d_values,X->d_val, Y->d_val, mymat->nrows, mymat->ncols)))
        }
        break;
    case 2:
        TIME_KERNEL((sparse_mvm<2><<<blocks,threads>>>(mymat->d_rows, mymat->d_cols, mymat->d_values,X->d_val, Y->d_val, mymat->nrows, mymat->ncols)))
        break;
    default:
        TIME_KERNEL((sparse_mvm<1><<<blocks,threads>>>(mymat->d_rows, mymat->d_cols, mymat->d_values,X->d_val, Y->d_val, mymat->nrows, mymat->ncols)))
        break;
    }
}

int main(int argc, char const *argv[]) {

    if( argc < 2 ){
        cout << "Usage: ./spmv <matrix market file>" << endl;
        return -1;
    }

    int ierr {};

    string filename {string(argv[1])};

    // int ntrials {atoi(argv[2])};

    std::unique_ptr<CSRMatrix> mymat {}; 

    CSRMatrixReader reader(filename);

    ierr = reader.mm_init_csr(mymat); // allocate memory

    if(ierr){
        cout << "Error" << ierr << endl;
        return ierr;
    }

    int mnnzpr = reader.mm_read_csr(mymat); //read from file and convert from coo to csr
    int avgnnzpr = mymat->nnz/mymat->nrows;
    cout << "nrows, ncols, nnz: " << mymat->nrows << ' ' << mymat->ncols << ' '  << mymat->nnz << endl;
    cout << "mnnzpr: " << mnnzpr << endl;
    cout << "avg nnzpr: " << avgnnzpr << endl;

    // mymat->print(); // Print all values. Commented out for large matrices.

    DenseVector X(mymat->ncols);

    X.generate(); // Fill with random numbers 

    DenseVector Y(mymat->ncols); // Initialize with zeros

    // X.print();
    // Y.print();

    DenseVector Ycsp(mymat->ncols); // Initialize with zeros

    // checkCudaErrors(cudaDeviceSetSharedMemConfig ( cudaSharedMemBankSizeEightByte ));

    // Use cuSparse
    // CUSPARSE APIs
    {
        cusparseHandle_t     handle = NULL;
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        void*                dBuffer    = NULL;
        size_t               bufferSize = 0;
        double alpha = 1.0;
        double beta  = 0.0;

        CHECK_CUSPARSE( cusparseCreate(&handle) )
        // Create sparse matrix A in CSR format
        CHECK_CUSPARSE( cusparseCreateCsr(&matA, mymat->nrows, mymat->ncols, mymat->nnz,
                                          mymat->d_rows, mymat->d_cols, mymat->d_values,
                                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
        // Create dense vector X
        CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, mymat->ncols, X.d_val, CUDA_R_64F) )
        // Create dense vector y
        CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, mymat->nrows, Ycsp.d_val, CUDA_R_64F) )
        
        // allocate an external buffer if needed      
        CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                     handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                     CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
        
        cout << "Buffer size: " << bufferSize << endl; 
        checkCudaErrors( cudaMalloc(&dBuffer, bufferSize) );

        // warmup run of the cusparseSpMV call (excluded from timing)
        CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                     CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )
        cudaDeviceSynchronize();

        // averaged cusparseSpMV call over 10 runs
        const int NRUNS = 10;
        double total = 0.0;
        for(int irun = 0; irun < NRUNS; irun++) {    
            auto t0 = std::chrono::high_resolution_clock::now();    
            CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                         CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1.e-6;
        }
        
        cout << endl << "-- cusparseSpMV duration (avg of " << NRUNS << " runs): " << (total / NRUNS) << " ms" << endl << endl;

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
        CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
        CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
        CHECK_CUSPARSE( cusparseDestroy(handle) )
    }

    Ycsp.update_host();

    int lim = 32;

    for( int bs = 4; bs < lim; bs *= 2){
        if(true){
            // reset Y
            Y.fill(0.0);
            run_test(bs,mymat,&X,&Y,mnnzpr,false); 
            Y.update_host(); // only comparing results from last test
            compare_values(&Y, &Ycsp);

            Y.fill(0.0);
            run_test(bs,mymat,&X,&Y,mnnzpr,true); 
            Y.update_host(); // only comparing results from last test
            compare_values(&Y, &Ycsp);
            cout << endl;
        }
    }

    return ierr;
}