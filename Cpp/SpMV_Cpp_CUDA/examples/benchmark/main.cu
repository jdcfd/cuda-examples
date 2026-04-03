/*
Sparse Matrix-Vector multiplication benchmark using the 'sparse' library.
*/

#include "helper_cuda.h"
#include "mmio_reader.cuh"
#include "sparse/csr_matrix.cuh"
#include "sparse/dense_vector.cuh"
#include "sparse/spmv.cuh"
#include <cusparse.h>
#include <chrono>
#include <memory>
#include <iostream>
#include <cmath>

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

void compare_values(const sparse::DenseVector& Y, const sparse::DenseVector& Yref)
{
    bool issame = true;
    for (int i = 0; i < Y.size; i++) {
        double error = fabs(Y.h_val[i] - Yref.h_val[i]);
        if (error >= fabs(EPS * Yref.h_val[i])) {
            printf("|%e[%d] - %e[%d]| = %e\n", Y.h_val[i], i, Yref.h_val[i], i, error);
            issame = false;
        }
    }

    if (issame) {
        std::cout << "-- Results are correct!" << std::endl;
    } else {
        std::cout << "-- Results are Wrong!" << std::endl;
    }
}

int main(int argc, char const *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./spmv <matrix market file>" << std::endl;
        return -1;
    }

    std::string filename = argv[1];
    std::unique_ptr<sparse::CSRMatrix> mymat;

    CSRMatrixReader reader(filename);

    int ierr = reader.mm_init_csr(mymat);
    if (ierr) {
        std::cout << "Error initializing matrix: " << ierr << std::endl;
        return ierr;
    }

    int mnnzpr = reader.mm_read_csr(mymat);
    std::cout << "nrows, ncols, nnz: " << mymat->nrows << ' ' << mymat->ncols << ' ' << mymat->nnz << std::endl;
    std::cout << "max nnz per row: " << mnnzpr << std::endl;
    std::cout << "avg nnz per row: " << (mymat->nnz / (double)mymat->nrows) << std::endl;

    sparse::DenseVector X(mymat->ncols);
    X.generate();

    sparse::DenseVector Y(mymat->nrows);
    sparse::DenseVector Ycsp(mymat->nrows);

    // --- cuSPARSE reference ---
    {
        cusparseHandle_t     handle = nullptr;
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        void* dBuffer = nullptr;
        size_t bufferSize = 0;
        double alpha = 1.0, beta = 0.0;

        CHECK_CUSPARSE(cusparseCreate(&handle));
        CHECK_CUSPARSE(cusparseCreateCsr(&matA, mymat->nrows, mymat->ncols, mymat->nnz,
                                         mymat->d_rows, mymat->d_cols, mymat->d_values,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, mymat->ncols, X.d_val, CUDA_R_64F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, mymat->nrows, Ycsp.d_val, CUDA_R_64F));

        CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                               CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

        std::cout << "cuSPARSE buffer size: " << bufferSize << std::endl;
        checkCudaErrors(cudaMalloc(&dBuffer, bufferSize));

        // Warmup
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        cudaDeviceSynchronize();

        // Timed runs
        const int NRUNS = 10;
        double total = 0.0;
        for (int irun = 0; irun < NRUNS; irun++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1.e-6;
        }
        std::cout << "\n-- cusparseSpMV duration (avg of " << NRUNS << " runs): " << (total / NRUNS) << " ms\n" << std::endl;

        CHECK_CUSPARSE(cusparseDestroySpMat(matA));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
        CHECK_CUSPARSE(cusparseDestroy(handle));
        if (dBuffer) cudaFree(dBuffer);
    }

    Ycsp.update_host();

    // Test our library kernels
    for (int bs = 4; bs <= 32; bs *= 2) {
        {
            Y.fill(0.0);
            std::cout << "Running test with block_size=" << bs << " shared=false" << std::endl;
            TIME_KERNEL(sparse::multiply(*mymat, X, Y, bs, false));
            Y.update_host();
            compare_values(Y, Ycsp);

            Y.fill(0.0);
            std::cout << "Running test with block_size=" << bs << " shared=true" << std::endl;
            TIME_KERNEL(sparse::multiply(*mymat, X, Y, bs, true));
            Y.update_host();
            compare_values(Y, Ycsp);
            std::cout << std::endl;
        }
    }

    return 0;
}
