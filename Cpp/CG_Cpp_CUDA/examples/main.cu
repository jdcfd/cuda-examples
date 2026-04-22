#include "mmio_reader.cuh"
#include "sparse/csr_matrix.cuh"
#include "sparse/dense_vector.cuh"
#include "sparse/spmv.cuh"
#include "../include/solver.cuh"

#include <cublas_v2.h>
#include <cusparse.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <cmath>
#include <chrono>

void compare(const sparse::DenseVector& x, const sparse::DenseVector& x_true) {
    const_cast<sparse::DenseVector&>(x).update_host(); // ensure host is up to date
    double max_err = 0.0;
    for (int i = 0; i < x.size; i++) {
        double err = std::abs(x.h_val[i] - x_true.h_val[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    std::cout << "Max absolute error in solution: " << max_err << std::endl;
}

__global__ void r1_div_x(double *r1, double *r0, double *b) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid == 0) {
        b[0] = r1[0] / r0[0];
    }
}

__global__ void a_minus(double *a, double *na) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid == 0) {
        na[0] = -(a[0]);
    }
}

int solve_cusparse_cublas(const sparse::CSRMatrix& mat, const sparse::DenseVector& b, sparse::DenseVector& x, double tol, int max_iter, double& elapsed_ms) {
    int N = mat.nrows;
    int nz = mat.nnz;
    
    cublasHandle_t cublasHandle = 0;
    cublasCreate(&cublasHandle);
    
    cusparseHandle_t cusparseHandle = 0;
    cusparseCreate(&cusparseHandle);
    
    double *d_r, *d_p, *d_Ax;
    cudaMalloc((void **)&d_r, N * sizeof(double));
    cudaMalloc((void **)&d_p, N * sizeof(double));
    cudaMalloc((void **)&d_Ax, N * sizeof(double));
    
    cusparseSpMatDescr_t matA = NULL;
    cusparseCreateCsr(&matA, N, N, nz, mat.d_rows, mat.d_cols, mat.d_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
                      
    cusparseDnVecDescr_t vecx = NULL;
    cusparseCreateDnVec(&vecx, N, x.d_val, CUDA_R_64F);
    cusparseDnVecDescr_t vecp = NULL;
    cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_64F);
    cusparseDnVecDescr_t vecAx = NULL;
    cusparseCreateDnVec(&vecAx, N, d_Ax, CUDA_R_64F);
    
    double alpha = 1.0;
    double alpham1 = -1.0;
    double beta = 0.0;
    double r1;
    
    double *d_r1, *d_r0, *d_dot, *d_a, *d_na, *d_b;
    cudaMalloc((void **)&d_r1, sizeof(double));
    cudaMalloc((void **)&d_r0, sizeof(double));
    cudaMalloc((void **)&d_dot, sizeof(double));
    cudaMalloc((void **)&d_a, sizeof(double));
    cudaMalloc((void **)&d_na, sizeof(double));
    cudaMalloc((void **)&d_b, sizeof(double));
    
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(cusparseHandle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecx, &beta, vecAx,
                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    void *buffer = NULL;
    if (bufferSize > 0) cudaMalloc(&buffer, bufferSize);
    
    // Copy b to d_r
    cudaMemcpy(d_r, b.d_val, N * sizeof(double), cudaMemcpyDeviceToDevice);
    
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // initial r = b - A*x
    cusparseSpMV(cusparseHandle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, vecx, &beta, vecAx,
                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);
                 
    cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    
    cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
    cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, d_r1);
    cudaMemcpy(&r1, d_r1, sizeof(double), cudaMemcpyDeviceToHost);
    
    int k = 1;
    while (r1 > tol * tol && k <= max_iter) {
        if (k > 1) {
            r1_div_x<<<1, 1>>>(d_r1, d_r0, d_b);
            cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
            cublasDscal(cublasHandle, N, d_b, d_p, 1);
            
            cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST);
            cublasDaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        } else {
            cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST);
            cublasDcopy(cublasHandle, N, d_r, 1, d_p, 1);
        }
        
        cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST);
        cusparseSpMV(cusparseHandle,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecp, &beta, vecAx,
                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);
                     
        cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
        cublasDdot(cublasHandle, N, d_p, 1, d_Ax, 1, d_dot);
        
        r1_div_x<<<1, 1>>>(d_r1, d_dot, d_a);
        
        cublasDaxpy(cublasHandle, N, d_a, d_p, 1, x.d_val, 1);
        
        a_minus<<<1, 1>>>(d_a, d_na);
        
        cublasDaxpy(cublasHandle, N, d_na, d_Ax, 1, d_r, 1);
        
        cudaMemcpyAsync(d_r0, d_r1, sizeof(double), cudaMemcpyDeviceToDevice);
        
        cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, d_r1);
        cudaMemcpyAsync(&r1, d_r1, sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        
        k++;
    }
    
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1.e-6;
    
    if (buffer) cudaFree(buffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecx);
    cusparseDestroyDnVec(vecp);
    cusparseDestroyDnVec(vecAx);
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);
    
    cudaFree(d_r1);
    cudaFree(d_r0);
    cudaFree(d_dot);
    cudaFree(d_a);
    cudaFree(d_na);
    cudaFree(d_b);
    
    return k - 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./cg_benchmark <matrix market file> [rhs vector file] [reference text file]\n";
        return -1;
    }

    std::string filename = argv[1];
    bool has_b_file = (argc >= 3);
    std::string b_filename = has_b_file ? argv[2] : "";
    bool has_ref_file = (argc >= 4);
    std::string ref_filename = has_ref_file ? argv[3] : "";

    std::unique_ptr<sparse::CSRMatrix> mymat;

    CSRMatrixReader reader(filename);
    int ierr = reader.mm_init_csr(mymat);
    if (ierr) {
        std::cout << "Error initializing matrix: " << ierr << std::endl;
        return ierr;
    }

    reader.mm_read_csr(mymat);
    std::cout << "nrows, ncols, nnz: " << mymat->nrows << ' ' << mymat->ncols << ' ' << mymat->nnz << std::endl;

    if (mymat->nrows != mymat->ncols) {
        std::cout << "Error: Matrix must be square for Conjugate Gradient.\n";
        return -1;
    }

    int n = mymat->nrows;
    sparse::DenseVector x_true(n);
    sparse::DenseVector b(n);

    if (has_b_file) {
        FILE *f = fopen(b_filename.c_str(), "r");
        if (f == NULL) {
            std::cout << "Could not open b file\n";
            return -1;
        }
        MM_typecode matcode;
        if (mm_read_banner(f, &matcode) != 0) {
            std::cout << "Could not process Matrix Market banner.\n";
            return -1;
        }
        if (!mm_is_array(matcode)) {
            std::cout << "b must be an array.\n";
            return -1;
        }
        int M, N;
        if (mm_read_mtx_array_size(f, &M, &N) != 0) {
            return -1;
        }
        if (M != n) {
            std::cout << "b dimension mismatch.\n";
            return -1;
        }
        for (int i = 0; i < M; i++) {
            double val;
            if (fscanf(f, "%lf", &val) != 1) {
                std::cout << "Error reading value\n";
                return -1;
            }
            b.h_val[i] = val;
        }
        fclose(f);
        b.update_device();
    } else {
        // Create ground truth x (x_true)
        for (int i = 0; i < n; ++i) {
            x_true.h_val[i] = 1.0;
        }
        x_true.update_device();

        // Create b = A * x_true
        sparse::multiply(*mymat, x_true, b, 32);
    }

    sparse::DenseVector x_ref(n);
    if (has_ref_file) {
        FILE *f = fopen(ref_filename.c_str(), "r");
        if (f == NULL) {
            std::cout << "Could not open reference file\n";
            return -1;
        }
        for (int i = 0; i < n; ++i) {
            double val;
            if (fscanf(f, "%lf", &val) != 1) {
                std::cout << "Error reading reference value\n";
                return -1;
            }
            x_ref.h_val[i] = val;
        }
        fclose(f);
        // host-only needed for comparison, but good practice
        x_ref.update_device();
    }

    // Initial guess x0 = 0
    sparse::DenseVector x(n);
    x.fill(0.0);

    // Warm-up iteration
    sparse::DenseVector x_warmup(n);
    x_warmup.fill(0.0);
    linsolvers::solve(*mymat, b, x_warmup, linsolvers::SolverType::CG, 1e-6, 1);

    std::cout << "Starting CG solver...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    int iters = linsolvers::solve(*mymat, b, x, linsolvers::SolverType::CG, 1e-6, 10000);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1.e-6;

    if (iters >= 0) {
        std::cout << "CG Converged in " << iters << " iterations.\n";
        std::cout << "Time elapsed: " << elapsed_ms << " ms\n";
        if (iters > 0) std::cout << "Time per iteration: " << elapsed_ms / iters << " ms\n";
    } else {
        std::cout << "CG Failed to converge within max iterations.\n";
    }

    if (has_ref_file) {
        compare(x, x_ref);
    } else if (has_b_file) {
        x.update_host(); // ensure host is up to date
        // std::cout << "Solution vector (first 10 elements):\n";
        // for (int i = 0; i < std::min(10, n); ++i) {
        //     std::cout << x.h_val[i] << " ";
        // }
        // std::cout << "\n";
    } else {
        compare(x, x_true);
    }

    // Now test PCG_JACOBI
    x.fill(0.0);
    std::cout << "\nStarting PCG_JACOBI solver...\n";
    auto t2 = std::chrono::high_resolution_clock::now();
    int iters_pcg = linsolvers::solve(*mymat, b, x, linsolvers::SolverType::PCG_JACOBI, 1e-6, 10000);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms_pcg = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() * 1.e-6;

    if (iters_pcg >= 0) {
        std::cout << "PCG_JACOBI Converged in " << iters_pcg << " iterations.\n";
        std::cout << "Time elapsed: " << elapsed_ms_pcg << " ms\n";
        if (iters_pcg > 0) std::cout << "Time per iteration: " << elapsed_ms_pcg / iters_pcg << " ms\n";
    } else {
        std::cout << "PCG_JACOBI Failed to converge within max iterations.\n";
    }

    if (has_ref_file) {
        compare(x, x_ref);
    } else if (has_b_file) {
        x.update_host(); // ensure host is up to date
        std::cout << "Solution vector (first 10 elements):\n";
        for (int i = 0; i < std::min(10, n); ++i) {
            std::cout << x.h_val[i] << " ";
        }
        std::cout << "\n";
    } else {
        compare(x, x_true);
    }

    // Now test CUSPARSE_CUBLAS
    x.fill(0.0);
    std::cout << "\nStarting CUSPARSE + CUBLAS CG solver...\n";
    double elapsed_ms_cusparse = 0.0;
    int iters_cusparse = solve_cusparse_cublas(*mymat, b, x, 1e-6, 10000, elapsed_ms_cusparse);

    if (iters_cusparse >= 0 && iters_cusparse < 10000) {
        std::cout << "CUSPARSE+CUBLAS CG Converged in " << iters_cusparse << " iterations.\n";
        std::cout << "Time elapsed: " << elapsed_ms_cusparse << " ms\n";
        if (iters_cusparse > 0)
            std::cout << "Time per iteration: " << elapsed_ms_cusparse / iters_cusparse << " ms\n";
    } else {
        std::cout << "CUSPARSE+CUBLAS CG Failed to converge within max iterations (Did " << iters_cusparse << " iterations).\n";
    }

    if (has_ref_file) {
        compare(x, x_ref);
    } else if (has_b_file) {
        x.update_host(); // ensure host is up to date
    } else {
        compare(x, x_true);
    }

    return 0;
}
