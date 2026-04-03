#include "mmio_reader.cuh"
#include "sparse/csr_matrix.cuh"
#include "sparse/dense_vector.cuh"
#include "sparse/spmv.cuh"
#include "../include/cg.cuh"

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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./cg_benchmark <matrix market file>\n";
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

    reader.mm_read_csr(mymat);
    std::cout << "nrows, ncols, nnz: " << mymat->nrows << ' ' << mymat->ncols << ' ' << mymat->nnz << std::endl;

    if (mymat->nrows != mymat->ncols) {
        std::cout << "Error: Matrix must be square for Conjugate Gradient.\n";
        return -1;
    }

    int n = mymat->nrows;

    // Create ground truth x (x_true)
    sparse::DenseVector x_true(n);
    for (int i = 0; i < n; ++i) {
        x_true.h_val[i] = 1.0;
    }
    x_true.update_device();

    // Create b = A * x_true
    sparse::DenseVector b(n);
    sparse::multiply(*mymat, x_true, b, 32);

    // Initial guess x0 = 0
    sparse::DenseVector x(n);
    x.fill(0.0);

    // Warm-up iteration
    sparse::DenseVector x_warmup(n);
    x_warmup.fill(0.0);
    cg::solve(*mymat, b, x_warmup, 1e-6, 1);

    std::cout << "Starting CG solver...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    int iters = cg::solve(*mymat, b, x, 1e-6, 10000);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1.e-6;

    if (iters >= 0) {
        std::cout << "CG Converged in " << iters << " iterations.\n";
        std::cout << "Time elapsed: " << elapsed_ms << " ms\n";
    } else {
        std::cout << "CG Failed to converge within max iterations.\n";
    }

    compare(x, x_true);

    return 0;
}
