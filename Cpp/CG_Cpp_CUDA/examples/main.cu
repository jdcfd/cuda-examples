#include "mmio_reader.cuh"
#include "sparse/csr_matrix.cuh"
#include "sparse/dense_vector.cuh"
#include "sparse/spmv.cuh"
#include "../include/solver.cuh"

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

    return 0;
}
