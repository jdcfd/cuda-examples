#include "../include/cg.cuh"
#include <sparse/spmv.cuh>
#include <dot.cuh>

#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <iostream>

namespace linsolvers {

// Functor for y = a * x + y
struct axpy_functor {
    const double* a;
    axpy_functor(const double* a) : a(a) {}
    
    __host__ __device__
    double operator()(const double& x, const double& y) const {
        return (*a) * x + y;
    }
};

// Functor for p = r + beta * p
struct p_update_functor {
    const double* beta;
    p_update_functor(const double* beta) : beta(beta) {}
    
    __host__ __device__
    double operator()(const double& r, const double& p) const {
        return r + (*beta) * p;
    }
};

__global__ void compute_alpha_kernel(const double* r_old_sq, const double* pAp, double* alpha, double* minus_alpha) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        alpha[0] = r_old_sq[0] / pAp[0];
        if (minus_alpha) minus_alpha[0] = -alpha[0];
    }
}

__global__ void compute_beta_kernel(const double* r_new_sq, const double* r_old_sq, double* beta) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        beta[0] = r_new_sq[0] / r_old_sq[0];
    }
}

int cg_solve(const sparse::CSRMatrix& A, 
             const sparse::DenseVector& b, 
             sparse::DenseVector& x, 
             double tol, 
             int max_iters,
             cudaStream_t stream) {
    
    int n = A.nrows;
    
    // Allocate temporary vectors
    sparse::DenseVector r(n);
    sparse::DenseVector p(n);
    sparse::DenseVector Ap(n);
    
    // Wrap device pointers with thrust::device_ptr
    thrust::device_ptr<double> r_ptr(r.d_val);
    thrust::device_ptr<double> p_ptr(p.d_val);
    thrust::device_ptr<double> Ap_ptr(Ap.d_val);
    thrust::device_ptr<double> x_ptr(x.d_val);
    thrust::device_ptr<const double> b_ptr(b.d_val);
    
    double *d_r_old_sq, *d_r_new_sq, *d_pAp, *d_alpha, *d_minus_alpha, *d_beta;
    cudaMalloc(&d_r_old_sq, sizeof(double));
    cudaMalloc(&d_r_new_sq, sizeof(double));
    cudaMalloc(&d_pAp, sizeof(double));
    cudaMalloc(&d_alpha, sizeof(double));
    cudaMalloc(&d_minus_alpha, sizeof(double));
    cudaMalloc(&d_beta, sizeof(double));
    
    struct CudaFreeWrapper {
        double* p1; double* p2; double* p3; double* p4; double* p5; double* p6;
        ~CudaFreeWrapper() {
            if (p1) cudaFree(p1); if (p2) cudaFree(p2); if (p3) cudaFree(p3);
            if (p4) cudaFree(p4); if (p5) cudaFree(p5); if (p6) cudaFree(p6);
        }
    } free_wrapper{d_r_old_sq, d_r_new_sq, d_pAp, d_alpha, d_minus_alpha, d_beta};

    // Ap = A * x_0
    sparse::multiply(A, x, Ap, 32, false, stream);
    
    // r_0 = b - A * x_0 (Ap)
    thrust::transform(thrust::cuda::par.on(stream), b_ptr, b_ptr + n, Ap_ptr, r_ptr, thrust::minus<double>());
    
    // p_0 = r_0
    thrust::copy(thrust::cuda::par.on(stream), r_ptr, r_ptr + n, p_ptr);
    
    // r_old_sq = r.T * r
    cudaMemsetAsync(d_r_old_sq, 0.0, sizeof(double), stream);
    dot_product(r.d_val, r.d_val, d_r_old_sq, n, stream);
    
    // b_norm = ||b||
    double* d_b_sq;
    cudaMalloc(&d_b_sq, sizeof(double));
    cudaMemsetAsync(d_b_sq, 0.0, sizeof(double), stream);
    dot_product(b.d_val, b.d_val, d_b_sq, n, stream);
    
    double h_b_sq = 0.0;
    cudaMemcpyAsync(&h_b_sq, d_b_sq, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_b_sq);
    
    double b_norm = std::sqrt(h_b_sq);
    if (b_norm == 0.0) {
        b_norm = 1.0;
    }
    
    double h_r_old_sq = 0.0;
    cudaMemcpyAsync(&h_r_old_sq, d_r_old_sq, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (std::sqrt(h_r_old_sq) / b_norm < tol) {
        return 0; // Already converged
    }
    
    for (int i = 0; i < max_iters; ++i) {
        // Ap = A * p
        sparse::multiply(A, p, Ap, 32, false, stream);
        
        // pAp = p.T * Ap
        cudaMemsetAsync(d_pAp, 0.0, sizeof(double), stream);
        dot_product(p.d_val, Ap.d_val, d_pAp, n, stream);
        
        // alpha = r_old_sq / pAp
        compute_alpha_kernel<<<1, 1, 0, stream>>>(d_r_old_sq, d_pAp, d_alpha, d_minus_alpha);
        
        // x = x + alpha * p
        thrust::transform(thrust::cuda::par.on(stream), p_ptr, p_ptr + n, x_ptr, x_ptr, axpy_functor(d_alpha));
        
        // r = r - alpha * Ap
        thrust::transform(thrust::cuda::par.on(stream), Ap_ptr, Ap_ptr + n, r_ptr, r_ptr, axpy_functor(d_minus_alpha));
        
        // r_new_sq = r.T * r
        cudaMemsetAsync(d_r_new_sq, 0.0, sizeof(double), stream);
        dot_product(r.d_val, r.d_val, d_r_new_sq, n, stream);
        
        // Check convergence
        double h_r_new_sq = 0.0;
        cudaMemcpyAsync(&h_r_new_sq, d_r_new_sq, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        if (std::sqrt(h_r_new_sq) / b_norm < tol) {
            return i + 1;
        }
        
        // beta = r_new_sq / r_old_sq
        compute_beta_kernel<<<1, 1, 0, stream>>>(d_r_new_sq, d_r_old_sq, d_beta);
        
        // p = r + beta * p
        thrust::transform(thrust::cuda::par.on(stream), r_ptr, r_ptr + n, p_ptr, p_ptr, p_update_functor(d_beta));
        
        // r_old_sq = r_new_sq
        cudaMemcpyAsync(d_r_old_sq, d_r_new_sq, sizeof(double), cudaMemcpyDeviceToDevice, stream);
    }
    
    return -1; // Did not converge within max_iters
}

__global__ void extract_inv_diag_kernel(int nrows, const int* row_ptrs, const int* cols, const double* values, double* inv_diag) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        int start = row_ptrs[row];
        int end = row_ptrs[row + 1];
        double diag_val = 1.0;
        for (int i = start; i < end; ++i) {
            if (cols[i] == row) {
                diag_val = values[i];
                break;
            }
        }
        inv_diag[row] = (diag_val != 0.0) ? 1.0 / diag_val : 1.0;
    }
}

int pcg_solve(const sparse::CSRMatrix& A, 
              const sparse::DenseVector& b, 
              sparse::DenseVector& x, 
              double tol, 
              int max_iters,
              cudaStream_t stream) {
    
    int n = A.nrows;
    
    // Allocate temporary vectors
    sparse::DenseVector r(n);
    sparse::DenseVector z(n);
    sparse::DenseVector p(n);
    sparse::DenseVector Ap(n);
    sparse::DenseVector inv_diag(n);
    
    // Extract inverse diagonal
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    extract_inv_diag_kernel<<<blocks, threads_per_block, 0, stream>>>(n, A.d_rows, A.d_cols, A.d_values, inv_diag.d_val);
    cudaStreamSynchronize(stream);
    
    // Wrap device pointers with thrust::device_ptr
    thrust::device_ptr<double> r_ptr(r.d_val);
    thrust::device_ptr<double> z_ptr(z.d_val);
    thrust::device_ptr<double> p_ptr(p.d_val);
    thrust::device_ptr<double> Ap_ptr(Ap.d_val);
    thrust::device_ptr<double> x_ptr(x.d_val);
    thrust::device_ptr<const double> b_ptr(b.d_val);
    thrust::device_ptr<double> inv_diag_ptr(inv_diag.d_val);
    
    double *d_r_old_sq, *d_r_new_sq, *d_r_new_z_sq, *d_pAp, *d_alpha, *d_minus_alpha, *d_beta;
    cudaMalloc(&d_r_old_sq, sizeof(double));
    cudaMalloc(&d_r_new_sq, sizeof(double));
    cudaMalloc(&d_r_new_z_sq, sizeof(double));
    cudaMalloc(&d_pAp, sizeof(double));
    cudaMalloc(&d_alpha, sizeof(double));
    cudaMalloc(&d_minus_alpha, sizeof(double));
    cudaMalloc(&d_beta, sizeof(double));
    
    struct CudaFreeWrapper {
        double* p1; double* p2; double* p3; double* p4; double* p5; double* p6; double* p7;
        ~CudaFreeWrapper() {
            if (p1) cudaFree(p1); if (p2) cudaFree(p2); if (p3) cudaFree(p3);
            if (p4) cudaFree(p4); if (p5) cudaFree(p5); if (p6) cudaFree(p6); if (p7) cudaFree(p7);
        }
    } free_wrapper{d_r_old_sq, d_r_new_sq, d_r_new_z_sq, d_pAp, d_alpha, d_minus_alpha, d_beta};

    // Ap = A * x_0
    sparse::multiply(A, x, Ap, 32, false, stream);
    
    // r_0 = b - A * x_0 (Ap)
    thrust::transform(thrust::cuda::par.on(stream), b_ptr, b_ptr + n, Ap_ptr, r_ptr, thrust::minus<double>());
    
    // z_0 = M^-1 r_0
    thrust::transform(thrust::cuda::par.on(stream), inv_diag_ptr, inv_diag_ptr + n, r_ptr, z_ptr, thrust::multiplies<double>());
    
    // p_0 = z_0
    thrust::copy(thrust::cuda::par.on(stream), z_ptr, z_ptr + n, p_ptr);
    
    // r_old_sq = r.T * z
    cudaMemsetAsync(d_r_old_sq, 0.0, sizeof(double), stream);
    dot_product(r.d_val, z.d_val, d_r_old_sq, n, stream);
    
    // b_norm = ||b||
    double* d_b_sq;
    cudaMalloc(&d_b_sq, sizeof(double));
    cudaMemsetAsync(d_b_sq, 0.0, sizeof(double), stream);
    dot_product(b.d_val, b.d_val, d_b_sq, n, stream);
    double h_b_sq = 0.0;
    cudaMemcpyAsync(&h_b_sq, d_b_sq, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_b_sq);
    double b_norm = std::sqrt(h_b_sq);
    if (b_norm == 0.0) {
        b_norm = 1.0;
    }
    
    cudaMemsetAsync(d_r_new_sq, 0.0, sizeof(double), stream);
    dot_product(r.d_val, r.d_val, d_r_new_sq, n, stream);
    double h_r_new_sq = 0.0;
    cudaMemcpyAsync(&h_r_new_sq, d_r_new_sq, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    double r_norm = std::sqrt(h_r_new_sq);
    if (r_norm / b_norm < tol) {
        return 0; // Already converged
    }
    
    for (int i = 0; i < max_iters; ++i) {
        // Ap = A * p
        sparse::multiply(A, p, Ap, 32, false, stream);
        
        // pAp = p.T * Ap
        cudaMemsetAsync(d_pAp, 0.0, sizeof(double), stream);
        dot_product(p.d_val, Ap.d_val, d_pAp, n, stream);
        
        // alpha = r_old_sq / pAp
        compute_alpha_kernel<<<1, 1, 0, stream>>>(d_r_old_sq, d_pAp, d_alpha, d_minus_alpha);
        
        // x = x + alpha * p
        thrust::transform(thrust::cuda::par.on(stream), p_ptr, p_ptr + n, x_ptr, x_ptr, axpy_functor(d_alpha));
        
        // r = r - alpha * Ap
        thrust::transform(thrust::cuda::par.on(stream), Ap_ptr, Ap_ptr + n, r_ptr, r_ptr, axpy_functor(d_minus_alpha));
        
        // Check convergence
        cudaMemsetAsync(d_r_new_sq, 0.0, sizeof(double), stream);
        dot_product(r.d_val, r.d_val, d_r_new_sq, n, stream);
        cudaMemcpyAsync(&h_r_new_sq, d_r_new_sq, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        r_norm = std::sqrt(h_r_new_sq);
        if (r_norm / b_norm < tol) {
            return i + 1;
        }
        
        // z = M^-1 r
        thrust::transform(thrust::cuda::par.on(stream), inv_diag_ptr, inv_diag_ptr + n, r_ptr, z_ptr, thrust::multiplies<double>());
        
        // r_new_z_sq = r.T * z
        cudaMemsetAsync(d_r_new_z_sq, 0.0, sizeof(double), stream);
        dot_product(r.d_val, z.d_val, d_r_new_z_sq, n, stream);
        
        // beta = r_new_z_sq / r_old_sq
        compute_beta_kernel<<<1, 1, 0, stream>>>(d_r_new_z_sq, d_r_old_sq, d_beta);
        
        // p = z + beta * p
        thrust::transform(thrust::cuda::par.on(stream), z_ptr, z_ptr + n, p_ptr, p_ptr, p_update_functor(d_beta));
        
        // r_old_sq = r_new_z_sq
        cudaMemcpyAsync(d_r_old_sq, d_r_new_z_sq, sizeof(double), cudaMemcpyDeviceToDevice, stream);
    }
    
    return -1; // Did not converge within max_iters
}

} // namespace linsolvers
