#include "../include/cg.cuh"
#include <sparse/spmv.cuh>

#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <cmath>
#include <iostream>

namespace cg {

// Functor for y = a * x + y
struct axpy_functor {
    const double a;
    axpy_functor(double a) : a(a) {}
    
    __host__ __device__
    double operator()(const double& x, const double& y) const {
        return a * x + y;
    }
};

// Functor for p = r + beta * p
struct p_update_functor {
    const double beta;
    p_update_functor(double beta) : beta(beta) {}
    
    __host__ __device__
    double operator()(const double& r, const double& p) const {
        return r + beta * p;
    }
};

int solve(const sparse::CSRMatrix& A, 
          const sparse::DenseVector& b, 
          sparse::DenseVector& x, 
          double tol, 
          int max_iters) {
    
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
    
    // Ap = A * x_0
    sparse::multiply(A, x, Ap, 32);
    
    // r_0 = b - A * x_0 (Ap)
    thrust::transform(b_ptr, b_ptr + n, Ap_ptr, r_ptr, thrust::minus<double>());
    
    // p_0 = r_0
    thrust::copy(r_ptr, r_ptr + n, p_ptr);
    
    // r_old_sq = r.T * r
    double r_old_sq = thrust::inner_product(r_ptr, r_ptr + n, r_ptr, 0.0);
    
    // b_norm = ||b||
    double b_norm = std::sqrt(thrust::inner_product(b_ptr, b_ptr + n, b_ptr, 0.0));
    if (b_norm == 0.0) {
        b_norm = 1.0;
    }
    
    if (std::sqrt(r_old_sq) / b_norm < tol) {
        return 0; // Already converged
    }
    
    for (int i = 0; i < max_iters; ++i) {
        // Ap = A * p
        sparse::multiply(A, p, Ap, 32);
        
        // pAp = p.T * Ap
        double pAp = thrust::inner_product(p_ptr, p_ptr + n, Ap_ptr, 0.0);
        
        // alpha = r_old_sq / pAp
        double alpha = r_old_sq / pAp;
        
        // x = x + alpha * p
        thrust::transform(p_ptr, p_ptr + n, x_ptr, x_ptr, axpy_functor(alpha));
        
        // r = r - alpha * Ap
        thrust::transform(Ap_ptr, Ap_ptr + n, r_ptr, r_ptr, axpy_functor(-alpha));
        
        // r_new_sq = r.T * r
        double r_new_sq = thrust::inner_product(r_ptr, r_ptr + n, r_ptr, 0.0);
        
        // Check convergence
        if (std::sqrt(r_new_sq) / b_norm < tol) {
            return i + 1;
        }
        
        // beta = r_new_sq / r_old_sq
        double beta = r_new_sq / r_old_sq;
        
        // p = r + beta * p
        thrust::transform(r_ptr, r_ptr + n, p_ptr, p_ptr, p_update_functor(beta));
        
        r_old_sq = r_new_sq;
    }
    
    return -1; // Did not converge within max_iters
}

} // namespace cg
