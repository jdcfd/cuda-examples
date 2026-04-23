#include "../include/cg.cuh"
#include <dot.cuh>
#include <sparse/spmv.cuh>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cmath>

namespace linsolvers {

namespace {

template <typename T>
struct axpy_functor {
    const T* a;
    explicit axpy_functor(const T* a_in) : a(a_in) {}

    __host__ __device__ T operator()(const T& x, const T& y) const { return (*a) * x + y; }
};

template <typename T>
struct p_update_functor {
    const T* beta;
    explicit p_update_functor(const T* beta_in) : beta(beta_in) {}

    __host__ __device__ T operator()(const T& r, const T& p) const { return r + (*beta) * p; }
};

template <typename T>
__global__ void compute_alpha_kernel(const T* r_old_sq, const T* pAp, T* alpha, T* minus_alpha)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        alpha[0] = r_old_sq[0] / pAp[0];
        if (minus_alpha) {
            minus_alpha[0] = -alpha[0];
        }
    }
}

template <typename T>
__global__ void compute_beta_kernel(const T* r_new_sq, const T* r_old_sq, T* beta)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        beta[0] = r_new_sq[0] / r_old_sq[0];
    }
}

template <typename T>
__global__ void extract_inv_diag_kernel(int nrows,
                                        const int* row_ptrs,
                                        const int* cols,
                                        const T* values,
                                        T* inv_diag)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        int start = row_ptrs[row];
        int end = row_ptrs[row + 1];
        T diag_val = T(1);
        for (int i = start; i < end; ++i) {
            if (cols[i] == row) {
                diag_val = values[i];
                break;
            }
        }
        inv_diag[row] = (diag_val != T(0)) ? T(1) / diag_val : T(1);
    }
}

template <typename T>
int cg_solve_impl(const sparse::CSRMatrixT<T>& A,
                  const sparse::DenseVectorT<T>& b,
                  sparse::DenseVectorT<T>& x,
                  double tol,
                  int max_iters,
                  cudaStream_t stream)
{
    int n = A.nrows;

    sparse::DenseVectorT<T> r(n);
    sparse::DenseVectorT<T> p(n);
    sparse::DenseVectorT<T> Ap(n);

    thrust::device_ptr<T> r_ptr(r.d_val);
    thrust::device_ptr<T> p_ptr(p.d_val);
    thrust::device_ptr<T> Ap_ptr(Ap.d_val);
    thrust::device_ptr<T> x_ptr(x.d_val);
    thrust::device_ptr<const T> b_ptr(b.d_val);

    T *d_r_old_sq, *d_r_new_sq, *d_pAp, *d_alpha, *d_minus_alpha, *d_beta;
    cudaMalloc(&d_r_old_sq, sizeof(T));
    cudaMalloc(&d_r_new_sq, sizeof(T));
    cudaMalloc(&d_pAp, sizeof(T));
    cudaMalloc(&d_alpha, sizeof(T));
    cudaMalloc(&d_minus_alpha, sizeof(T));
    cudaMalloc(&d_beta, sizeof(T));

    struct CudaFreeWrapper {
        T* p1;
        T* p2;
        T* p3;
        T* p4;
        T* p5;
        T* p6;
        ~CudaFreeWrapper()
        {
            if (p1) cudaFree(p1);
            if (p2) cudaFree(p2);
            if (p3) cudaFree(p3);
            if (p4) cudaFree(p4);
            if (p5) cudaFree(p5);
            if (p6) cudaFree(p6);
        }
    } free_wrapper{d_r_old_sq, d_r_new_sq, d_pAp, d_alpha, d_minus_alpha, d_beta};

    sparse::multiply(A, x, Ap, 32, false, stream);

    thrust::transform(thrust::cuda::par.on(stream), b_ptr, b_ptr + n, Ap_ptr, r_ptr, thrust::minus<T>());

    thrust::copy(thrust::cuda::par.on(stream), r_ptr, r_ptr + n, p_ptr);

    cudaMemsetAsync(d_r_old_sq, 0, sizeof(T), stream);
    dot_product(r.d_val, r.d_val, d_r_old_sq, n, stream);

    T* d_b_sq = nullptr;
    cudaMalloc(&d_b_sq, sizeof(T));
    cudaMemsetAsync(d_b_sq, 0, sizeof(T), stream);
    dot_product(b.d_val, b.d_val, d_b_sq, n, stream);

    T h_b_sq = T(0);
    cudaMemcpyAsync(&h_b_sq, d_b_sq, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_b_sq);

    double b_norm = std::sqrt(static_cast<double>(h_b_sq));
    if (b_norm == 0.0) {
        b_norm = 1.0;
    }

    T h_r_old_sq = T(0);
    cudaMemcpyAsync(&h_r_old_sq, d_r_old_sq, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (std::sqrt(static_cast<double>(h_r_old_sq)) / b_norm < tol) {
        return 0;
    }

    for (int i = 0; i < max_iters; ++i) {
        sparse::multiply(A, p, Ap, 32, false, stream);

        cudaMemsetAsync(d_pAp, 0, sizeof(T), stream);
        dot_product(p.d_val, Ap.d_val, d_pAp, n, stream);

        compute_alpha_kernel<T><<<1, 1, 0, stream>>>(d_r_old_sq, d_pAp, d_alpha, d_minus_alpha);

        thrust::transform(thrust::cuda::par.on(stream), p_ptr, p_ptr + n, x_ptr, x_ptr, axpy_functor<T>(d_alpha));

        thrust::transform(thrust::cuda::par.on(stream), Ap_ptr, Ap_ptr + n, r_ptr, r_ptr, axpy_functor<T>(d_minus_alpha));

        cudaMemsetAsync(d_r_new_sq, 0, sizeof(T), stream);
        dot_product(r.d_val, r.d_val, d_r_new_sq, n, stream);

        T h_r_new_sq = T(0);
        cudaMemcpyAsync(&h_r_new_sq, d_r_new_sq, sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (std::sqrt(static_cast<double>(h_r_new_sq)) / b_norm < tol) {
            return i + 1;
        }

        compute_beta_kernel<T><<<1, 1, 0, stream>>>(d_r_new_sq, d_r_old_sq, d_beta);

        thrust::transform(thrust::cuda::par.on(stream), r_ptr, r_ptr + n, p_ptr, p_ptr, p_update_functor<T>(d_beta));

        cudaMemcpyAsync(d_r_old_sq, d_r_new_sq, sizeof(T), cudaMemcpyDeviceToDevice, stream);
    }

    return -1;
}

template <typename T>
int pcg_solve_impl(const sparse::CSRMatrixT<T>& A,
                   const sparse::DenseVectorT<T>& b,
                   sparse::DenseVectorT<T>& x,
                   double tol,
                   int max_iters,
                   cudaStream_t stream)
{
    int n = A.nrows;

    sparse::DenseVectorT<T> r(n);
    sparse::DenseVectorT<T> z(n);
    sparse::DenseVectorT<T> p(n);
    sparse::DenseVectorT<T> Ap(n);
    sparse::DenseVectorT<T> inv_diag(n);

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    extract_inv_diag_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
        n, A.d_rows, A.d_cols, A.d_values, inv_diag.d_val);
    cudaStreamSynchronize(stream);

    thrust::device_ptr<T> r_ptr(r.d_val);
    thrust::device_ptr<T> z_ptr(z.d_val);
    thrust::device_ptr<T> p_ptr(p.d_val);
    thrust::device_ptr<T> Ap_ptr(Ap.d_val);
    thrust::device_ptr<T> x_ptr(x.d_val);
    thrust::device_ptr<const T> b_ptr(b.d_val);
    thrust::device_ptr<T> inv_diag_ptr(inv_diag.d_val);

    T *d_r_old_sq, *d_r_new_sq, *d_r_new_z_sq, *d_pAp, *d_alpha, *d_minus_alpha, *d_beta;
    cudaMalloc(&d_r_old_sq, sizeof(T));
    cudaMalloc(&d_r_new_sq, sizeof(T));
    cudaMalloc(&d_r_new_z_sq, sizeof(T));
    cudaMalloc(&d_pAp, sizeof(T));
    cudaMalloc(&d_alpha, sizeof(T));
    cudaMalloc(&d_minus_alpha, sizeof(T));
    cudaMalloc(&d_beta, sizeof(T));

    struct CudaFreeWrapper {
        T* p1;
        T* p2;
        T* p3;
        T* p4;
        T* p5;
        T* p6;
        T* p7;
        ~CudaFreeWrapper()
        {
            if (p1) cudaFree(p1);
            if (p2) cudaFree(p2);
            if (p3) cudaFree(p3);
            if (p4) cudaFree(p4);
            if (p5) cudaFree(p5);
            if (p6) cudaFree(p6);
            if (p7) cudaFree(p7);
        }
    } free_wrapper{d_r_old_sq, d_r_new_sq, d_r_new_z_sq, d_pAp, d_alpha, d_minus_alpha, d_beta};

    sparse::multiply(A, x, Ap, 32, false, stream);

    thrust::transform(thrust::cuda::par.on(stream), b_ptr, b_ptr + n, Ap_ptr, r_ptr, thrust::minus<T>());

    thrust::transform(thrust::cuda::par.on(stream), inv_diag_ptr, inv_diag_ptr + n, r_ptr, z_ptr, thrust::multiplies<T>());

    thrust::copy(thrust::cuda::par.on(stream), z_ptr, z_ptr + n, p_ptr);

    cudaMemsetAsync(d_r_old_sq, 0, sizeof(T), stream);
    dot_product(r.d_val, z.d_val, d_r_old_sq, n, stream);

    T* d_b_sq = nullptr;
    cudaMalloc(&d_b_sq, sizeof(T));
    cudaMemsetAsync(d_b_sq, 0, sizeof(T), stream);
    dot_product(b.d_val, b.d_val, d_b_sq, n, stream);

    T h_b_sq = T(0);
    cudaMemcpyAsync(&h_b_sq, d_b_sq, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_b_sq);

    double b_norm = std::sqrt(static_cast<double>(h_b_sq));
    if (b_norm == 0.0) {
        b_norm = 1.0;
    }

    cudaMemsetAsync(d_r_new_sq, 0, sizeof(T), stream);
    dot_product(r.d_val, r.d_val, d_r_new_sq, n, stream);
    T h_r_new_sq = T(0);
    cudaMemcpyAsync(&h_r_new_sq, d_r_new_sq, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    double r_norm = std::sqrt(static_cast<double>(h_r_new_sq));
    if (r_norm / b_norm < tol) {
        return 0;
    }

    for (int i = 0; i < max_iters; ++i) {
        sparse::multiply(A, p, Ap, 32, false, stream);

        cudaMemsetAsync(d_pAp, 0, sizeof(T), stream);
        dot_product(p.d_val, Ap.d_val, d_pAp, n, stream);

        compute_alpha_kernel<T><<<1, 1, 0, stream>>>(d_r_old_sq, d_pAp, d_alpha, d_minus_alpha);

        thrust::transform(thrust::cuda::par.on(stream), p_ptr, p_ptr + n, x_ptr, x_ptr, axpy_functor<T>(d_alpha));

        thrust::transform(thrust::cuda::par.on(stream), Ap_ptr, Ap_ptr + n, r_ptr, r_ptr, axpy_functor<T>(d_minus_alpha));

        cudaMemsetAsync(d_r_new_sq, 0, sizeof(T), stream);
        dot_product(r.d_val, r.d_val, d_r_new_sq, n, stream);
        cudaMemcpyAsync(&h_r_new_sq, d_r_new_sq, sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        r_norm = std::sqrt(static_cast<double>(h_r_new_sq));
        if (r_norm / b_norm < tol) {
            return i + 1;
        }

        thrust::transform(thrust::cuda::par.on(stream), inv_diag_ptr, inv_diag_ptr + n, r_ptr, z_ptr, thrust::multiplies<T>());

        cudaMemsetAsync(d_r_new_z_sq, 0, sizeof(T), stream);
        dot_product(r.d_val, z.d_val, d_r_new_z_sq, n, stream);

        compute_beta_kernel<T><<<1, 1, 0, stream>>>(d_r_new_z_sq, d_r_old_sq, d_beta);

        thrust::transform(thrust::cuda::par.on(stream), z_ptr, z_ptr + n, p_ptr, p_ptr, p_update_functor<T>(d_beta));

        cudaMemcpyAsync(d_r_old_sq, d_r_new_z_sq, sizeof(T), cudaMemcpyDeviceToDevice, stream);
    }

    return -1;
}

} // namespace

int cg_solve(const sparse::CSRMatrix& A,
             const sparse::DenseVector& b,
             sparse::DenseVector& x,
             double tol,
             int max_iters,
             cudaStream_t stream)
{
    return cg_solve_impl<double>(A, b, x, tol, max_iters, stream);
}

int cg_solve(const sparse::CSRMatrixF& A,
             const sparse::DenseVectorF& b,
             sparse::DenseVectorF& x,
             double tol,
             int max_iters,
             cudaStream_t stream)
{
    return cg_solve_impl<float>(A, b, x, tol, max_iters, stream);
}

int pcg_solve(const sparse::CSRMatrix& A,
              const sparse::DenseVector& b,
              sparse::DenseVector& x,
              double tol,
              int max_iters,
              cudaStream_t stream)
{
    return pcg_solve_impl<double>(A, b, x, tol, max_iters, stream);
}

int pcg_solve(const sparse::CSRMatrixF& A,
              const sparse::DenseVectorF& b,
              sparse::DenseVectorF& x,
              double tol,
              int max_iters,
              cudaStream_t stream)
{
    return pcg_solve_impl<float>(A, b, x, tol, max_iters, stream);
}

} // namespace linsolvers
