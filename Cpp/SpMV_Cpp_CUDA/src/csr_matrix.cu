#include "sparse/csr_matrix.cuh"
#include <iostream>
#include <helper_cuda.h>
#include <stdexcept>

namespace sparse {

template <typename T>
CSRMatrixT<T>::CSRMatrixT(int nr, int nc, int nonzeros) {
    if (nonzeros > ((long)nr) * nc) {
        throw std::invalid_argument("received nnz > nrows * ncols");
    }
    nrows = nr;
    ncols = nc;
    nnz = nonzeros;

    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_rows), sizeof(int) * (nrows + 1)));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_cols), sizeof(int) * nnz));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_values), sizeof(T) * nnz));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_rows), sizeof(int) * (nrows + 1)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_cols), sizeof(int) * nnz));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_values), sizeof(T) * nnz));
}

template <typename T>
CSRMatrixT<T>::~CSRMatrixT() {
    if (h_rows) {
        checkCudaErrors(cudaFreeHost(h_rows));
        h_rows = nullptr;
    }
    if (h_cols) {
        checkCudaErrors(cudaFreeHost(h_cols));
        h_cols = nullptr;
    }
    if (h_values) {
        checkCudaErrors(cudaFreeHost(h_values));
        h_values = nullptr;
    }
    if (d_rows) {
        checkCudaErrors(cudaFree(d_rows));
        d_rows = nullptr;
    }
    if (d_cols) {
        checkCudaErrors(cudaFree(d_cols));
        d_cols = nullptr;
    }
    if (d_values) {
        checkCudaErrors(cudaFree(d_values));
        d_values = nullptr;
    }
    nrows = ncols = nnz = 0;
}

template <typename T>
void CSRMatrixT<T>::update_host() {
    checkCudaErrors(cudaMemcpyAsync(h_values, d_values, sizeof(T) * nnz, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpyAsync(h_cols, d_cols, sizeof(int) * nnz, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpyAsync(h_rows, d_rows, sizeof(int) * (nrows + 1), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}

template <typename T>
void CSRMatrixT<T>::update_device() {
    checkCudaErrors(cudaMemcpyAsync(d_values, h_values, sizeof(T) * nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_cols, h_cols, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_rows, h_rows, sizeof(int) * (nrows + 1), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
}

template <typename T>
void CSRMatrixT<T>::print() const {
    if (nrows > 0 && nnz > 0) {
        std::cout << "Nrows: " << nrows << " Ncols: " << ncols << std::endl;
        std::cout << "Nnz: " << nnz << std::endl;
        for (int i = 0; i < nrows + 1; i++) {
            std::cout << "rows[" << i << "] = " << h_rows[i] << std::endl;
        }
        for (int i = 0; i < nnz; i++) {
            std::cout << "cols[" << i << "]= " << h_cols[i]
                      << ", val[" << i << "]= " << h_values[i] << std::endl;
        }
        std::cout << std::endl;
    } else {
        std::cout << "Matrix has not been initialized." << std::endl;
    }
}

template class CSRMatrixT<double>;
template class CSRMatrixT<float>;

} // namespace sparse
