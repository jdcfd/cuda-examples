#include "mmio_reader.cuh"
#include "sparse/csr_matrix.cuh"
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <stdexcept>

CSRMatrixReader::CSRMatrixReader(const std::string& fname) : filename(fname) {
    f = fopen(filename.c_str(), "r");
    if (!f) {
        throw std::runtime_error("Could not open matrix file: " + filename);
    }
}

CSRMatrixReader::~CSRMatrixReader() {
    if (f) fclose(f);
}

int CSRMatrixReader::mm_init_csr(std::unique_ptr<sparse::CSRMatrix>& mat) {
    int ierr = mm_read_banner(this->f, &(this->mmtc));
    if (ierr) {
        std::cout << "Error reading Banner!" << std::endl;
        return -1;
    }

    if (!mm_is_sparse(this->mmtc)) {
        throw std::invalid_argument("Non-sparse matrices not supported!");
    }

    int nrows, ncols, nnz_file;
    ierr = mm_read_mtx_crd_size(this->f, &nrows, &ncols, &nnz_file);

    this->nnz_from_file = nnz_file;
    this->is_symmetric = mm_is_symmetric(this->mmtc);

    if (ierr) return ierr;

    int nnz = nnz_file;
    if (this->is_symmetric) {
        nnz = nnz_file * 2 - nrows; // assumes diagonals present
    }

    if (!mat) {
        mat = std::make_unique<sparse::CSRMatrix>(nrows, ncols, nnz);
    } else {
        std::cout << "Matrix is not empty!" << std::endl;
        return -1;
    }
    return ierr;
}

int CSRMatrixReader::mm_read_csr(std::unique_ptr<sparse::CSRMatrix>& mat) {
    int *indx = new int[mat->nnz];
    int ierr = mm_read_mtx_crd_data(this->f, mat->nrows, mat->ncols, this->nnz_from_file,
                                    indx, mat->h_cols, mat->h_values, this->mmtc);

    for (int i = 0; i < this->nnz_from_file; i++) {
        indx[i]--;
        mat->h_cols[i]--;
    }

    if (ierr) {
        std::cout << "Error while reading matrix data" << std::endl;
        delete[] indx;
        return ierr;
    }

    if (this->is_symmetric) {
        int k = this->nnz_from_file;
        for (int i = 0; i < this->nnz_from_file; i++) {
            if (indx[i] != mat->h_cols[i]) {
                indx[k] = mat->h_cols[i];
                mat->h_cols[k] = indx[i];
                mat->h_values[k] = mat->h_values[i];
                k++;
            }
        }
    }

    int &nnz = mat->nnz;
    int m = mat->nrows;

    // sort COO
    auto begin_keys = thrust::make_zip_iterator(thrust::make_tuple(&indx[0], mat->h_cols));
    auto end_keys   = thrust::make_zip_iterator(thrust::make_tuple(&indx[0] + nnz, mat->h_cols + nnz));

    thrust::stable_sort_by_key(thrust::host, begin_keys, end_keys, mat->h_values,
                               thrust::less<thrust::tuple<int,int>>());

    // build row offsets
    thrust::fill(thrust::host, mat->h_rows, mat->h_rows + m + 1, 0);
    for (int nz = 0; nz < nnz; ++nz)
        mat->h_rows[indx[nz]]++;

    int max_nnz_per_row = 0;
    for (int i = 0; i < m; ++i)
        if (mat->h_rows[i] > max_nnz_per_row)
            max_nnz_per_row = mat->h_rows[i];

    thrust::exclusive_scan(mat->h_rows, mat->h_rows + m + 1, mat->h_rows);

    mat->update_device();

    delete[] indx;
    return max_nnz_per_row;
}
