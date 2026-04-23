#pragma once
#include <iostream>
#include <memory>
#include <string>
extern "C" {
#include "mmio.h"
}
#include "sparse/csr_matrix.cuh"

class CSRMatrixReader {
    FILE* f = nullptr;
    MM_typecode mmtc;
    int nnz_from_file = 0;
    bool is_symmetric = false;

public:
    std::string filename;

    CSRMatrixReader(const std::string& fname);
    ~CSRMatrixReader();

    int mm_init_csr(std::unique_ptr<sparse::CSRMatrix>& mat);
    int mm_init_csr(std::unique_ptr<sparse::CSRMatrixF>& mat);
    int mm_read_csr(std::unique_ptr<sparse::CSRMatrix>& mat);
    int mm_read_csr(std::unique_ptr<sparse::CSRMatrixF>& mat);
};
