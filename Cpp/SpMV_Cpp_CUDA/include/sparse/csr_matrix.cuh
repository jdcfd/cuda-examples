#pragma once
#include <cstdint>

namespace sparse {

class CSRMatrix {
public:
    int nrows = 0;
    int ncols = 0;
    int nnz   = 0;

    // Host pointers
    int*    h_rows   = nullptr;
    int*    h_cols   = nullptr;
    double* h_values = nullptr;

    // Device pointers
    int*    d_rows   = nullptr;
    int*    d_cols   = nullptr;
    double* d_values = nullptr;

    CSRMatrix(int nr, int nc, int nonzeros);
    ~CSRMatrix();

    void update_host();
    void update_device();
    void print() const;
};

} // namespace sparse
