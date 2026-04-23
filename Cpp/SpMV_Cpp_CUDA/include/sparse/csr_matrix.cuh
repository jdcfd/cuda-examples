#pragma once
#include <cstdint>

namespace sparse {

template <typename T>
class CSRMatrixT {
public:
    int nrows = 0;
    int ncols = 0;
    int nnz   = 0;

    int*    h_rows   = nullptr;
    int*    h_cols   = nullptr;
    T*      h_values = nullptr;

    int*    d_rows   = nullptr;
    int*    d_cols   = nullptr;
    T*      d_values = nullptr;

    CSRMatrixT(int nr, int nc, int nonzeros);
    ~CSRMatrixT();

    void update_host();
    void update_device();
    void print() const;
};

using CSRMatrix  = CSRMatrixT<double>;
using CSRMatrixF = CSRMatrixT<float>;

} // namespace sparse
