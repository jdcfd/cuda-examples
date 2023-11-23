#include <matrix.hpp>

using namespace std;

// Inheritance is not necessary, just did it to learn how to use it.
class CSRMatrix : public MatrixBase {
    public:
    int nnz {};
    int * rows {};
    int * cols {};
    double * values {};
    CSRMatrix(int nr, int nc, int nnz);
    ~CSRMatrix();
};
