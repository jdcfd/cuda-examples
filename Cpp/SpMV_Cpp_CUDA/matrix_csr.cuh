using namespace std;

// Inheritance is not necessary, just did it to learn OOP in c++
class CSRMatrix : public MatrixBase {
    public:
    int nnz {};
    // Host Variables
    int * rows {};
    int * cols {};
    double * values {};
    // Device Variables
    int * d_rows {};
    int * d_cols {};
    double * d_values {};
    CSRMatrix(int nr, int nc, int nnz);
    ~CSRMatrix();
    void update_device();
    void update_host();
    void print();
};
