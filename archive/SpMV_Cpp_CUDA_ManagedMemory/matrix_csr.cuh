using namespace std;

// Inheritance is not necessary, just did it to learn OOP in c++
class CSRMatrix {
    public:
        int nrows;
        int ncols;
        int nnz;
        // Device Variables
        int * rows;
        int * cols;
        double * values;
    //---------------------------
        CSRMatrix(int nr, int nc, int nnz);
        ~CSRMatrix();     
        void alloc_mem();
        void free_mem();
        void print();
};
