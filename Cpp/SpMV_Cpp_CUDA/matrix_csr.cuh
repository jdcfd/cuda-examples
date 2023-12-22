// Inheritance is not necessary, just did it to learn OOP in c++
class CSRMatrix {
    public:
        int nrows;
        int ncols;
        int nnz;
        // Host Variables
        int * h_rows;
        int * h_cols;
        double * h_values;
        // Device Variables
        int * d_rows;
        int * d_cols;
        double * d_values;
    //---------------------------
        CSRMatrix(int nr, int nc, int nnz);
        ~CSRMatrix();     
        void alloc_mem();
        void free_mem();
        void update_host();
        void update_device();
        void print();
};
