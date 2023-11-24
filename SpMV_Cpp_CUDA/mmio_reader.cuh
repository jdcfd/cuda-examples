#include <iostream>
#include "stdio.h"
extern "C"{
    #include "mmio.h"
}
#include <matrix_csr.cuh>

using namespace std;

class CSRMatrixReader {
    FILE* f;
    MM_typecode mmtc;
    int nnz;
    bool symm;
    public:
    string filename;
    CSRMatrixReader(string filename);
    ~CSRMatrixReader();
    CSRMatrix* mm_init_csr();
    int mm_read_csr(CSRMatrix* mat);
};