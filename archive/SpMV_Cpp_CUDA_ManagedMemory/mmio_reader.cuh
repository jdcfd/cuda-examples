#include <iostream>
#include "stdio.h"
extern "C"{
    #include "mmio.h"
}
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
    int mm_init_csr(CSRMatrix** mat);
    int mm_read_csr(CSRMatrix* mat);
};