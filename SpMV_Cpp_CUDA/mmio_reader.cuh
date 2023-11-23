#include <iostream>
#include "stdio.h"
extern "C"{
    #include "mmio.h"
}
#include <matrix_csr.cuh>

using namespace std;

class CSRMatrixReader {
    FILE* f;
    public:
    string filename;
    CSRMatrixReader(string filename);
    ~CSRMatrixReader();
    FILE *get_fio(){ return f; }    
};