/*
Author: Juan D. Colmenares F.
User  : jdcfd@github.com

Sparse Matrix-Vector multiplication in CUDA

Reads in Sparse matrix in MatrixMarket COO format and multiplies
it by a dense vector with random values.

*/

#include <mmio_reader.cuh>
#include <vector_dense.cuh>

using namespace std;

void print_mat_data(CSRMatrix * mat){
    if(mat){
        for(int i {}; i < mat->nrows + 1; i++){
            cout << "rows[" << i << "] = " << mat->rows[i] << endl;
        }
        for(int i {}; i < mat->nnz; i++){
            cout << "cols[" << i << "]= " << mat->cols[i] << ", val[" << i << "]= " << mat->values[i] << endl;
        }
    }else{
        cout << "Matrix has not been set." << endl;
    }
}

void print_vec_data(DenseVector * v){
    for(int i {}; i < v->size; i++){
        cout << "v[" << i << "] = " << v->h_v[i] << endl;
    }
}

int main(int argc, char const *argv[]) {

    if( argc < 3 ){
        cout << "Usage: ./vector_csr <matrix market file> <ntrials>" << endl;
        return 1;
    }

    int ierr {};

    string filename {string(argv[1])};

    int ntrials {atoi(argv[2])};

    CSRMatrix *mymat {}; //nullptr

    CSRMatrixReader reader(filename);

    mymat = reader.mm_init_csr(); // allocate memory

    ierr = reader.mm_read_csr(mymat); //read from file and convert from coo to csr

    if( mymat ){
        cout << "Nrows: " << mymat->nrows << " Ncols: " << mymat->ncols << endl;
        cout << "Nnz: " << mymat->nnz << endl;
    }
    mymat->update_host();

    // print_mat_data(mymat); // Print all values. Commented out for large matrices.

    DenseVector X(mymat->ncols);

    X.generate(); // Fill with random numbers 

    DenseVector Y(mymat->ncols); // Initialize with zeros

    print_vec_data(&X);
    print_vec_data(&Y);

    delete mymat; // Calls destroyer
    mymat = nullptr; 

    return ierr;
}