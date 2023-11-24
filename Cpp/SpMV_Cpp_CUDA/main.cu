#include <mmio_reader.cuh>

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
    print_mat_data(mymat);

    delete mymat; // save because pointing to nullptr

    return 0;
}