#include <mmio_reader.cuh>

using namespace std;

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

    mymat = reader.read_mm_csr();

    if( mymat != nullptr ){
        cout << "Nrows: " << mymat->nrows << " Ncols: " << mymat->ncols << endl;
        cout << "Nnz: " << mymat->nnz << endl;
        for(int i {}; i < mymat->nnz; i++){
            cout << "val[" << i << "]=" << mymat->values[i] << endl;
        }
    }

    delete mymat; // save because pointing to nullptr

    return 0;
}