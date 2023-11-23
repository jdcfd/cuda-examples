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

    MM_typecode mmtc;

    ierr = mm_read_banner(reader.get_fio(),&mmtc);
    if(ierr){
        cout << "Error reading banner!" << endl;
        return 1;
    }

    cout << "Banner: " << mm_typecode_to_str(mmtc) << endl;

    if( mymat ){
        cout << "Nrows: " << mymat->nrows << " Ncols: " << mymat->ncols << endl;
        cout << "Nnz: " << mymat->nnz << endl;
    }

    delete mymat; // save because pointing to nullptr

    return 0;
}