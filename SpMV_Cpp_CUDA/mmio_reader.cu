#include <mmio_reader.cuh>

CSRMatrixReader::CSRMatrixReader(string filename){
    this->filename=filename;
    this->f= fopen(filename.c_str(),"r");
}

CSRMatrixReader::~CSRMatrixReader(){
    fclose(this->f);
}

CSRMatrix* CSRMatrixReader::read_mm_csr(){
    int ierr;
    MM_typecode mmtc;
    ierr = mm_read_banner(this->f,&mmtc);
    if(ierr){
        cout << "Error reading Banner!" << endl;
        return nullptr;
    }
    else{
        cout << mm_typecode_to_str(mmtc) << endl;
    }
    if(!(mm_is_sparse(mmtc))){
        throw std::invalid_argument("Non-sparse matrices not supported!");
    }
    int M, N, nz;
    ierr = mm_read_mtx_crd_size(this->f, &M, &N, &nz);
    return new CSRMatrix(M,N,nz);
}