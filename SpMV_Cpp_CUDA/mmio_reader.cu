#include <mmio_reader.cuh>

CSRMatrixReader::CSRMatrixReader(string filename){
    this->filename=filename;
    this->f= fopen(filename.c_str(),"r");
}

CSRMatrixReader::~CSRMatrixReader(){
    fclose(this->f);
}