#include <matrix_csr.cuh>
#include <mmio_reader.cuh>
#include <thrust/sort.h>

CSRMatrixReader::CSRMatrixReader(string filename){
    this->filename=filename;
    this->f= fopen(filename.c_str(),"r");
}

CSRMatrixReader::~CSRMatrixReader(){
    fclose(this->f);
}

int CSRMatrixReader::mm_init_csr(CSRMatrix **mat){
    int ierr;
    ierr = mm_read_banner(this->f,&(this->mmtc));
    if(ierr){
        cout << "Error reading Banner!" << endl;
        return -1;
    }
    else{
        cout << mm_typecode_to_str(this->mmtc) << endl;
    }
    if(!(mm_is_sparse(this->mmtc))){
        throw std::invalid_argument("Non-sparse matrices not supported!");
    }
    int nrows, ncols, nnz;

    ierr = mm_read_mtx_crd_size(this->f, &nrows, &ncols, &nnz);

    this->nnz = nnz;

    if( mm_is_symmetric(this->mmtc) ){
        nnz = nnz * 2 - nrows; // Assumes diagonals are non-zero
        this->symm = true;
    }

    if(ierr){
        return ierr;
    }

    if(*mat == nullptr){
        *mat = new CSRMatrix(nrows, ncols, nnz);
    }
    else{
        cout << "Matrix is not empty!" << endl;
        ierr = -1;
    }
    
    return ierr;
}

int CSRMatrixReader::mm_read_csr(CSRMatrix *mat){
    
    int *indx = new int[mat->nnz]; // Temporary rows array
    int ierr {};
    int i {};
    ierr = mm_read_mtx_crd_data(this->f, mat->nrows, mat->ncols, this->nnz, 
                                indx, mat->h_cols, mat->h_values, this->mmtc);

    for(i=0;i < this->nnz; i++){
        // Subtract 1 for 0-indexing
        indx[i]--;
        mat->h_cols[i]--;
    }

    if(ierr){
        cout << "There was an error while reading matrix data" << endl;
        return ierr;
    }
    
    if(this->symm){
        int k {this->nnz};
        for(int i=0; i < this->nnz; i++){
            if( indx[i] != mat->h_cols[i] ){
                indx[k] = mat->h_cols[i];
                mat->h_cols[k] = indx[i];
                mat->h_values[k] = mat->h_values[i];
                k++;
            }
        }
        if(k != mat->nnz){
            cout << "Something went wrong with settign symmetric matrix" << endl;
            ierr = 1;
        }  
    }

    int &nnz = mat->nnz;
    int &m = mat->nrows;

    /* sort the coo matrix */
    auto begin_keys =
      thrust::make_zip_iterator(thrust::make_tuple( &indx[0], mat->h_cols ));
    auto end_keys =
      thrust::make_zip_iterator(thrust::make_tuple( &indx[0] + nnz, mat->h_cols + nnz));

    thrust::stable_sort_by_key(thrust::host, begin_keys, end_keys, mat->h_values,
                      thrust::less<thrust::tuple<int, int>>());

    /* fill the row counts */
    thrust::fill(thrust::host, mat->h_rows, mat->h_rows + m + 1, 0);
    for (int nz = 0; nz < nnz; ++nz)
      mat->h_rows[indx[nz]]++;

    /* calculate max_nnz_per_row */
    int max_nnz_per_row=0;
    for (int i = 0; i < m; ++i)
      if (mat->h_rows[i] > max_nnz_per_row)
        max_nnz_per_row = mat->h_rows[i];

    /* transform the row counts to row offsets */
    thrust::exclusive_scan(mat->h_rows, mat->h_rows + m + 1, mat->h_rows);

    mat->update_device();
    
    delete indx;
    return max_nnz_per_row;
    
}