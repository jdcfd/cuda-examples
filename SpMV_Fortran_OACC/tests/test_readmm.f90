program main
    
  use matrix_utils

  integer, dimension(:), allocatable :: indx, jndx 
  real(8), dimension(:), allocatable :: rval 
  integer :: nnz, nrows, ncols
  character(len=128) :: filename
  character(len=19) :: symm

  ! write(filename,*) "../data/CurlCurl_4.mtx"
  write(filename,*) "../data/cage3.mtx"

  ! Read matrix in COO format and save as dense matrix
  call matrix_info(filename, nnz) 

  allocate(indx(nnz))
  allocate(jndx(nnz))
  allocate(rval(nnz))

  call matrix_readcoo(filename,nnz,nrows,ncols,symm,indx,jndx,rval)

  print *, "INDX"
  print '(1i4)', indx
  print *, "JNDX"
  print '(1i4)', jndx
  print *, "RVAL"
  print '(1d24.15)', rval
  print *, "Symm:", trim(symm)

  call matrix_coo2csr(nrows, ncols, nnz,&
                      indx, jndx, rvals,&
                      csrOffsets, csrCols, csrvals)

  deallocate(indx)
  deallocate(jndx)
  deallocate(rval)

end program


    !! Load Matrix in COO format
    
    !! convert to CSR format

    !! Perform matrix multiplication

    !! compare result using matmul

