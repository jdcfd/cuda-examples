program main
    
  use cutensorex
  use matrix_utils

  integer, dimension(:), allocatable :: indx, jndx 
  real, dimension(:), allocatable :: rval 
  integer :: nnz
  character(len=128) :: filename

  write(filename,*) "data/CurlCurl_4.mtx"

  ! Read matrix in COO format and save as dense matrix
  call matrix_info(filename, nnz) 

  allocate(indx(nnz))
  allocate(jndx(nnz))
  allocate(rval(nnz))


  deallocate(indx)
  deallocate(jndx)
  deallocate(rval)

end program


    !! Load Matrix in COO format
    
    !! convert to CSR format

    !! Perform matrix multiplication

    !! compare result using matmul

