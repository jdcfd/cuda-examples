program main
    
  use matrix_utils
  
  implicit none

  integer, parameter :: rk = 8 ! Double
  integer, dimension(:), allocatable :: indx, jndx 
  real(rk), dimension(:), allocatable :: cooval 
  integer :: nnz, ncols, nrows
  character(len=128) :: filename
  !---- CSR format
  integer, dimension(:), allocatable :: csrOffsets 
  integer, dimension(:), allocatable :: csrCols
  real(rk), dimension(:), allocatable :: csrvals
  !
  integer, dimension(:), allocatable :: csrOffsets_res 
  integer, dimension(:), allocatable :: csrCols_res
  real(rk), dimension(:), allocatable :: csrvals_res

  real(rk), dimension(:,:), allocatable :: test_matrix

  integer :: i

  write(filename,*) "data/cage3_symm.mtx"

  ! Read matrix in COO format and save as dense matrix
  call matrix_info(filename, nnz) 

  allocate(indx(nnz))
  allocate(jndx(nnz))
  allocate(cooval(nnz))

  call matrix_readcoo(filename,nnz,nrows,ncols,indx,jndx,cooval)

  allocate(test_matrix(nrows,ncols))

  print '(3i4,d24.15)', (i, indx(i), jndx(i), cooval(i), i=1,nnz)
  print *

  test_matrix = 0.0
  do i=1,nnz
    test_matrix(indx(i),jndx(i)) = cooval(i)
  enddo

  print '(5d24.15)', transpose(test_matrix)

  deallocate(indx)
  deallocate(jndx)
  deallocate(cooval)

end program


    !! Load Matrix in COO format
    
    !! convert to CSR format

    !! Perform matrix multiplication

    !! compare result using matmul

