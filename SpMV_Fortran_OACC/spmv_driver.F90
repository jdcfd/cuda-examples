program main
    
  use cutensorex
  use prec_const
  use matrix_utils

  real(rp), pointer, dimension(:,:) :: a, b, d

  ! Read matrix in COO format and save as dense matrix
  call load_matrix("somefile.txt", a) 

end program


    !! Load Matrix in COO format
    
    !! convert to CSR format

    !! Perform matrix multiplication

    !! compare result using matmul

