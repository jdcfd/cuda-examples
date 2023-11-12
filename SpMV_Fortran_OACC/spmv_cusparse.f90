program spmv_cusparse

    use cudafor
    use cusparse
    implicit none
    integer, parameter :: nrows = 4, ncols = 4, nnz = 9
    integer, dimension(nrows+1):: csrOffsets 
    integer, dimension(nnz) :: columns 
    real, dimension(ncols) :: values
    real, dimension(ncols) :: x, y
    real :: alpha = 1.0, beta = 0.0

    ! Indices from C example, shifted for 1-indexing
    csrOffsets = [0, 3, 4, 7, 9] + 1
    columns = [ 0, 2, 3, 1, 0, 2, 3, 1, 3 ] + 1 
    values = [1.0, 2.0, 3.0, 4.0, 5.0, &
              6.0, 7.0, 8.0, 9.0]
    x = [1.0,2.0,3.0,4.0]
    y = 0.0


    print *, "Compilation Successfull!!"

end program spmv_cusparse            