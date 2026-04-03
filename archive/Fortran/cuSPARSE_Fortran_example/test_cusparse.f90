program test_cusparse

    use cusparse_routines

    integer, parameter :: rk = 4
    integer, parameter :: nrows=4, ncols=4, nnz=9
    integer, dimension(nrows+1):: csrOffsets 
    integer, dimension(nnz) :: columns 
    real(rk), dimension(nnz) :: values
    real(rk), dimension(ncols) :: x
    real(rk), dimension(ncols) :: y, y_result
    real(rk) :: alpha=1.0, beta=0.0
    real :: time_setup=0.0, time_spmv=0.0

    csrOffsets = [ 0, 3, 4, 7, 9 ] + 1 ! Change to 1-indexing
    columns    = [ 0, 2, 3, 1, 0, 2, 3, 1, 3 ] + 1 ! Change to 1-indexing
    values     = [ 1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0, &
                   6.0d0, 7.0d0, 8.0d0, 9.0d0 ]
    X        = [  1.0d0, 2.0d0,  3.0d0,  4.0d0 ]
    Y        = [  0.0d0, 0.0d0,  0.0d0,  0.0d0 ]
    Y_result = [ 19.0d0, 8.0d0, 51.0d0, 52.0d0 ]

    call cusparse_mv(nrows, ncols, nnz, csrOffsets, columns, values, &
                       x, y, alpha, beta, time_setup, time_spmv)

    if( all( abs(y - y_result ) < 1e-15) )then
        print *, "TEST CUSPARSE ROUTINES SUCCESSFUL!"
        print *, "Y        =", y
        print *, "Y_result =", y_result
    else
        print *, "TEST CUSPARSE ROUTINES FAILED!"
        print *, "Y        =", y
        print *, "Y_result =", y_result
        stop
    endif

    print *
    print *, "Setup time:", time_setup
    print *, "Solution time:", time_spmv

end program