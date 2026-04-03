module cusparse_routines

    use cudafor
    use cusparse
    use iso_c_binding

    contains 

    subroutine cusparse_spmv(nrows, ncols, nnz, csrOffsets, columns, values, x, y, alpha, beta, time_setup, time_spmv)
        implicit none
        integer, intent(in) :: nrows, ncols, nnz
        integer, dimension(nrows+1), intent(in):: csrOffsets 
        integer, dimension(nnz), intent(in) :: columns 
        real(8), dimension(nnz), intent(in) :: values
        real(8), dimension(ncols), intent(in) :: x
        real(8), dimension(ncols), intent(inout) :: y
        real(8), intent(in) :: alpha, beta
        real, intent(out) :: time_setup, time_spmv
        !---------------------------------------
        integer :: status
        real :: t0, t1
        !---------------------------------------
        ! Cusparse variables
        type(cusparseHandle) :: handle
        type(cusparseSpMatDescr) :: matA
        type(cusparseDnVecDescr) :: vecX, vecY
        integer(1), pointer, dimension(:) :: buffer => null()
        integer(c_size_t) :: bufferSize = 0

        !$acc data pcopyin(csrOffsets, columns, values, x) pcopy(y)
        !$acc host_data use_device(csrOffsets, columns, values, x, y)

        ! initalize CUSPARSE and matrix descriptor
        call cpu_time(t0)
        status = cusparseCreate(handle)
        if(status/=CUSPARSE_STATUS_SUCCESS) print *, 'cusparseCreate error: ', status

        status = cusparseCreateCsr(matA, nrows, ncols, nnz, &
                                csrOffsets, columns, values, &
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, &
                                CUSPARSE_INDEX_BASE_ONE, CUDA_R_64F)
        if(status/=CUSPARSE_STATUS_SUCCESS) print *, 'cusparseCreateCsr error: ', status

        status = cusparseCreateDnVec(vecX, ncols, x, CUDA_R_64F)
        if(status/=CUSPARSE_STATUS_SUCCESS) print *, 'cusparseCreateDnVec for vecX error: ', status
        status = cusparseCreateDnVec(vecY, ncols, y, CUDA_R_64F)
        if(status/=CUSPARSE_STATUS_SUCCESS) print *, 'cusparseCreateDnVec for vecY error: ', status

        status = cusparseSpMV_bufferSize( handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &
                                    alpha, matA, vecX, beta, vecY, CUDA_R_64F, &
                                    CUSPARSE_SPMV_ALG_DEFAULT, bufferSize)
        if(status/=CUSPARSE_STATUS_SUCCESS) print *, 'cusparseSpMV_bufferSize error: ', status
        
        call cpu_time(t1)
        time_setup = t1-t0

        !$acc end host_data         
        
        allocate(buffer(bufferSize))
        !$acc data create(buffer)
        !$acc host_data use_device(buffer)

        call cpu_time(t0)
        status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,&
                            alpha, matA, vecX, beta, vecY, CUDA_R_64F,&
                            CUSPARSE_SPMV_ALG_DEFAULT, buffer)
        if(status/=CUSPARSE_STATUS_SUCCESS) print *, 'cusparseSpMV error: ', status
        call cpu_time(t1)
        time_spmv = t1-t0

        status=cudaDeviceSynchronize

        ! !! destroy matrix/vector descriptors
        status = cusparseDestroySpMat(matA) 
        status = cusparseDestroyDnVec(vecX) 
        status = cusparseDestroyDnVec(vecY) 
        status = cusparseDestroy(handle) 

        !$acc end host_data
        !$acc end data
        deallocate(buffer)
        !$acc end data
    
    end subroutine cusparse_spmv

end module cusparse_routines         