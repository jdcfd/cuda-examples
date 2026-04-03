module matrix_utils

    use thrust
    implicit none

    contains

    subroutine matrix_info(filename, nnz)
        ! Load matrix from *.mm file.
        ! Arguments:
        !   filename: Name of the file conatining matrix data in COO format
        !   a: Pointer to a rank 2 matrix which will be allocated and populated
        character(len=*), intent(in) :: filename
        character(len=10) :: rep 
        character(len=7)  :: field
        character(len=19) :: symm
        integer, intent(inout) :: nnz
        integer :: nrows, ncols
        integer :: iunit = 10
        integer :: ios

        open(unit=iunit,file=filename,status='OLD',iostat=ios)

        if(ios .ne. 0)then
            write(0,*) "Failed to open file with iostat=",ios
            close(iunit)
            error stop
        endif

        call mminfo(iunit,rep,field,symm,nrows,ncols,nnz)

        print *
        print *, "Matrix file:", filename
        print *, "-------------------------------------------"
        print *, "Rows:", nrows
        print *, "Columns:", ncols
        print *, "NNZ:", nnz
        print *, "Rep: ", rep
        print *, "Symm: ", symm
        print *, "Field: ", field
        print*

        close(iunit)

        if( trim(symm) == "symmetric" )then
            print *, " - Matrix is symmetric"
            nnz = 2 * nnz - nrows ! This assumes that diagonal values are all non-zero
            print *, " - NEW nnz:", nnz
        endif

        if(field(1:4) .ne. "real")then
            print *, "Types other than REAL are not supported. Exiting..."
            error stop
        endif      

        if( trim(rep) .ne. "coordinate" )then
            write(0,*) "Wrong type of representation in tile:", rep
            error stop
        endif       
        
        
    end subroutine matrix_info

    subroutine matrix_readcoo(filename,nnz,nrows,ncols,indx,jndx,rval)
        character(len=*), intent(in) :: filename
        integer, intent(inout) :: nrows, ncols
        integer, intent(in) :: nnz
        integer, intent(inout) :: indx(*), jndx(*)
        real(8), intent(inout) :: rval(*)
        integer, dimension(:), allocatable :: indx_symm, jndx_symm
        real(8), dimension(:), allocatable :: rval_symm
        integer, dimension(nnz) :: ival
        complex, dimension(nnz) :: cval
        character(len=19) :: symm
        character(len=10) :: rep 
        character(len=7)  :: field
        
        integer :: nnz_n
        integer :: iunit = 10
        integer :: i, j, k, n
        real(8) :: val
        integer :: ios

        open(unit=iunit,file=filename,status='OLD',iostat=ios)

        if(ios .ne. 0)then
            write(0,*) "Failed to open file with iostat=",ios
            close(iunit)
            error stop
        endif
        
        call mmread(iunit,rep,field,symm,nrows,ncols,nnz_n,nnz,&
                                        indx,jndx,ival,rval,cval)

        close(iunit)

        if( nnz < nnz_n )then
            write(0,*) "Wrong number of non-zero values."
            error stop
        endif

        if( trim(rep) .ne. "coordinate" )then
            write(0,*) "Wrong type of representation in tile."
            error stop
        endif

        if(trim(symm) == "symmetric")then
            if( (nnz_n*2-nrows) .ne. nnz )then
                write(0,*) "Wrong number of non-zero values."
                error stop
            endif

            allocate(indx_symm(nnz_n))
            allocate(jndx_symm(nnz_n))
            allocate(rval_symm(nnz_n))

            indx_symm = indx(1:nnz_n)
            jndx_symm = jndx(1:nnz_n)
            rval_symm = rval(1:nnz_n)

            k=1
            do n=1,nnz_n
                indx(k) = indx_symm(n)
                jndx(k) = jndx_symm(n)
                rval(k) = rval_symm(n)
                k = k+1
                if( indx_symm(n) .ne. jndx_symm(n) )then
                    indx(k) = jndx_symm(n)
                    jndx(k) = indx_symm(n)
                    rval(k) = rval_symm(n)
                    k = k+1
                endif
            enddo

            deallocate(indx_symm)
            deallocate(jndx_symm)
            deallocate(rval_symm)

            ! Using GPU to sort, which may be inneficient for small matrices

            ! !$acc data copy(indx, jndx, rval) 
            ! !$acc host_data use_device(indx, jndx, rval)

            call thrust_sort_by_tuple(indx,jndx,rval,nnz)

            ! !$acc end host_data
            ! !$acc end data

        endif

    end subroutine matrix_readcoo
! !-----------------------------------------------------------------------
!     subroutine matrix_coo2csr(nrows, ncols, nnz,&
!                             cooindx, coojndx, coovals,&
!                             csrOffsets, csrCols, csrvals)
!         integer, intent(in) :: nrows, ncols, nnz
!         !---- COO format, assume full matirix info, not symm
!         integer, dimension(nnz), intent(in) :: cooindx, coojndx 
!         real, dimension(nnz), intent(in) :: coovals
!         !---- CSR format
!         integer, dimension(nrows+1), intent(out):: csrOffsets 
!         integer, dimension(nnz), intent(out) :: csrCols
!         real, dimension(nnz), intent(out) :: csrvals

!         ! Use thrust library to sort indices 


!     end subroutine matrix_coo2csr

end module matrix_utils