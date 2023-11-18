module matrix_utils

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

        if(field(1:4) .ne. "real")then
            print *, "Types other than REAL are not supported. Exiting..."
            close(iunit)
            error stop
        endif      
        
        close(iunit)
        
    end subroutine matrix_info

    subroutine matrix_readcoo(filename,nnz,nrows,ncols,symm,indx,jndx,rval)
        character(len=*), intent(in) :: filename
        integer, intent(inout) :: nrows, ncols
        integer, intent(in) :: nnz
        integer, dimension(nnz), intent(inout) :: indx, jndx
        character(len=*), intent(inout) :: symm
        integer, dimension(nnz) :: ival
        real(8), dimension(nnz), intent(inout) :: rval
        complex, dimension(nnz) :: cval
        character(len=10) :: rep 
        character(len=7)  :: field
        
        integer :: nnz_n
        integer :: iunit = 10
        integer :: ios

        open(unit=iunit,file=filename,status='OLD',iostat=ios)

        if(ios .ne. 0)then
            write(0,*) "Failed to open file with iostat=",ios
            close(iunit)
            error stop
        endif
        
        call mmread(iunit,rep,field,symm,nrows,ncols,nnz_n,nnz*2,&
                                        indx,jndx,ival,rval,cval)

        if( nnz .ne. nnz_n )then
            write(0,*) "Number of non zeros does not match input nnz."
            close(iunit)
            error stop
        endif

        close(iunit)

    end subroutine matrix_readcoo
!-----------------------------------------------------------------------
    subroutine matrix_coo2csr(nrows, ncols, nnz,&
                            cooindx, coojndx, coovals,&
                            csrOffsets, csrCols, csrvals)
        integer, intent(in) :: nrows, ncols, nnz
        !---- COO format, assume full matirix info, not symm
        integer, dimension(nnz), intent(in) :: cooindx, coojndx 
        real, dimension(nnz), intent(in) :: coovals
        !---- CSR format
        integer, dimension(nrows+1), intent(out):: csrOffsets 
        integer, dimension(nnz), intent(out) :: csrCols
        real, dimension(nnz), intent(out) :: csrvals

        ! Use thrust library to sort indices 


    end subroutine matrix_coo2csr

end module matrix_utils