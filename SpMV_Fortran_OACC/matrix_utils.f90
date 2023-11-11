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
        
    end subroutine

end module matrix_utils