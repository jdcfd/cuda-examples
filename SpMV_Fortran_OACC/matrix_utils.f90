module matrix_utils

    use prec_const

    implicit none

    contains

    subroutine load_matrix(filename, a)
        ! Load matrix from *.mm file.
        ! Arguments:
        !   filename: Name of the file conatining matrix data in COO format
        !   a: Pointer to a rank 2 matrix which will be allocated and populated
        character(len=*) :: filename
        real(rp), dimension(:,:), pointer :: a
        integer :: iunit = 10
        integer :: ios

        if(associated(a))then
            write(*,*) "Passed argument pointer A is associated. Exiting..."
            error stop
        endif

        open(unit=iunit,file=filename,status='OLD',iostat=ios)
        if(ios .ne. 0)then
            write(0,*) "Failed to open file with iostat=",ios
            error stop
        endif
        
    end subroutine

end module matrix_utils