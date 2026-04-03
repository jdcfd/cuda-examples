program main

    use thrust
    implicit none 

    real, allocatable :: vals(:)
    integer :: N=10
    integer :: i

    allocate(vals(N))

    call random_number(vals)

    vals(5)=100.

    print *,"Before sorting", vals
    
    !$acc data copy(vals)

    !$acc host_data use_device(vals)
    call thrustsort(vals,N)    
    !$acc end host_data

    !$acc end data

    print *,"After sorting", vals

end program