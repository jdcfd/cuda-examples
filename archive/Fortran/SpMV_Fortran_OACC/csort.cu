#include <thrust/device_vector.h>
#include <thrust/sort.h>

extern "C" {
    void sort_int_wrapper( int *data, int N)
    {
        thrust::device_ptr<int> dev_ptr(data);
        thrust::sort(dev_ptr, dev_ptr+N);
    }
    void sort_float_wrapper( float *data, int N)
    {
        thrust::device_ptr<float> dev_ptr(data);
        thrust::sort(dev_ptr, dev_ptr+N);
    }
    void sort_double_wrapper( double *data, int N)
    {
        thrust::device_ptr<double> dev_ptr(data);
        thrust::sort(dev_ptr, dev_ptr+N);
    }
    void sort_by_tuple_wrapper( int *indx, int *jndx, double *rval, int N)
    {
        // wrap data from fortran into thrust vectors
        thrust::device_ptr<double> dev_rval(rval);
        thrust::device_ptr<int> dev_indx(indx);
        thrust::device_ptr<int> dev_jndx(jndx);
        /* sort the coo matrix */
        auto begin_keys =
            thrust::make_zip_iterator(thrust::make_tuple(dev_indx,dev_jndx));
        auto end_keys =
            thrust::make_zip_iterator(thrust::make_tuple(dev_indx + N, dev_jndx + N));

        thrust::stable_sort_by_key(begin_keys, end_keys, dev_rval,
                                    thrust::less<thrust::tuple<int, int>>());
    }
    void sort_by_tuple_host_wrapper( int *indx, int *jndx, double *rval, int N)
    {
        // wrap data from fortran into thrust vectors
        /* sort the coo matrix */
        std::cout << "Running cpu version. N = " << N << std::endl;
        auto begin_keys =
            thrust::make_zip_iterator(thrust::make_tuple(indx    ,jndx    ));
        auto end_keys =
            thrust::make_zip_iterator(thrust::make_tuple(indx + N,jndx + N));

        thrust::stable_sort_by_key(thrust::host, begin_keys, end_keys, rval,
                                    thrust::less<thrust::tuple<int, int>>());
    }
}