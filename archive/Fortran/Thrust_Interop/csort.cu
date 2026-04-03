#include <thrust/device_vector.h>
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
}