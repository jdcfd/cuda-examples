#pragma once

namespace sparse {

template <typename T>
class DenseVectorT {
public:
    int size;
    T* h_val = nullptr;
    T* d_val = nullptr;

    explicit DenseVectorT(int n);
    ~DenseVectorT();

    void generate();
    void print() const;
    void update_host();
    void update_device();
    void fill(T v);
};

using DenseVector  = DenseVectorT<double>;
using DenseVectorF = DenseVectorT<float>;

} // namespace sparse
