#pragma once

namespace sparse {

class DenseVector {
public:
    int size;
    double * h_val = nullptr;
    double * d_val = nullptr;

    explicit DenseVector(int n);
    ~DenseVector();

    void generate();
    void print() const;
    void update_host();
    void update_device();
    void fill(double v);
};

} // namespace sparse
