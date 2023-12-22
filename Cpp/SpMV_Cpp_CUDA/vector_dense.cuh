
class DenseVector {
    public:
        int size;
        double * h_val;
        double * d_val;
        DenseVector(int n);
        ~DenseVector();
        void generate();
        void print();
        void update_host();
        void update_device();
        void fill(double v);
        // void operator=(const DenseVector& dv);
};