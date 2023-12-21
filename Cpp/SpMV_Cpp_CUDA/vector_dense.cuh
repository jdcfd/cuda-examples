
class DenseVector {
    public:
        int size;
        double * val;
        DenseVector(int n);
        ~DenseVector();
        void generate();
        void print();
        void fill(double v);
        // void operator=(const DenseVector& dv);
};