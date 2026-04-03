Fortran implementation of sparse matrix dense vector multiplication using OpenACC. The idea is to avoid using CUDA Fortran, where possible. This is an exercise to test if using OpenACC alone can be performant compared to an optimized library.

- cuSPARSE is used to validate results and compare performance.

- Thrust C++ library is used for sorting data to convert COO matrix to CSR.

TODO:
- implement sparse matrix multiplication using OpenACC.


