#ifndef REAL
#define REAL float
#endif

#if REAL==float
    #define CUSPARSE_REAL CUDA_R_32F
#elif REAL==double
    #define CUSPARSE_REAL CUDA_R_64F
#endif