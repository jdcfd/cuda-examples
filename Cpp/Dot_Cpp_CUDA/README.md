## Single-precision API

`dot_product` is overloaded for `double*` and `float*` device arrays. The output (`result`) must be zeroed on the device before the call, as in the double-precision path.

## Build and Run

```bash
cmake -B build -S . && cmake --build build && ./build/dot_example
```

## Profile

To profile custom kernel using Nsight Compute:

```bash
cmake --build build --target profile
```

# Expected output

```
$ ./build/dot_example
Precision: single (float)
Array Size (N): 10000001
Runs per benchmark: 10
Expected Dot Product: 3e+07
========================================
Custom Dot Product Result: 3e+07
Custom Mean Time: 0.2xx ms
----------------------------------------
Thrust Dot Product Result: 3e+07
Thrust Mean Time: 0.2xx ms
----------------------------------------
cuBLAS Dot Product Result: 3e+07
cuBLAS Mean Time: 0.2xx ms
========================================

Precision: double
Array Size (N): 10000001
Runs per benchmark: 10
Expected Dot Product: 3e+07
========================================
Custom Dot Product Result: 3e+07
Custom Mean Time: 0.3xx ms
----------------------------------------
Thrust Dot Product Result: 3e+07
Thrust Mean Time: 0.4xx ms
----------------------------------------
cuBLAS Dot Product Result: 3e+07
cuBLAS Mean Time: 0.3xx ms
========================================
``` 
