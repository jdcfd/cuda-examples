
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
Array Size (N): 10000001
Runs per benchmark: 10
Expected Dot Product: 3e+07
========================================
Custom Dot Product Result: 3e+07
Custom Mean Time: 0.389757 ms
----------------------------------------
Thrust Dot Product Result: 3e+07
Thrust Mean Time: 0.422432 ms
----------------------------------------
cuBLAS Dot Product Result: 3e+07
cuBLAS Mean Time: 0.399037 ms
========================================
``` 