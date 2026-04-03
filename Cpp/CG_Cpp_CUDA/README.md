# Conjugate Gradient Solver (CUDA Engine)

This project contains a GPU-accelerated implementation of the **Conjugate Gradient (CG) Method** written in C++ and CUDA. The CG method is an iterative algorithm utilized for solving large, symmetric, and positive-definite systems of linear equations. 

This solver hinges directly on the sparse matrix-vector multiplication (SpMV) logic and data structures provided by the neighboring `../SpMV_Cpp_CUDA` project.

## Project Structure

*   `CMakeLists.txt`: Project definitions connecting seamlessly with SpMV via `add_subdirectory`.
*   `include/cg.cuh`: Function prototypes for the CG solver module.
*   `src/cg.cu`: Core solver implementation. By keeping data natively on GPUs with `sparse::DenseVector`, we achieve peak hardware acceleration wrapping device pointers using the [NVIDIA Thrust](https://developer.nvidia.com/thrust) library for vector-based mathematics (BLAS Level 1).
*   `examples/main.cu`: A standalone test mimicking `../SpMV_Cpp_CUDA` examples that reads MTX matrices.

## Build Instructions

Because this relies statically on `SpMV_Cpp_CUDA`, make sure the SpMV directory is sitting correctly adjacent to this project. We utilize `cmake` directly to organize our target compilations. 

```bash
# Navigate to the project root
cd Cpp/CG_Cpp_CUDA

# Configure the project 
cmake -B build -S .

# Build both the SpMV dependency library and the CG benchmarking executable
cmake --build build
```

## Running the Benchmark

You can test the conjugate gradient loop directly by feeding it symmetric sparse matrices (found in Matrix Market `.mtx` formatting under `../SpMV_Cpp_CUDA/data/`). Note that CG intrinsically requires symmetric positive definite (SPD) arrays.

```bash
./build/cg_benchmark ../SpMV_Cpp_CUDA/data/1138_bus.mtx
```

### Understanding the Benchmark Logic

The executable script creates an artificial "ground truth" response matrix mapped strictly to $x_{true} = 1.0$. It derives our testing $b$ values exactly as $b = A * x_{true}$. Once initialized, the solver evaluates standard progression mapping the residual decay back exactly matching $A x = b$. 

Output should follow formatting:
```text
nrows, ncols, nnz: 1138 1138 4054
Starting CG solver...
CG Converged in 1744 iterations.
Time elapsed: 201.921 ms
Max absolute error in solution: 0.00016059
```
