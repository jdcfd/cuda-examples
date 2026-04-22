cmake_minimum_required(VERSION 3.18)
project(LinearSolverSuite LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86) # Keep from user's current settings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0 # Or latest stable
)

FetchContent_MakeAvailable(Catch2)

include(CTest)

# Add Common module
add_subdirectory(Common)

# Add the solver modules
add_subdirectory(Dot_Cpp_CUDA)
add_subdirectory(SpMV_Cpp_CUDA)
add_subdirectory(CG_Cpp_CUDA)
