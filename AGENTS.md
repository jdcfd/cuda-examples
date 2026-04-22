# AI Agent Instructions

To ensure consistency in this project, please follow these guidelines:

## General
- Always document your changes in the relevant `README.md` files.
- If you make changes that break the code, provide a fix for it.

## Python
- Use `pixi` for managing Python environments, running scripts, and adding dependencies.
- Always create a `tests/` directory to house your tests, unless there is already one.

## C++ / CUDA
- Use `cmake` for building and managing these projects.
- The project is consolidated into a single CMake project in the `Cpp/` directory.
- Build the entire suite from `Cpp/` using:
  ```bash
  cmake -B build -S .
  cmake --build build -j
  ```
- **Testing with Catch2**:
  - Always create a `tests/` directory within each component (e.g., `Cpp/Dot_Cpp_CUDA/tests/`).
  - The `tests/CMakeLists.txt` should look like this:
    ```cmake
    add_executable(test_component test_file.cu)
    target_link_libraries(test_component PRIVATE component_lib Catch2::Catch2WithMain)
    include(Catch)
    catch_discover_tests(test_component)
    ```
  - **Running Tests**:
    - **Individual**: Run the binary directly, e.g., `./build/Dot_Cpp_CUDA/tests/test_dot`.
    - **All Tests**: Run `ctest` from the `build/` directory.
- **DO NOT USE `pixi` for C++/CUDA**

## Documentation
- Always update the `README.md` files when making significant changes to the code.
