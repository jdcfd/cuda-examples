#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "dot.cuh"
#include <vector>
#include <numeric>

TEST_CASE("Dot product calculation is correct", "[dot]") {
    const int N = 10000;
    std::vector<double> h_x(N, 1.0);
    std::vector<double> h_y(N, 2.0);
    double h_result = 0.0;

    double *d_x, *d_y, *d_result;
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_y, N * sizeof(double));
    cudaMalloc(&d_result, sizeof(double));

    cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(double));

    SECTION("Standard dot product") {
        dot_product(d_x, d_y, d_result, N);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

        double expected = 20000.0; // 10000 * (1.0 * 2.0)
        REQUIRE_THAT(h_result, Catch::Matchers::WithinRel(expected, 1e-10));
    }

    SECTION("Dot product with different values") {
        for (int i = 0; i < N; ++i) {
            h_x[i] = static_cast<double>(i);
            h_y[i] = 1.0;
        }
        cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y.data(), N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_result, 0, sizeof(double));

        dot_product(d_x, d_y, d_result, N);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

        double expected = static_cast<double>(N - 1) * N / 2.0; // Sum of 0 to N-1
        REQUIRE_THAT(h_result, Catch::Matchers::WithinRel(expected, 1e-10));
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
}
