#include <array>
#include <iostream>
#include <sycl/sycl.hpp>

class matrix_multiply_gpu;

int main() {
    const int N = 3;  // Size of the matrix NxN
    std::array<int, N * N> matrixA = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::array<int, N * N> matrixB = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    std::array<int, N * N> matrixR = {0};  // Result matrix initialized to zero

    // Check for available GPU devices
    auto gpu_selector = sycl::gpu_selector_v;
    try {
        // Create a queue using the GPU selector
        sycl::queue gpuQueue{gpu_selector};

        {
            // Buffers for matrices A, B, and R
            sycl::buffer<int, 2> bufA(matrixA.data(), sycl::range<2>{N, N});
            sycl::buffer<int, 2> bufB(matrixB.data(), sycl::range<2>{N, N});
            sycl::buffer<int, 2> bufR(matrixR.data(), sycl::range<2>{N, N});

            // Submit a command group to the queue for execution
            gpuQueue
                .submit([&](sycl::handler& cgh) {
                    // Accessors for buffers A, B, and R
                    auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
                    auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
                    auto accR = bufR.get_access<sycl::access::mode::write>(cgh);

                    // Implement the parallel_for with the right dimensions and
                    // index
                    cgh.parallel_for<matrix_multiply_gpu>(
                        sycl::range<2>{N, N}, [=](sycl::id<2> idx) {
                            int row = idx[0];
                            int col = idx[1];
                            for (int k = 0; k < N; k++) {
                                accR[row][col] += accA[row][k] * accB[k][col];
                            }
                        });
                })
                .wait();

            std::cout
                << "Matrix multiplication completed successfully on GPU.\n";
        }

    } catch (sycl::exception const& e) {
        std::cerr << "Failed to execute on GPU. Error: " << e.what() << '\n';
    }

    // Print the result matrix
    std::cout << "Result matrix:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << matrixR[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
