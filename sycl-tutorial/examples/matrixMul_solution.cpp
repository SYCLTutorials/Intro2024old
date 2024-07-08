#include <iostream>

#include <sycl/sycl.hpp>

class matrix_multiply;

int main() {
    const int N = 3; // Size of the matrix NxN
    int matrixA[N][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int matrixB[N][N] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int matrixR[N][N] = {0}; // Result matrix initialized to zero

    // Check for available GPU devices
    auto gpu_selector = sycl::gpu_selector{};
    try {
        // Create a queue using the GPU selector
        auto gpuQueue = sycl::queue{gpu_selector};

        {
            // TODO:
            // write Buffers for matrices A, B, and R
            auto bufA = sycl::buffer{matrixA, sycl::range{N, N}};
            auto bufB = sycl::buffer{matrixB, sycl::range{N, N}};
            auto bufR = sycl::buffer{matrixR, sycl::range{N, N}};

            // Submit a command group to the queue for execution
            gpuQueue.submit([&](sycl::handler& cgh) {
                // TODO:
                // write accesors for buffers A, B and C
                auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
                auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
                auto accR = bufR.get_access<sycl::access::mode::write>(cgh);


                // TODO:
                // implement the parallel_for with the right dimenstions and index
                cgh.parallel_for<matrix_multiply>(sycl::range{N, N}, [=](sycl::id<2> idx) {
                    // TODO:
                    // write the computation to perform the the matrix multiplication 
                    // in parallel 
                    int row = idx[0];
                    int col = idx[1];
                    for (int k = 0; k < N; k++) {
                        accR[row][col] += accA[row][k] * accB[k][col];
                    }
                });
            }).wait(); /

            std::cout << "Matrix multiplication completed successfully on GPU.\n";
        }

    } catch (sycl::exception const& e) {
        // Fallback if no GPU is found
        std::cerr << "No GPU device found. Error: " << e.what() << '\n';
        std::cerr << "Trying to fallback to CPU.\n";
        auto cpuQueue = sycl::queue{sycl::cpu_selector{}};

        // Same operations but targeting the CPU
        {
            // TODO: 
            // write Buffers for matrices A, B, and R
            // paste your solution above here as well 
            auto bufA = sycl::buffer{matrixA, sycl::range{N, N}};
            auto bufB = sycl::buffer{matrixB, sycl::range{N, N}};
            auto bufR = sycl::buffer{matrixR, sycl::range{N, N}};

            cpuQueue.submit([&](sycl::handler& cgh) {
                // TODO:
                // write accesors for buffers A, B and C
                // paste your solution above here as well 
                auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
                auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
                auto accR = bufR.get_access<sycl::access::mode::write>(cgh);

                // TODO:
                // implement the parallel_for with the right dimenstions and index
                cgh.parallel_for<matrix_multiply>(sycl::range{N, N}, [=](sycl::id<2> idx) {
                    // TODO:
                    // write the computation to perform the the matrix multiplication 
                    // in parallel 
                    int row = idx[0];
                    int col = idx[1];
                    for (int k = 0; k < N; k++) {
                        accR[row][col] += accA[row][k] * accB[k][col];
                    }
                });
            }).wait(); 
        }

        std::cerr << "Matrix multiplication completed on CPU.\n";
    }

    return 0;
}
