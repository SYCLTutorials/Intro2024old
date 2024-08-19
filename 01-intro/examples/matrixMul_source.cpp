#include <iostream>
#include <sycl/sycl.hpp>

class matrix_multiply_gpu;
class matrix_multiply_cpu;

int main() {
    const int N = 3;  // Size of the matrix NxN
    int matrixA[N][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int matrixB[N][N] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int matrixR[N][N] = {0};  // Result matrix initialized to zero

    // Check for available GPU devices
    auto gpu_selector = sycl::gpu_selector{};
    try {
        // Create a queue using the GPU selector
        auto gpuQueue = sycl::queue{gpu_selector};

        // TODO: Allocate buffers for matrices A, B, and R

        // Submit a command group to the queue for execution
        gpuQueue
            .submit([&](sycl::handler& cgh) {
                // TODO: Create accessors for buffers A, B, and R

                // TODO: Implement matrix multiplication using parallel_for
                cgh.parallel_for<matrix_multiply_gpu>(sycl::range{N, N},
                                                      [=](sycl::id<2> idx) {
                                                          // The computation for
                                                          // matrix
                                                          // multiplication
                                                          // should be
                                                          // implemented here
                                                      });
            })
            .wait();

        std::cout << "Matrix multiplication completed successfully on GPU.\n";
    } catch (sycl::exception const& e) {
        // Fallback if no GPU is found
        std::cerr << "No GPU device found. Error: " << e.what() << '\n';
        std::cerr << "Trying to fallback to CPU.\n";
        auto cpuQueue = sycl::queue{sycl::cpu_selector{}};

        // TODO: Repeat the same operations but targeting the CPU
        // Ensure to include buffer allocation, accessor creation, and the
        // parallel_for implementation

        std::cerr << "Matrix multiplication completed on CPU.\n";
    }

    return 0;
}
