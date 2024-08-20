#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

class vector_dot_product_gpu;
class vector_dot_product_cpu;

int main() {
    const int N = 1024;              // Length of the vectors
    std::vector<int> vectorA(N, 1);  // Vector A filled with 1s
    std::vector<int> vectorB(N, 2);  // Vector B filled with 2s
    int result = 0;  // Result of dot product initialized to zero

    auto gpu_selector = sycl::gpu_selector_v;
    try {
        auto gpuQueue = sycl::queue{gpu_selector};

        {
            auto bufA = sycl::buffer{vectorA.data(), sycl::range{N}};
            auto bufB = sycl::buffer{vectorB.data(), sycl::range{N}};
            auto bufResult = sycl::buffer{&result, sycl::range{1}};

            gpuQueue
                .submit([&](sycl::handler& cgh) {
                    auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
                    auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
                    auto accResult =
                        bufResult.get_access<sycl::access::mode::read_write>(
                            cgh);

                    cgh.parallel_for<vector_dot_product_gpu>(
                        sycl::range{N}, [=](sycl::id<1> idx) {
                            int i = idx[0];
                            accResult[0] += accA[i] * accB[i];
                        });
                })
                .wait();

            std::cout << "Dot product completed successfully on GPU. Result: "
                      << result << '\n';
        }

    } catch (sycl::exception const& e) {
        std::cerr << "No GPU device found. Error: " << e.what() << '\n';
        std::cerr << "Trying to fallback to CPU.\n";
        auto cpuQueue = sycl::queue{sycl::cpu_selector_v};

        {
            auto bufA = sycl::buffer{vectorA.data(), sycl::range{N}};
            auto bufB = sycl::buffer{vectorB.data(), sycl::range{N}};
            auto bufResult = sycl::buffer{&result, sycl::range{1}};

            cpuQueue
                .submit([&](sycl::handler& cgh) {
                    auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
                    auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
                    auto accResult =
                        bufResult.get_access<sycl::access::mode::read_write>(
                            cgh);

                    cgh.parallel_for<vector_dot_product_cpu>(
                        sycl::range{N}, [=](sycl::id<1> idx) {
                            int i = idx[0];
                            accResult[0] += accA[i] * accB[i];
                        });
                })
                .wait();

            std::cout << "Dot product completed on CPU. Result: " << result
                      << '\n';
        }
    }

    return 0;
}
