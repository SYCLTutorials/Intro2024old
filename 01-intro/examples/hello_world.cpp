#include <iostream>
#include <sycl/sycl.hpp>

class hello_world_gpu;
class hello_world_cpu;

int main() {
    // Check for available GPU devices
    auto gpu_selector = sycl::gpu_selector_v;

    try {
        // Create a queue using the GPU selector
        auto gpuQueue = sycl::queue{gpu_selector};

        // Submit a command group to the queue
        gpuQueue
            .submit([&](sycl::handler &cgh) {
                // Create a stream for output within kernel
                auto os = sycl::stream{128, 128, cgh};

                // Execute a single task
                cgh.single_task<hello_world_gpu>(
                    [=]() { os << "Hello World!\n"; });
            })
            .wait();  // Wait for completion

        std::cout << "Successfully executed on GPU.\n";

    } catch (sycl::exception const &e) {
        // Fallback if no GPU is found
        std::cerr << "No GPU device found. Error: " << e.what() << '\n';
        std::cerr << "Trying to fallback to CPU.\n";
        auto cpuQueue = sycl::queue{sycl::cpu_selector_v};
        cpuQueue
            .submit([&](sycl::handler &cgh) {
                auto os = sycl::stream{128, 128, cgh};
                cgh.single_task<hello_world_cpu>(
                    [=]() { os << "Hello World from CPU!\n"; });
            })
            .wait();
    }

    return 0;
}
