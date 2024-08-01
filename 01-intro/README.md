# Introduction to SYCL Programming for GPUs

In the rapidly evolving world of computing, the ability to harness the power of heterogeneous systems—where CPUs coexist with GPUs and other accelerators—has become increasingly vital. **SYCL** stands as a cutting-edge, single-source programming model designed to bridge this gap. Developed to be used with modern C++, SYCL abstracts the complexities associated with direct accelerator programming, making it accessible to both novice and experienced developers.

### What is SYCL?

SYCL is an open standard developed by the Khronos Group, the same group responsible for OpenGL. It allows developers to write code for heterogeneous systems using completely standard C++. This means that the same code can target CPUs, GPUs, DSPs, FPGAs, and other types of accelerators without modification. SYCL builds upon the foundation laid by OpenCL, offering a higher level of abstraction and deeper integration with C++.

### Advantages of SYCL

One of the primary advantages of SYCL is its ability to integrate seamlessly with C++17 and upcoming versions, enabling features like lambda functions, auto-typing, and templating. This integration not only improves the programmability and readability of the code but also leverages the type safety and performance optimizations provided by modern C++. Here are a few key benefits:
- **Single-Source Development**: Unlike traditional approaches that might require maintaining separate code bases for different architectures, SYCL unifies the code into a single source. This simplifies development and reduces maintenance burdens.
- **Cross-Platform Portability**: SYCL code can be executed on any device that has a compatible SYCL runtime, providing true cross-platform capabilities.
- **Performance**: With SYCL, developers do not have to sacrifice performance for portability. It allows fine control over parallel execution and memory management, which are critical for achieving optimal performance on GPUs.

As GPUs continue to play a crucial role in fields ranging from scientific computing to machine learning, mastering SYCL can provide developers with the tools needed to fully exploit the capabilities of these powerful devices. The following sections will guide you through setting up your development environment, understanding the core concepts of SYCL, and walking you through practical examples to kickstart your journey in high-performance computing with SYCL.

---

This introduction sets the stage for learning SYCL by highlighting its relevance, advantages, and integration with modern C++. It aims to build a strong foundation for the subsequent sections that delve deeper into SYCL programming.

---


# Enqueuing A Kernel 

In SYCL, all computations are submitted through a queue. This queue is associated with a device, and any computation assigned to the queue is executed on this device.

SYCL offers two methods for managing data:
1. **Buffer/Accessor Model:** This model uses buffers to store data and accessors to read or write data, ensuring safe memory management and synchronization.
2. **Unified Shared Memory (USM) Model:** This model allows for direct data sharing between the host and device, simplifying memory management by eliminating the need for explicit buffers and accessors.

# Command Groups 

 A command group is a fundamental construct that encapsulates a set of operations meant to be executed on a device.

<img width="455" alt="" src="/01-intro/images/image1.png" >


- Command groups are defined by calling the **submit** function on the queue.
- The **submit** function takes a command group handler (`cgh`) which facilitates the composition of the command group.
- Inside the **submit** function, a handler is created and passed to the `cgh`.
- This handler is then used by the `cgh` to assemble the command group.

```cpp
gpuQueue.submit([&](sycl::handler &cgh) {
  /* Command group function */
})
```


# Scheduling

A schedulre is a component responsible for managing the order and execution of tasks on computational resources.

![Scheduling Overview](/01-intro/images/image3.png)

- When the **submit** function is called, it creates a command group handler (`cgh`) and submits it to the scheduler.
- The scheduler is responsible for executing the commands on the designated target device.

#### Enqueuing SYCL Kernel Function example

```cpp
class hello_world;

// Check for available GPU devices
auto gpu_selector = sycl::gpu_selector{};
try {
  // Create a queue using the GPU selector
  auto gpuQueue = sycl::queue{gpu_selector};

  // Submit a command group to the queue
  gpuQueue.submit([&](sycl::handler &cgh) {
    // Create a stream for output within kernel
    auto os = sycl::stream{128, 128, cgh};

    // Execute a single task
    cgh.single_task<hello_world>([=]() {
      os << "Hello World!\n";
    });
  }).wait(); // Wait for completion

  std::cout << "Successfully executed on GPU.\n";
} catch (sycl::exception const& e) {
  // Fallback if no GPU is found
  std::cerr << "No GPU device found. Error: " << e.what() << '\n';
  std::cerr << "Trying to fallback to CPU.\n";
  auto cpuQueue = sycl::queue{sycl::cpu_selector{}};
  cpuQueue.submit([&](sycl::handler &cgh) {
    auto os = sycl::stream{128, 128, cgh};
    cgh.single_task<hello_world>([=]() {
      os << "Hello World from CPU!\n";
    });
  }).wait();
```



# Buffers & Accessors

Buffers and accessors are used in SYCL for managing and accessing data across different computing devices, including CPUs, GPUs, and other accelerators:

![Diagram illustrating the relationship between buffers, accessors, and devices](/01-intro/images/image2.png)

- **Buffers**: Buffers are used to manage data across the host and various devices. A buffer abstractly represents a block of data and handles the storage, synchronization, and consistency of this data across different memory environments. When a buffer object is constructed, it does not immediately allocate or copy data to the device memory. This allocation or transfer only occurs when the runtime determines that a device needs to access the data, optimizing memory usage and data transfer.

- **Accessors**: Accessors are used to request access to data that is stored in buffers. They specify how and when the data in a buffer should be accessed by a kernel function, either on the host or a specific device. Accessors help in defining the required access pattern (read, write, or read/write) and are crucial for ensuring data consistency and coherency between the host and devices.

```cpp

class vector_dot_product;

const int N = 1024; // Length of the vectors
std::vector<int> vectorA(N, 1); // Vector A filled with 1s
std::vector<int> vectorB(N, 2); // Vector B filled with 2s
int result = 0; // Result of dot product initialized to zero

auto gpu_selector = sycl::gpu_selector{};

auto bufA = sycl::buffer{vectorA.data(), sycl::range{N}};
auto bufB = sycl::buffer{vectorB.data(), sycl::range{N}};
auto bufResult = sycl::buffer{&result, sycl::range{1}};

gpuQueue.submit([&](sycl::handler& cgh) {
    auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
    auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
    auto accResult = bufResult.get_access<sycl::access::mode::read_write>(cgh);

    cgh.parallel_for<vector_dot_product>(sycl::range{N}, [=](sycl::id<1> idx) {
        int i = idx[0];
        accResult[0] += accA[i] * accB[i];
    });
}).wait();

```

# How to compile SYCL code



> If you're gonna build a time machine into a car, why not do it with some style?
>
> — *Back to the Future*
