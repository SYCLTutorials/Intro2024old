# Introduction to SYCL Programming for GPUs

In the rapidly evolving world of computing, the ability to harness the power of heterogeneous systems—where CPUs coexist with GPUs and other accelerators—has become increasingly vital. **SYCL** stands as a cutting-edge, single-source programming model designed to bridge this gap. Developed to be used with modern C++, SYCL abstracts the complexities associated with direct accelerator programming, making it accessible to both novice and experienced developers.

### What is SYCL?

SYCL is an open standard developed by the Khronos Group, the same group responsible for OpenGL. It allows developers to write code for heterogeneous systems using completely standard C++. This means that the same code can target CPUs, GPUs, DSPs, FPGAs, and other types of accelerators without modification. SYCL builds upon the foundation laid by OpenCL, offering a higher level of abstraction and deeper integration with C++.

### Advantages of SYCL

One of the primary advantages of SYCL is its ability to integrate seamlessly with C++17 and upcoming versions, enabling features like lambda functions, auto-typing, and templating[^2]. This integration not only improves the programmability and readability of the code but also leverages the type safety and performance optimizations provided by modern C++. Here are a few key benefits:
- **Single-Source Development**: Unlike traditional approaches that might require maintaining separate code bases for different architectures, SYCL unifies the code into a single source. This simplifies development and reduces maintenance burdens.
- **Cross-Platform Portability**: SYCL code can be executed on any device that has a compatible SYCL runtime, providing true cross-platform capabilities.
- **Performance**: With SYCL, developers do not have to sacrifice performance for portability. It allows fine control over parallel execution and memory management, which are critical for achieving optimal performance on GPUs.

As GPUs continue to play a crucial role in fields ranging from scientific computing to machine learning, mastering SYCL can provide developers with the tools needed to fully exploit the capabilities of these powerful devices. The following sections will guide you through setting up your development environment, understanding the core concepts of SYCL, and walking you through practical examples to kickstart your journey in high-performance computing with SYCL.

---

This introduction sets the stage for learning SYCL by highlighting its relevance, advantages, and integration with modern C++. It aims to build a strong foundation for the subsequent sections that delve deeper into SYCL programming.

---

[^1]: SYCL Academy by Codeplay Software, available at [https://github.com/codeplaysoftware/syclacademy](https://github.com/codeplaysoftware/syclacademy).
[^2]: Reinders, J., Ashbaugh, B., Brodman, J., Kinsner, M., Pennycook, J., & Tian, X. (2021). *Data Parallel C++: Mastering DPC++ for Programming of Heterogeneous Systems using C++ and SYCL*. Apress. ISBN: 978-1484275282. Available at [https://www.apress.com/gp/book/9781484275282](https://www.apress.com/gp/book/9781484275282).


# Basics of a SYCL Kernel 

In SYCL, all computations are submitted through a queue. This queue is associated with a device, and any computation assigned to the queue is executed on this device[^1].
This is how we check if a gpu is available for use and then initialize a sycl queue for a gpu:
```cpp
// Check for available GPU devices
auto gpu_selector = sycl::gpu_selector{};

// Create a queue using the GPU selector
auto gpuQueue = sycl::queue{gpu_selector};
```

SYCL offers two methods for managing data:
1. **Buffer/Accessor Model:** This model uses buffers to store data and accessors to read or write data, ensuring safe memory management and synchronization. Here is an example of how you could do it for a dot product between 2 vectors and store the answer:

```cpp
// Buffers 
auto bufA = sycl::buffer{vectorA.data(), sycl::range{N}};
auto bufB = sycl::buffer{vectorB.data(), sycl::range{N}};
auto bufResult = sycl::buffer{&result, sycl::range{1}};

// Accessor
auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
auto accResult = bufResult.get_access<sycl::access::mode::read_write>(cgh);
```

3. **Unified Shared Memory (USM) Model:** This model allows for direct data sharing between the host and device, simplifying memory management by eliminating the need for explicit buffers and accessors. Here is the following changes from the buffer/accessor model to USM model:

```cpp
// Allocate memory using USM
 float* usmA = sycl::malloc_shared<float>(N, gpuQueue);
 float* usmB = sycl::malloc_shared<float>(N, gpuQueue);
 float* usmResult = sycl::malloc_shared<float>(1, gpuQueue);

 // Initialize USM memory
 std::copy(vectorA.begin(), vectorA.end(), usmA);
 std::copy(vectorB.begin(), vectorB.end(), usmB);
 *usmResult = 0.0f;
```

# Understanding SYCL Kernel Command Group Execution

 A command group is a fundamental construct that encapsulates a set of operations meant to be executed on a device.

 ```cpp
gpuQueue.submit([&](sycl::handler &cgh) {
  /* Command group function */
})
```

<img width="455" alt="" src="/01-intro/images/image1.png" >

> The diagram illustrates the process of defining and submitting a SYCL command group.
> It begins with a call to the submit function on a SYCL queue, which initiates the creation of a command group.
> The submit function takes a command group function as its argument, within which a command group handler `cgh` is created.
> Inside the command group function, the handler is used to specify dependencies, define the kernel function, and set up accessors for memory objects that the kernel will use. Once these elements are defined, the command group is assembled and ready for execution on the device.


# Task Scheduling and Execution in SYCL 

A schedulre is a component responsible for managing the order and execution of tasks on computational resources.

![Scheduling Overview](/01-intro/images/image3.png)

> The provided diagram illustrates the process of task scheduling and execution in SYCL.
> The sequence begins with the Queue, where tasks are initially submitted. Once a task is submitted, it is encapsulated within a Command Group (CG), which contains all the necessary information and dependencies for execution.
> The Scheduler then takes over, determining the optimal order and timing for executing the command group on the available computational resources.
> Finally, the commands are dispatched to the Target Device, where the actual computation takes place

# Getting started with Hello World 

This example demonstrates the basics of getting started with SYCL by executing a simple `Hello World` program on a GPU, with a fallback to the CPU if no GPU is available.

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

The program begins by attempting to select an available GPU device using `sycl::gpu_selector`. If a GPU is found, a queue `gpuQueue` is created for the GPU device. Within this queue, a command group is submitted using the submit function. Inside the command group, a `sycl::stream` object `os` is created for output within the kernel, and a single task is executed using `cgh.single_task`, which prints `Hello World!` to the stream. The program waits for the completion of the submitted task with `.wait()` and prints a success message to the console.


# Putting it all together with a vector dot product  

If you look over the code, you'll identify that it incorporates everything we learned from the above, utilizing these concepts to form a parallel computation for a linear algebra problem. In the example, we use small sizes for our vectors, but if given vectors of larger size, SYCL would take advantage of the GPU to compute the dot product $\langle\ u, v \rangle$.

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
To compile SYCL code on your computer, you need to have a SYCL-compatible compiler installed. One commonly used compiler is DPC++ (Data Parallel C++), which is part of the Intel oneAPI toolkit. Once you have this you
can run for example:

```bash
dpcpp -o hello_world hello_world.cpp
```

If you're using `clang` the following should help you compile the code:

```bash
clang++ -fsycl -o hello_world hello_world.cpp
```

If you have access to Argonne National Lab you can check out our way to compile on Polaris by going here [Polaris](polaris.md)

# Addtional examples

Explore more advanced examples in our other directories, where we go beyond simple mathematical computations. Here, we focus on leveraging parallelism to tackle problems in electron density, marching cubes, and shortest-path algorithms using heterogeneous hardware, all implemented with SYCL. Each example is designed to showcase the power and flexibility of SYCL in handling complex computational tasks efficiently.

If you're curious and want a sneak peek into these problems, check out the following directories for a brief description and detailed examples:

- [Electron denisty](../02-electrondensity)
- [Marching Cubes](../03-marchingCubes)
- [Single Source Shortest-Path](../04-sssp)


***


> If you're gonna build a time machine into a car, why not do it with some style?
>
> — *Back to the Future*

