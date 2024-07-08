## Exercise: Matrix Multiplication

---

In this exercise, you will learn how to perform matrix multiplication using SYCL, managing data with buffers and accessors, and executing code on both GPU and CPU devices as fallback.

---

### 1.) Matrix Setup

Initialize matrices for the operation. You will multiply two matrices and store the result in a third matrix.

```cpp
const int N = 3; // Size of the matrix NxN
int matrixA[N][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
int matrixB[N][N] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
int matrixR[N][N] = {0}; // Result matrix initialized to zero
```

### 2.) Device Selection and Queue Setup

Create a queue to manage the device operations. Start with GPU and fallback to CPU if no suitable GPU is found.

```cpp
auto gpu_selector = sycl::gpu_selector{};
try {
    auto gpuQueue = sycl::queue{gpu_selector};
} catch (sycl::exception const& e) {
    std::cerr << "No GPU device found. Error: " << e.what() << '\n';
    auto cpuQueue = sycl::queue{sycl::cpu_selector{}};
}
```

### 3.) Data Management with Buffers and Accessors

#### **TODO:**
- Construct buffers for matrices A, B, and R.
- Create accessors within a command group.

### 4.) Kernel Implementation

#### **TODO:**
- Write a kernel function to perform matrix multiplication. This function should:
  - Utilize a 2D index to access elements correctly.
  - Compute the product for a single element of the result matrix by iterating over one dimension.

### 5.) Execution and Synchronization

Ensure your SYCL operations are correctly synchronized and the results are written back to the host memory.

```cpp
gpuQueue.submit([&](sycl::handler& cgh) {
    // Your kernel launch code here
}).wait();
```

