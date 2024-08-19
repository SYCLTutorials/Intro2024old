
---

## Exercise: Matrix Multiplication

---

In this exercise, you will learn how to perform matrix multiplication using SYCL, managing data with buffers and accessors, and executing code on a GPU. You will set up the matrices, select the appropriate device, manage data using SYCL buffers, and implement a parallel kernel to perform the multiplication.

---

### 1.) Matrix Setup

Initialize matrices for the operation. You will multiply two matrices and store the result in a third matrix. Here, we'll use `std::array` to store the matrix data.

```cpp
const int N = 3; // Size of the matrix NxN
std::array<int, N * N> matrixA = {1, 2, 3, 4, 5, 6, 7, 8, 9};
std::array<int, N * N> matrixB = {9, 8, 7, 6, 5, 4, 3, 2, 1};
std::array<int, N * N> matrixR = {0}; // Result matrix initialized to zero
```

### 2.) Device Selection and Queue Setup

Create a queue to manage the device operations. Start with a GPU queue, and handle the device selection.

```cpp
auto gpu_selector = sycl::gpu_selector_v;
try {
    sycl::queue gpuQueue{gpu_selector};
} catch (sycl::exception const& e) {
    std::cerr << "Failed to execute on GPU. Error: " << e.what() << '\n';
}
```

### 3.) Data Management with Buffers and Accessors

#### **TODO:**
- Construct buffers for matrices A, B, and R using `sycl::buffer` and the `data()` method from `std::array`.
- Create accessors within a command group to manage read and write operations on the buffers.

### 4.) Kernel Implementation

#### **TODO:**
- Write a kernel function using `parallel_for` to perform matrix multiplication. This function should:
  - Utilize a 2D index to access elements correctly.
  - Compute the product for a single element of the result matrix by iterating over one dimension.
  
Example kernel implementation:
```cpp
cgh.parallel_for<matrix_multiply_gpu>(sycl::range<2>{N, N}, [=](sycl::id<2> idx) {
    int row = idx[0];
    int col = idx[1];
    for (int k = 0; k < N; k++) {
        accR[row][col] += accA[row * N + k] * accB[k * N + col];
    }
});
```

### 5.) Execution and Synchronization

Ensure your SYCL operations are correctly synchronized and the results are written back to the host memory.

```cpp
gpuQueue.submit([&](sycl::handler& cgh) {
    // Your kernel launch code here
}).wait();
```

### 6.) Result Verification

Print the result matrix to verify that the multiplication was performed correctly.

### 7.) Test on Development Platforms

Test your SYCL program on different platforms. Below are examples for compiling and running on Intel DevCloud and Polaris.

#### **Test on Intel DevCloud**

```bash
icpx -fsycl source_file.cpp -o output_file -std=c++17 -lOpenCL
```

#### **Test on Polaris**

```bash
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -std=c++17 source_file.cpp -o output_file -lOpenCL
```

---
