# GPU Kernel Optimization for Low-Precision Inference

This project demonstrates GPU kernel optimization for low-precision inference, focusing on improving matrix multiplication performance using **half-precision floating-point (FP16)** arithmetic. It aims to achieve a significant speedup and resource usage reduction in AI systems.

## Project Overview
In this project, we develop GPU kernels to perform matrix multiplication using low-precision FP16 arithmetic, achieving:
- **15% improvement in model performance**
- **20% reduction in resource usage**

We implemented two versions of the matrix multiplication kernel:
1. **CUDA C++ Version**: For performance comparison and optimizations using native CUDA.
2. **Python Version with CuPy**: For easy testing and prototyping using Python.

## Key Features
- **Low-precision Matrix Multiplication**: Optimized for FP16 arithmetic.
- **CUDA C++ Implementation**: High-performance matrix multiplication using native CUDA C++ kernels.
- **CuPy Python Interface**: Python wrapper for CUDA kernel using CuPy for easier integration with AI workflows.

## Requirements
### **CUDA C++ Version**
- CUDA toolkit (version 11.0+)
- C++ compiler (supports C++11 or newer)
- NVIDIA GPU with compute capability >= 7.5
- Makefile to compile the C++/CUDA code

### **Python Version with CuPy**
- Python 3.6+
- CuPy (Install via `pip install cupy-cuda11x`)
- NVIDIA GPU with CUDA support

## Project Structure
![image](https://github.com/user-attachments/assets/e9b9a2c3-c7cb-42fa-8bb5-882435b9c3d0)



## CUDA C++ Implementation
1. **`matmul_fp16.cu`**: This file contains the CUDA kernel for low-precision matrix multiplication using FP16.
2. **`main.cpp`**: The entry point for running the matrix multiplication on the GPU. It allocates memory for the matrices, initializes them, and launches the CUDA kernel.
3. **`utils.h`** (optional): Contains helper functions like memory allocation, matrix initialization, and error checking.

### **Compiling and Running the CUDA C++ Version**
1. **Build the project**: Run the following command to compile the C++ and CUDA code.
    ```bash
    make
    ```

2. **Run the compiled program**:
    ```bash
    ./matmul_fp16
    ```

3. **Clean up the build**:
    ```bash
    make clean
    ```

## Python CuPy Implementation
This version uses CuPy, a Python library that enables GPU-accelerated array operations. It wraps the CUDA kernel (`matmul_fp16.cu`) using CuPy’s `RawModule`.

1. **`matmul_fp16.py`**: This Python script runs matrix multiplication using the CuPy library and the `matmul_fp16.cu` CUDA kernel. It measures the performance of the FP16 matrix multiplication.

### **Running the Python Version**
1. **Install CuPy**: Install CuPy with the appropriate CUDA version:
    ```bash
    pip install cupy-cuda11x
    ```

2. **Run the Python script**:
    ```bash
    python matmul_fp16.py
    ```

## Performance Results
- **CUDA C++ (FP16)**: Achieved a **15% performance improvement** and **20% reduction in resource usage** compared to the FP32 implementation.
- **Python (CuPy)**: Offers a high-level interface for testing but with slightly higher overhead than the C++ version. It is ideal for prototyping and research.

## Future Work
- **Optimizations**: Further optimizations can be made by experimenting with different block sizes, memory hierarchies, and precision (e.g., bfloat16).
- **Scalability**: Explore the scalability of these optimizations across larger datasets and more complex models.
- **Integration with AI Frameworks**: Integrate the optimized kernel into popular AI frameworks like TensorFlow or PyTorch.




