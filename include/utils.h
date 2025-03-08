#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <iostream>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << " at line "        \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

// Allocate memory for a matrix
inline half* allocate_matrix(int N) {
    half *matrix;
    CUDA_CHECK(cudaMallocManaged(&matrix, N * N * sizeof(half)));
    return matrix;
}

// Initialize a matrix with random half-precision values
inline void initialize_matrix(half *matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
}

#endif  // UTILS_H
