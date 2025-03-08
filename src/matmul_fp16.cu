#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

#define N 1024  // Matrix size

__global__ void matmul_fp16(const half *A, const half *B, half *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        half sum = __float2half(0.0f);
        for (int k = 0; k < N; ++k) {
            sum = __hadd(sum, __hmul(A[row * N + k], B[k * N + col]));
        }
        C[row * N + col] = sum;
    }
}
