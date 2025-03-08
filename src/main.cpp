#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

#define N 1024

__global__ void matmul_fp16(const half *A, const half *B, half *C, int N);

void gpu_matmul_fp16() {
    size_t size = N * N * sizeof(half);
    half *h_A = (half*)malloc(size);
    half *h_B = (half*)malloc(size);
    half *h_C = (half*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
        h_B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }

    half *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    
    matmul_fp16<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() {
    gpu_matmul_fp16();
    std::cout << "Low-precision matrix multiplication completed!" << std::endl;
    return 0;
}
