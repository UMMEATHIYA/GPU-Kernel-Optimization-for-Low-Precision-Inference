import cupy as cp
import time

N = 1024
A = cp.random.rand(N, N).astype(cp.float16)
B = cp.random.rand(N, N).astype(cp.float16)

with open("kernels/matmul_fp16.cu", "r") as f:
    kernel_code = f.read()

module = cp.RawModule(code=kernel_code)
matmul_fp16 = module.get_function("matmul_fp16")

C = cp.zeros((N, N), dtype=cp.float16)

block = (16, 16)
grid = ((N + 15) // 16, (N + 15) // 16)

start = time.time()
matmul_fp16((grid,), (block,), (A, B, C, N))
cp.cuda.Device(0).synchronize()
end = time.time()

print(f"GPU FP16 Matrix Multiplication Time: {end - start:.6f} seconds")
