# Compiler
NVCC = nvcc

# Output file
TARGET = matmul_fp16

# Source files
SRC = src/main.cpp src/matmul_fp16.cu

# CUDA architecture (adjust based on your GPU)
ARCH = -arch=sm_75

# Compilation rule
all:
	$(NVCC) -o $(TARGET) $(SRC) $(ARCH)

# Clean up
clean:
	rm -f $(TARGET)
