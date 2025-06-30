# Makefile for CUDA ANNS program
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++14 -O3 -fopenmp
NVCCFLAGS = -std=c++14 -O3 -arch=sm_75 -lcublas
TARGET = anns_gpu
TARGET_IVF = anns_ivf_gpu
SOURCE = main.cu
SOURCE_IVF = main_ivf.cu

# Default target - build both programs
all: $(TARGET) $(TARGET_IVF)

# Compile original CUDA program
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCE)

# 修复IVF CUDA程序编译 - 使用C++17和更兼容的参数
$(TARGET_IVF): $(SOURCE_IVF)
	$(NVCC) $(NVCCFLAGS) -Xcompiler "-fopenmp" -o $(TARGET_IVF) $(SOURCE_IVF)

# Run the original program
run: $(TARGET)
	./$(TARGET)

# Run the IVF program
run_ivf: $(TARGET_IVF)
	./$(TARGET_IVF)

# Clean build files
clean:
	rm -f $(TARGET) $(TARGET_IVF) $(TARGET)_* $(TARGET_IVF)_* *.txt anns_gpu_* anns_ivf_gpu_*

# Performance test with different optimization levels for IVF
test_performance_ivf: 
	@echo "=== Testing IVF CUDA with different optimization levels ==="
	@echo "Building with -O0..."
	$(NVCC) -std=c++14 -O0 -arch=sm_75 -lcublas -Xcompiler -fopenmp --expt-relaxed-constexpr --expt-extended-lambda -o $(TARGET_IVF)_O0 $(SOURCE_IVF)
	@echo "Building with -O2..."
	$(NVCC) -std=c++14 -O2 -arch=sm_75 -lcublas -Xcompiler -fopenmp --expt-relaxed-constexpr --expt-extended-lambda -o $(TARGET_IVF)_O2 $(SOURCE_IVF)
	@echo "Building with -O3..."
	$(NVCC) -std=c++14 -O3 -arch=sm_75 -lcublas -Xcompiler -fopenmp --expt-relaxed-constexpr --expt-extended-lambda -o $(TARGET_IVF)_O3 $(SOURCE_IVF)
	@echo "Building with -O3 and fast math..."
	$(NVCC) -std=c++14 -O3 -arch=sm_75 -lcublas -Xcompiler -fopenmp --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda -o $(TARGET_IVF)_O3_fast $(SOURCE_IVF)

# Performance test with different optimization levels for original
test_performance: 
	@echo "=== Testing different CUDA optimization levels ==="
	@echo "Building with -O0..."
	$(NVCC) -std=c++14 -O0 -arch=sm_75 -lcublas -o $(TARGET)_O0 $(SOURCE)
	@echo "Building with -O2..."
	$(NVCC) -std=c++14 -O2 -arch=sm_75 -lcublas -o $(TARGET)_O2 $(SOURCE)
	@echo "Building with -O3..."
	$(NVCC) -std=c++14 -O3 -arch=sm_75 -lcublas -o $(TARGET)_O3 $(SOURCE)
	@echo "Building with -O3 and fast math..."
	$(NVCC) -std=c++14 -O3 -arch=sm_75 -lcublas --use_fast_math -o $(TARGET)_O3_fast $(SOURCE)

# Run performance comparison for IVF
benchmark_ivf: test_performance_ivf
	@echo "\n=== IVF Performance Benchmark Results ==="
	@echo "\n--- O0 Optimization ---"
	./$(TARGET_IVF)_O0 > ivf_results_O0.txt 2>&1 || true
	@echo "\n--- O2 Optimization ---"
	./$(TARGET_IVF)_O2 > ivf_results_O2.txt 2>&1 || true
	@echo "\n--- O3 Optimization ---"
	./$(TARGET_IVF)_O3 > ivf_results_O3.txt 2>&1 || true
	@echo "\n--- O3 + Fast Math ---"
	./$(TARGET_IVF)_O3_fast > ivf_results_O3_fast.txt 2>&1 || true

# Run performance comparison for original
benchmark: test_performance
	@echo "\n=== Performance Benchmark Results ==="
	@echo "\n--- O0 Optimization ---"
	./$(TARGET)_O0 > results_O0.txt 2>&1 || true
	@echo "\n--- O2 Optimization ---"
	./$(TARGET)_O2 > results_O2.txt 2>&1 || true
	@echo "\n--- O3 Optimization ---"
	./$(TARGET)_O3 > results_O3.txt 2>&1 || true
	@echo "\n--- O3 + Fast Math ---"
	./$(TARGET)_O3_fast > results_O3_fast.txt 2>&1 || true

# Check CUDA devices
check_gpu:
	nvidia-smi || echo "GPU检测失败，但不影响编译"
	nvcc --version

.PHONY: all run run_ivf clean test_performance test_performance_ivf benchmark benchmark_ivf check_gpu