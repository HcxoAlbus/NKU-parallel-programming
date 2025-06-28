# Makefile for CUDA ANNS program
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenmp
NVCCFLAGS = -std=c++11 -O3 -arch=sm_75 -lcublas
TARGET = anns_gpu
SOURCE = main.cu

# Default target
all: $(TARGET)

# Compile CUDA program
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCE)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build files
clean:
	rm -f $(TARGET) $(TARGET)_* *.txt

# Performance test with different optimization levels
test_performance: 
	@echo "=== Testing different CUDA optimization levels ==="
	@echo "Building with -O0..."
	$(NVCC) -std=c++11 -O0 -arch=sm_75 -lcublas -o $(TARGET)_O0 $(SOURCE)
	@echo "Building with -O2..."
	$(NVCC) -std=c++11 -O2 -arch=sm_75 -lcublas -o $(TARGET)_O2 $(SOURCE)
	@echo "Building with -O3..."
	$(NVCC) -std=c++11 -O3 -arch=sm_75 -lcublas -o $(TARGET)_O3 $(SOURCE)
	@echo "Building with -O3 and fast math..."
	$(NVCC) -std=c++11 -O3 -arch=sm_75 -lcublas --use_fast_math -o $(TARGET)_O3_fast $(SOURCE)

# Run performance comparison
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

.PHONY: all run clean test_performance benchmark check_gpu