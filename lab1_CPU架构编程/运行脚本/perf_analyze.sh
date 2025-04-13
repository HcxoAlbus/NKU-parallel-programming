#!/bin/bash

# 设置变量
SOURCE_FILE="matrix_operations.cpp"
BINARY_NAME="matrix_operations"
MATRIX_SIZES=(1024 2048 3000 3400 3600)  # 包含可能观察到异常行为的矩阵尺寸
ITERATIONS=3  # 每个尺寸运行的基准测试迭代次数
RESULTS_DIR="perf_results"

# 确保结果目录存在
mkdir -p "$RESULTS_DIR"

# 使用优化标志编译程序并保留调试信息
g++ -O2 -g -o "$BINARY_NAME" "$SOURCE_FILE"

echo "===== 编译完成 ====="

# 添加时间戳函数
timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

# 创建运行摘要文件
SUMMARY_FILE="$RESULTS_DIR/performance_summary.txt"
echo "============ perf 性能分析摘要 ============" > "$SUMMARY_FILE"
echo "生成时间: $(timestamp)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# 对每种矩阵尺寸运行分析
for size in "${MATRIX_SIZES[@]}"; do
  echo "===== 开始分析矩阵尺寸: $size ====="
  
  # 创建此矩阵尺寸的目录
  SIZE_DIR="$RESULTS_DIR/size_$size"
  mkdir -p "$SIZE_DIR"
  
  echo "矩阵尺寸: $size x $size" >> "$SUMMARY_FILE"
  echo "------------------------------------" >> "$SUMMARY_FILE"
  
  # 1. 先运行普通性能测试作为基准
  echo "运行基准性能测试..."
  ./"$BINARY_NAME" "$size" "$ITERATIONS" > "$SIZE_DIR/baseline_results.txt"
  
  # 2. 使用 perf stat 收集基本性能指标
  echo "收集基本性能统计..."
  perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,branch-misses,bus-cycles -o "$SIZE_DIR/perf_stat.txt" ./"$BINARY_NAME" "$size" 1
  
  # 提取关键缓存指标并添加到摘要
  echo "基本性能统计指标:" >> "$SUMMARY_FILE"
  grep -E 'cache-misses|L1-dcache-load-misses|cycles|instructions|branch-misses' "$SIZE_DIR/perf_stat.txt" >> "$SUMMARY_FILE"
  
  # 3. 对每个算法使用 perf record 采样
  echo "对四个算法分别进行性能采样..."
  
  # 创建临时测试程序，一次只运行一个算法
  for algo_num in {1..4}; do
    ALGO_NAME="algorithm$algo_num"
    if [ "$algo_num" -eq 3 ]; then
      ALGO_NAME="algorithm1_unroll4"
    elif [ "$algo_num" -eq 4 ]; then
      ALGO_NAME="algorithm2_unroll4"
    fi
    
    echo "分析 $ALGO_NAME..."
    
    # 创建单算法测试程序
    cat > "single_algo_test.cpp" << EOF
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <string>
using namespace std;
using namespace std::chrono;

// 从原程序复制必要的代码，但只运行一个算法
class MatrixOperations {
private:
    int n;
    vector<double> a;
    vector<vector<double>> b;
    vector<double> sum;

    void initializeData() {
        mt19937 rng(12345);
        uniform_real_distribution<double> dist(0.5, 1.5);

        for (int i = 0; i < n; i++) {
            a[i] = 1.0 + (i % 10) / 10.0;
            for (int j = 0; j < n; j++) {
                b[i][j] = 0.5 + ((i+j) % 10) / 20.0;
            }
        }
    }

public:
    MatrixOperations(int size) : n(size) {
        a.resize(n);
        b.resize(n, vector<double>(n));
        sum.resize(n);
        initializeData();
    }

    // 平凡算法：逐列访问矩阵元素
    void algorithm1() {
        for (int i = 0; i < n; i++) {
            sum[i] = 0.0;
            for (int j = 0; j < n; j++) {
                sum[i] += b[j][i] * a[j];
            }
        }
    }

    // 优化算法：逐行访问矩阵元素
    void algorithm2() {
        for (int i = 0; i < n; i++) {
            sum[i] = 0.0;
        }
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                sum[i] += b[j][i] * a[j];
            }
        }
    }

    // 平凡算法的循环展开版本(展开4次)
    void algorithm1_unroll4() {
        for (int i = 0; i < n; i++) {
            sum[i] = 0.0;
            int j = 0;
            for (; j <= n-4; j += 4) {
                sum[i] += b[j][i] * a[j];
                sum[i] += b[j+1][i] * a[j+1];
                sum[i] += b[j+2][i] * a[j+2];
                sum[i] += b[j+3][i] * a[j+3];
            }
            for (; j < n; j++) {
                sum[i] += b[j][i] * a[j];
            }
        }
    }

    // 优化算法的循环展开版本(展开4次)
    void algorithm2_unroll4() {
        for (int i = 0; i < n; i++) {
            sum[i] = 0.0;
        }
        
        for (int j = 0; j < n; j++) {
            double a_val = a[j];
            int i = 0;
            for (; i <= n-4; i += 4) {
                sum[i] += b[j][i] * a_val;
                sum[i+1] += b[j][i+1] * a_val;
                sum[i+2] += b[j][i+2] * a_val;
                sum[i+3] += b[j][i+3] * a_val;
            }
            for (; i < n; i++) {
                sum[i] += b[j][i] * a_val;
            }
        }
    }

    void runAlgorithm$algo_num() {
        // 多次运行以确保足够的采样
        for (int i = 0; i < 3; i++) {
            $ALGO_NAME();
        }
    }
};

int main(int argc, char* argv[]) {
    int n = 1024;
    if (argc > 1) n = atoi(argv[1]);
    
    try {
        MatrixOperations matrixOps(n);
        matrixOps.runAlgorithm$algo_num();
    } catch (const exception& e) {
        cerr << "发生错误: " << e.what() << endl;
        return 1;
    }
    return 0;
}
EOF

    # 编译单算法程序
    g++ -O2 -g -o "single_algo_$algo_num" "single_algo_test.cpp"
    
    # 使用perf record收集采样数据
    perf record -F 999 -g -o "$SIZE_DIR/perf_${ALGO_NAME}.data" "./single_algo_$algo_num" "$size"
    
    # 生成报告
    perf report -i "$SIZE_DIR/perf_${ALGO_NAME}.data" --stdio > "$SIZE_DIR/perf_${ALGO_NAME}_report.txt"
    
    # 生成缓存相关详细统计
    perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses -o "$SIZE_DIR/perf_${ALGO_NAME}_cache_stat.txt" "./single_algo_$algo_num" "$size"
    
    # 添加到摘要
    echo "" >> "$SUMMARY_FILE"
    echo "$ALGO_NAME 性能指标:" >> "$SUMMARY_FILE"
    grep -E 'load-misses|loads|cycles|instructions' "$SIZE_DIR/perf_${ALGO_NAME}_cache_stat.txt" >> "$SUMMARY_FILE"
    
    # 清理
    rm "single_algo_$algo_num"
  done
  
  rm "single_algo_test.cpp"
  
  echo "" >> "$SUMMARY_FILE"
  echo "完成矩阵尺寸 $size 的分析"
done

echo "===== 性能分析完成 ====="
echo "结果保存在 $RESULTS_DIR 目录"