#!/bin/bash

# 设置测试参数
MATRIX_SIZES=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300
1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700 
2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000 )
ITERATIONS=10
ARCHITECTURE=$(uname -m)
OUTPUT_DIR="results_${ARCHITECTURE}"

# 创建结果目录
mkdir -p $OUTPUT_DIR

# 编译不同优化级别的程序
compile_programs() {
    echo "编译测试程序..."
    g++ -std=c++11 -O0 -o matrix_test_x86_O0 matrix_benchmark.cpp
    g++ -std=c++11 -O1 -o matrix_test_x86_O1 matrix_benchmark.cpp
    g++ -std=c++11 -O2 -o matrix_test_x86_O2 matrix_benchmark.cpp
    g++ -std=c++11 -O3 -o matrix_test_x86_O3 matrix_benchmark.cpp
    g++ -std=c++11 -O3 -march=native -o matrix_test_x86_O3_native matrix_test.cpp
    echo "编译完成"
}

# 运行测试并保存结果
run_tests() {
    echo "运行x86架构性能测试..."
    RESULT_FILE="$OUTPUT_DIR/performance_${ARCHITECTURE}.csv"
    
    # 创建CSV标题
    echo "架构,优化级别,矩阵大小,算法,执行时间(秒)" > $RESULT_FILE
    
    # 运行各种优化级别的测试
    for OPT in O0 O1 O2 O3 O3_native; do
        for SIZE in "${MATRIX_SIZES[@]}"; do
            echo "测试 $OPT 优化级别, 矩阵大小 $SIZE..."
            OUTPUT=$(./matrix_test_x86_$OPT $SIZE $ITERATIONS)
            
            # 提取算法执行时间
            ALG1_TIME=$(echo "$OUTPUT" | grep "算法1平均执行时间" | awk '{print $2}')
            ALG1_UNROLL_TIME=$(echo "$OUTPUT" | grep "算法1循环展开版本平均执行时间" | awk '{print $2}')
            ALG2_TIME=$(echo "$OUTPUT" | grep "算法2平均执行时间" | awk '{print $2}')
            ALG2_UNROLL_TIME=$(echo "$OUTPUT" | grep "算法2循环展开版本平均执行时间" | awk '{print $2}')
            
            # 写入CSV
            echo "$ARCHITECTURE,$OPT,$SIZE,算法1,$ALG1_TIME" >> $RESULT_FILE
            echo "$ARCHITECTURE,$OPT,$SIZE,算法1_展开,$ALG1_UNROLL_TIME" >> $RESULT_FILE
            echo "$ARCHITECTURE,$OPT,$SIZE,算法2,$ALG2_TIME" >> $RESULT_FILE
            echo "$ARCHITECTURE,$OPT,$SIZE,算法2_展开,$ALG2_UNROLL_TIME" >> $RESULT_FILE
            
            # 保存完整输出
            echo "$OUTPUT" > "$OUTPUT_DIR/${ARCHITECTURE}_${OPT}_${SIZE}.txt"
        done
    done
    
    echo "性能测试完成，结果保存在 $RESULT_FILE"
}

# 主流程
compile_programs
run_tests

echo "所有测试完成！"
echo "架构: $ARCHITECTURE"
echo "测试结果目录: $OUTPUT_DIR"
