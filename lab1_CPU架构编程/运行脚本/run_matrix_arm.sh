#!/bin/bash

# 设置测试参数
MATRIX_SIZES=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300
1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700 
2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000)
ITERATIONS=10
TARGET_ARCH="arm64"
HOST_ARCH=$(uname -m)
OUTPUT_DIR="results_${TARGET_ARCH}_emulated"

# ARM交叉编译器
ARM_CROSS_COMPILER="aarch64-linux-gnu-g++"
QEMU_USER="qemu-aarch64"

# 检查依赖是否安装
check_dependencies() {
    echo "检查必要依赖..."
    
    # 检查交叉编译器
    if ! command -v $ARM_CROSS_COMPILER &> /dev/null; then
        echo "错误：ARM交叉编译器未安装。请运行以下命令安装："
        echo "sudo apt update && sudo apt install g++-aarch64-linux-gnu"
        exit 1
    fi
    
    # 检查QEMU用户模式模拟器
    if ! command -v $QEMU_USER &> /dev/null; then
        echo "错误：QEMU用户模式模拟器未安装。请运行以下命令安装："
        echo "sudo apt update && sudo apt install qemu-user qemu-user-static"
        exit 1
    fi
    
    echo "所有依赖已安装"
}

# 创建结果目录
mkdir -p $OUTPUT_DIR

# 编译不同优化级别的程序
compile_programs() {
    echo "使用ARM交叉编译器编译测试程序(静态链接)..."
    $ARM_CROSS_COMPILER -std=c++11 -O0 -static -o matrix_test_O0 matrix_benchmark.cpp
    $ARM_CROSS_COMPILER -std=c++11 -O1 -static -o matrix_test_O1 matrix_benchmark.cpp
    $ARM_CROSS_COMPILER -std=c++11 -O2 -static -o matrix_test_O2 matrix_benchmark.cpp
    $ARM_CROSS_COMPILER -std=c++11 -O3 -static -o matrix_test_O3 matrix_benchmark.cpp
    $ARM_CROSS_COMPILER -std=c++11 -O3 -mcpu=cortex-a72 -static -o matrix_test_O3_native matrix_benchmark.cpp
    echo "编译完成"
}

# 运行测试并保存结果
run_tests() {
    echo "运行QEMU模拟的ARM架构性能测试..."
    RESULT_FILE="$OUTPUT_DIR/performance_${TARGET_ARCH}_emulated.csv"
    
    # 创建CSV标题
    echo "架构,优化级别,矩阵大小,算法,执行时间(秒)" > $RESULT_FILE
    
    # 运行各种优化级别的测试
    for OPT in O0 O1 O2 O3 O3_native; do
        for SIZE in "${MATRIX_SIZES[@]}"; do
            echo "测试 $OPT 优化级别, 矩阵大小 $SIZE..."
            # 使用QEMU运行ARM二进制文件
            OUTPUT=$($QEMU_USER ./matrix_test_$OPT $SIZE $ITERATIONS)
            
            # 提取算法执行时间
            ALG1_TIME=$(echo "$OUTPUT" | grep "算法1平均执行时间" | awk '{print $2}')
            ALG1_UNROLL_TIME=$(echo "$OUTPUT" | grep "算法1循环展开版本平均执行时间" | awk '{print $2}')
            ALG2_TIME=$(echo "$OUTPUT" | grep "算法2平均执行时间" | awk '{print $2}')
            ALG2_UNROLL_TIME=$(echo "$OUTPUT" | grep "算法2循环展开版本平均执行时间" | awk '{print $2}')
            
            # 写入CSV
            echo "$TARGET_ARCH,$OPT,$SIZE,算法1,$ALG1_TIME" >> $RESULT_FILE
            echo "$TARGET_ARCH,$OPT,$SIZE,算法1_展开,$ALG1_UNROLL_TIME" >> $RESULT_FILE
            echo "$TARGET_ARCH,$OPT,$SIZE,算法2,$ALG2_TIME" >> $RESULT_FILE
            echo "$TARGET_ARCH,$OPT,$SIZE,算法2_展开,$ALG2_UNROLL_TIME" >> $RESULT_FILE
            
            # 保存完整输出
            echo "$OUTPUT" > "$OUTPUT_DIR/${TARGET_ARCH}_${OPT}_${SIZE}.txt"
        done
    done
    
    echo "性能测试完成，结果保存在 $RESULT_FILE"
}

# 主流程
check_dependencies
compile_programs
run_tests

echo "所有测试完成！"
echo "模拟架构: $TARGET_ARCH"
echo "主机架构: $HOST_ARCH"
echo "测试结果目录: $OUTPUT_DIR"