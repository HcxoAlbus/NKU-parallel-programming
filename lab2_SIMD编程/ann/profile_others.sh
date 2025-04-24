#!/bin/bash

# 确保脚本出错时退出
set -e

# --- 配置 ---
EXECUTABLE="./main" # 指向编译好的 C++ 程序
ALGORITHMS=("flat" "simd" "sq") # 要测试的算法
K_NEIGHBORS=10        # 搜索的近邻数 k
NUM_QUERIES=2000      # 测试的查询数量

STAT_DIR="perf_stat_other_results" # 存储 perf stat 输出的目录
RESULTS_FILE="other_results.csv" # 存储召回率和延迟结果的 CSV 文件

# Perf 配置 (与之前相同，确保事件有效)
EVENTS="cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,branch-instructions,branch-misses"

# --- 准备工作 ---
mkdir -p "$STAT_DIR"
# 初始化结果文件，写入表头
echo "algorithm,recall,latency_us" > "$RESULTS_FILE"

# 检查可执行文件是否存在
if [ ! -x "$EXECUTABLE" ]; then
    echo "错误: 可执行文件 '$EXECUTABLE' 不存在或不可执行。"
    echo "请先编译 C++ 代码: g++ main.cpp -o main -g -O2 -fopenmp -lpthread -std=c++11"
    exit 1
fi

# --- 循环测试 ---
echo "开始 Flat, SIMD, SQ 性能分析 (使用 perf stat)..."

for algo in "${ALGORITHMS[@]}"; do
    echo "--------------------------------------------------"
    echo "测试算法: $algo"
    echo "--------------------------------------------------"

    # 1. 运行 perf stat 并保存输出
    PERF_STAT_FILE="$STAT_DIR/perf_stat_${algo}.txt"
    echo "运行 perf stat, 输出到 $PERF_STAT_FILE ..."
    perf stat -e "$EVENTS" -o "$PERF_STAT_FILE" -- \
        "$EXECUTABLE" --algo "$algo" --k "$K_NEIGHBORS" --num_queries "$NUM_QUERIES" \
        > /dev/null # 重定向 C++ 程序的标准输出 (CSV 在下面捕获)

    # 检查 perf stat 是否成功 (可选，检查退出码)
    if [ $? -ne 0 ]; then
        echo "警告: perf stat 运行算法 '$algo' 时可能失败。检查 $PERF_STAT_FILE。"
        # 可以选择跳过或继续
    fi
    echo "perf stat 完成."

    # 2. 再次运行程序以捕获性能指标 (Recall, Latency)
    echo "收集召回率和延迟..."
    # 追加 C++ 程序输出的 CSV 行到结果文件
    "$EXECUTABLE" --algo "$algo" --k "$K_NEIGHBORS" --num_queries "$NUM_QUERIES" \
        >> "$RESULTS_FILE"

    echo "算法 $algo 测试完成."
    echo ""
    sleep 1 # 短暂暂停

done # 结束 algo 循环

echo "所有算法测试完成。"
echo "性能指标保存在: $RESULTS_FILE"
echo "Perf stat 输出保存在: $STAT_DIR"