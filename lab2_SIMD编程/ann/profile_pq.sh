#!/bin/bash

# 确保脚本出错时退出
set -e

# --- 配置 ---
EXECUTABLE="./main" # 指向编译好的 C++ 程序
NSUB_VALUES=(4 8 16) # 要测试的 nsub 值 (确保能整除维度 96)
RERANK_VALUES=(10 50 100 200 400 600) # 要测试的 rerank_k 值
K_NEIGHBORS=10        # 搜索的近邻数 k
NUM_QUERIES=2000      # 测试的查询数量

STAT_DIR="perf_stat_results" # <--- 修改：存储 perf stat 输出的目录
RESULTS_FILE="pq_results.csv" # 存储召回率和延迟结果的 CSV 文件

# Perf 配置
# 选择你关心的事件，确保你的 CPU 支持这些事件
# 通用事件: cycles, instructions, L1-dcache-loads, L1-dcache-load-misses, LLC-loads, LLC-load-misses, branch-instructions, branch-misses
# 你可能需要根据 `perf list` 的输出来调整事件名称
EVENTS="cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,branch-instructions,branch-misses"

# --- 准备工作 ---
mkdir -p "$STAT_DIR" # <--- 修改目录名
# 初始化结果文件，写入表头
echo "nsub,rerank_k,recall,latency_us" > "$RESULTS_FILE"

# 检查可执行文件是否存在
if [ ! -x "$EXECUTABLE" ]; then
    echo "错误: 可执行文件 '$EXECUTABLE' 不存在或不可执行。"
    echo "请先编译 C++ 代码: g++ main.cpp -o main -g -O2 -fopenmp -lpthread -std=c++11"
    exit 1
fi

# --- 循环测试 ---
echo "开始 PQ 性能分析 (使用 perf stat)..."

for nsub in "${NSUB_VALUES[@]}"; do
    for rerank in "${RERANK_VALUES[@]}"; do
        echo "--------------------------------------------------"
        echo "测试配置: nsub=$nsub, rerank_k=$rerank"
        echo "--------------------------------------------------"

        # 1. 运行 perf stat 并保存输出
        PERF_STAT_FILE="$STAT_DIR/perf_stat_nsub${nsub}_rerank${rerank}.txt" # <--- 修改文件名
        echo "运行 perf stat, 输出到 $PERF_STAT_FILE ..."
        perf stat -e "$EVENTS" -o "$PERF_STAT_FILE" -- \
            "$EXECUTABLE" --nsub "$nsub" --rerank "$rerank" --k "$K_NEIGHBORS" --num_queries "$NUM_QUERIES" \
            > /dev/null # 将 C++ 程序的标准输出重定向 (CSV 输出在下面单独捕获)

        # 注意: perf stat 通常将结果输出到 stderr。-o 会重定向 stderr。
        # 如果 -o 不工作或你想同时看到程序的 stderr，可以这样：
        # perf stat -e "$EVENTS" -- \
        #     "$EXECUTABLE" --nsub "$nsub" --rerank "$rerank" --k "$K_NEIGHBORS" --num_queries "$NUM_QUERIES" \
        #     > /dev/null 2> "$PERF_STAT_FILE"

        echo "perf stat 完成."

        # 2. 再次运行程序以捕获性能指标 (Recall, Latency)
        echo "收集召回率和延迟..."
        # 追加 C++ 程序输出的 CSV 行到结果文件
        "$EXECUTABLE" --nsub "$nsub" --rerank "$rerank" --k "$K_NEIGHBORS" --num_queries "$NUM_QUERIES" \
            >> "$RESULTS_FILE"

        echo "配置 nsub=$nsub, rerank_k=$rerank 测试完成."
        echo ""
        sleep 1 # 短暂暂停

    done # 结束 rerank 循环
done # 结束 nsub 循环

echo "所有 PQ 测试完成。"
echo "性能指标保存在: $RESULTS_FILE"
echo "Perf stat 输出保存在: $STAT_DIR"