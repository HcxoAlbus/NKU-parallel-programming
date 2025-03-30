#!/bin/bash

# 设置变量
SOURCE_FILE="matrix_operations.cpp"
MATRIX_PROGRAM="matrix_operations.cpp"  # 你的源程序文件

# 检查程序文件是否存在
if [ ! -f "$MATRIX_PROGRAM" ]; then
    echo "错误: 找不到程序文件 $MATRIX_PROGRAM"
    exit 1
fi

# 检查perf是否可用
if ! command -v perf &> /dev/null; then
    echo "错误: 找不到perf命令。请确保perf已正确安装。"
    echo "提示: 在WSL中，可以使用以下命令安装perf:"
    echo "      sudo apt-get update && sudo apt-get install linux-tools-common linux-tools-generic"
    exit 1
fi

# 复制矩阵操作程序到当前目录
cp "$MATRIX_PROGRAM" "$SOURCE_FILE"

# 运行perf分析脚本
echo "开始perf性能分析..."
chmod +x perf_analyze.sh
./perf_analyze.sh

# 运行分析结果处理脚本
echo "处理perf分析结果..."
python3 analyze_perf_results.py

echo "perf分析完成！"
echo "详细结果保存在perf_results目录"
echo "图表保存在perf_charts目录"
echo "分析报告保存在perf_analysis_report.md文件"

# 提示如何进一步分析
echo ""
echo "提示: 如需查看更详细的性能剖析，可以使用以下命令查看火焰图或调用图:"
echo "flamegraph生成: cd perf_results && perf script -i size_*/perf_*.data | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg"
echo "热点函数详细分析: perf report -i perf_results/size_*/perf_*.data"