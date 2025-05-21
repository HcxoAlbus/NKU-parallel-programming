#!/bin/bash

# 确保脚本出错时退出
set -e

# --- 配置 ---
EXECUTABLE="./main"         # 指向编译好的 C++ 程序
DATA_ROOT_PATH="./"         # 数据集根目录 (作为第一个参数传递给 EXECUTABLE)

# Perf 事件集 (参考您的脚本，并确保CPU支持)
# perf list 查看可用事件
PERF_EVENTS="cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,branch-instructions,branch-misses"

OUTPUT_DIR_BASE="perf_analysis_results" # 基础输出目录
PERF_STAT_SUBDIR="${OUTPUT_DIR_BASE}/perf_stat_reports"
PROGRAM_METRICS_SUBDIR="${OUTPUT_DIR_BASE}/program_metrics"

# --- 准备工作 ---
mkdir -p "${PERF_STAT_SUBDIR}"
mkdir -p "${PROGRAM_METRICS_SUBDIR}"

# 检查可执行文件是否存在
if [ ! -x "$EXECUTABLE" ]; then
    echo "错误: 可执行文件 '$EXECUTABLE' 不存在或不可执行。"
    echo "请确保已编译 C++ 代码 (例如: g++ main.cc ... -o main) "
    echo "并且该程序能够通过命令行参数指定运行单个测试方法。"
    exit 1
fi

# --- 函数：运行单个测试 ---
run_single_test() {
    local test_tag="$1"         # 用于文件名的唯一标签, e.g., "ivf_np4_c256"
    local method_arg="$2"       # 传递给 --method 的参数, e.g., "ivf"
    local additional_args="$3"  # 其他参数, e.g., "--nprobe 4 --clusters 256"

    echo "INFO: === Testing: ${test_tag} ==="
    local perf_report_file="${PERF_STAT_SUBDIR}/perf_report_${test_tag}.txt"
    local program_metrics_file="${PROGRAM_METRICS_SUBDIR}/metrics_${test_tag}.txt"
    local program_stderr_perf_file="${PERF_STAT_SUBDIR}/stderr_perf_${test_tag}.txt"
    local program_stderr_metrics_file="${PROGRAM_METRICS_SUBDIR}/stderr_metrics_${test_tag}.txt"

    # 构建传递给 ann_benchmark 的完整命令参数
    # 假设您的 main 程序总是先接收 DATA_ROOT_PATH，然后是方法和特定参数
    local executable_full_args="${DATA_ROOT_PATH}"
    if [ -n "${method_arg}" ]; then # 如果 method_arg 不为空
        executable_full_args="${executable_full_args} --method ${method_arg}"
    fi
    executable_full_args="${executable_full_args} ${additional_args}"


    # 1. 运行 perf stat 并保存硬件计数器输出
    #    程序本身的 stdout 被丢弃，stderr 被重定向到 perf_report_file (与 perf stat 输出一起)
    echo "INFO: Running perf stat for hardware counters (output to ${perf_report_file})..."
    perf stat -e "${PERF_EVENTS}" -o "${perf_report_file}" -- \
        "${EXECUTABLE}" ${executable_full_args} \
        > /dev/null 2>> "${perf_report_file}" # 追加程序的stderr到perf报告，以防-o不捕获它
                                           # 或者，如果-o确实捕获了stderr, 2>/dev/null
    # perf stat 的主要输出会通过 -o 定向到 $perf_report_file。
    # 如果程序在perf stat运行时输出了到stderr，它也可能被 -o 捕获或需要单独处理。
    # 上面的 2>> "${perf_report_file}" 是一个尝试，确保任何来自程序的stderr（如果未被-o捕获）也进入该文件。
    # 一个更干净的方法可能是让程序在perf运行时完全静默，或将其stderr重定向到单独文件。
    # 例如: (perf stat -e "$PERF_EVENTS" -o "$perf_report_file" -- "$EXECUTABLE" $executable_full_args >/dev/null) 2> "$program_stderr_perf_file"
    # 但我们将遵循您脚本中 -o 的用法，并假设它能正确工作。

    # 2. 再次运行程序以捕获召回率和延迟 (程序输出到 program_metrics_file)
    echo "INFO: Running program for recall/latency metrics (output to ${program_metrics_file})..."
    "${EXECUTABLE}" ${executable_full_args} \
        > "${program_metrics_file}" 2> "${program_stderr_metrics_file}"
    
    echo "INFO: Test '${test_tag}' completed."
    echo "-----------------------------------------------------"
    sleep 1 # 短暂暂停
}

# --- 定义测试用例 ---
# 格式: "TAG;METHOD_ARG;ADDITIONAL_ARGS_FOR_MAIN_CC"
# 您需要根据您的 main 程序实际支持的参数来修改这些测试用例。
# 确保您的 main 程序在被调用时，例如 "./main ./ --method ivf --nprobe 1" 时，
# 只运行 IVF (nprobe=1) 的测试并输出其召回率和延迟。
TEST_CASES=(
    "flat;flat;"
    "simd;simd;"
    # "pq_default;pq;" # 假设您的 main 支持一个默认的 PQ 测试
    "pq_nsub4_rrk600;pq;--nsub 4 --rerank_k 600" # 示例：如果您的 main 支持这些参数
    "sq;sq;"

    "ivf_np1_c256;ivf;--nprobe 1 --clusters 256"
    "ivf_np2_c256;ivf;--nprobe 2 --clusters 256"
    "ivf_np4_c256;ivf;--nprobe 4 --clusters 256"
    "ivf_np8_c256;ivf;--nprobe 8 --clusters 256"
    "ivf_np16_c256;ivf;--nprobe 16 --clusters 256"
    "ivf_np32_c256;ivf;--nprobe 32 --clusters 256"

    "ivfadc_np1_ic64_pq4_rrk600;ivfadc;--nprobe 1 --ivf_clusters 64 --pq_nsub 4 --rerank_k 600"
    "ivfadc_np2_ic64_pq4_rrk600;ivfadc;--nprobe 2 --ivf_clusters 64 --pq_nsub 4 --rerank_k 600"
    "ivfadc_np4_ic64_pq4_rrk600;ivfadc;--nprobe 4 --ivf_clusters 64 --pq_nsub 4 --rerank_k 600"
    "ivfadc_np8_ic64_pq4_rrk600;ivfadc;--nprobe 8 --ivf_clusters 64 --pq_nsub 4 --rerank_k 600"
    "ivfadc_np16_ic64_pq4_rrk600;ivfadc;--nprobe 16 --ivf_clusters 64 --pq_nsub 4 --rerank_k 600"
)

# --- 主逻辑 ---
echo "开始详细性能分析..."
for test_entry in "${TEST_CASES[@]}"; do
    IFS=';' read -r tag method_arg main_args <<< "$test_entry"
    # 确保 method_arg 和 main_args 被正确传递
    run_single_test "${tag}" "${method_arg}" "${main_args}"
done

echo "所有测试完成。"
echo "Perf stat 输出保存在: ${PERF_STAT_SUBDIR}"
echo "程序召回率/延迟输出保存在: ${PROGRAM_METRICS_SUBDIR}"
echo "下一步: 运行 Python 脚本 'visualize_detailed_results.py' (您可能需要调整或使用之前的脚本) 来解析和绘图。"