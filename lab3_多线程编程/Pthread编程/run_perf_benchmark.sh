#!/bin/bash

# --- 配置 ---
EXECUTABLE="./main" # 您的已编译的可执行文件
DATA_ROOT_PATH="./"     # 数据集根目录, 会作为第一个参数传递给 EXECUTABLE

PERF_EVENTS="cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,branch-instructions,branch-misses"
OUTPUT_DIR="./perf_results_per_algorithm"
mkdir -p "${OUTPUT_DIR}"

# --- 函数：运行 perf 并保存输出 ---
run_single_test_with_perf() {
    local test_tag="$1"         # 用于文件名的唯一标签, e.g., "ivf_nprobe4_clusters256"
    local method_arg="$2"       # 传递给 --method 的参数, e.g., "ivf"
    local additional_args="$3"  # 其他参数, e.g., "--nprobe 4 --clusters 256"

    echo "INFO: Running test: ${test_tag}"
    local perf_report_file="${OUTPUT_DIR}/perf_report_${test_tag}.txt"
    local program_stdout_file="${OUTPUT_DIR}/program_stdout_${test_tag}.txt"
    local program_stderr_file="${OUTPUT_DIR}/program_stderr_${test_tag}.txt"

    if [ ! -f "${EXECUTABLE}" ]; then
        echo "ERROR: Executable '${EXECUTABLE}' not found."
        exit 1
    fi

    # 构建传递给 ann_benchmark 的完整命令参数
    # 假设您的 ann_benchmark 总是先接收 DATA_ROOT_PATH，然后是方法和特定参数
    local executable_run_args="${DATA_ROOT_PATH} --method ${method_arg} ${additional_args}"

    echo "INFO: Perf command: perf stat -o \"${perf_report_file}\" -e \"${PERF_EVENTS}\" ${EXECUTABLE} ${executable_run_args}"
    
    perf stat -o "${perf_report_file}" -e "${PERF_EVENTS}" "${EXECUTABLE}" ${executable_run_args} > "${program_stdout_file}" 2> "${program_stderr_file}"
    
    local exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
        echo "WARNING: Execution for '${test_tag}' exited with code ${exit_code}. Check '${program_stderr_file}'."
    else
        echo "INFO: Test '${test_tag}' completed."
    fi
    echo "-----------------------------------------------------"
}

# --- 定义测试用例 ---
# 根据您的 parsed_program_data.csv 和 main.cc 的输出格式定义
# 格式: "TAG;METHOD_ARG;ADDITIONAL_ARGS_FOR_MAIN_CC"

TEST_CASES=(
    "flat;flat;"
    "simd;simd;"
    "pq_rerank600;pq;--rerank_k 600" # 假设您的 main.cc 支持 --rerank_k
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
for test_entry in "${TEST_CASES[@]}"; do
    IFS=';' read -r tag method_arg main_args <<< "$test_entry"
    run_single_test_with_perf "${tag}" "${method_arg}" "${main_args}"
done

echo "INFO: All per-algorithm profiling finished. Results are in '${OUTPUT_DIR}'."
echo "INFO: Next step: run the updated Python script 'visualize_per_algorithm_results.py'."