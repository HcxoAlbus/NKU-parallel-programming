import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

# --- 配置 ---
STAT_DIR = "perf_stat_other_results"   # perf stat 输出目录
RESULTS_FILE = "other_results.csv" # C++ 输出的 CSV 文件
OUTPUT_DIR = "perf_analysis_other_charts" # 图表输出目录
REPORT_FILE = "perf_analysis_others_report.md" # Markdown 报告文件

# 从文件名提取参数的正则表达式 (更简单)
FILENAME_PATTERN = re.compile(r"perf_stat_(flat|simd|sq)\.txt")

# --- 复用之前的解析函数 ---
def parse_perf_stat_file(filepath):
    """解析单个 perf stat 输出文件 (与 analyze_perf_data.py 中的版本相同或类似)"""
    metrics = {
        "l1_loads": np.nan, "l1_misses": np.nan,
        "llc_loads": np.nan, "llc_misses": np.nan,
        "instructions": np.nan, "cycles": np.nan,
        "branches": np.nan, "branch_misses": np.nan,
        "l1_miss_rate": np.nan, "llc_miss_rate": np.nan,
        "ipc": np.nan, "branch_miss_rate": np.nan
    }
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        def extract_value(pattern):
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                try: return int(match.group(1).replace(',', ''))
                except (ValueError, AttributeError): return np.nan
            return np.nan

        def extract_percentage(pattern):
             match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
             if match:
                 try: return float(match.group(1).replace(',', ''))
                 except (ValueError, AttributeError): return np.nan
             return np.nan

        metrics["l1_loads"] = extract_value(r"^\s*([0-9,]+)\s+L1-dcache-loads")
        metrics["l1_misses"] = extract_value(r"^\s*([0-9,]+)\s+L1-dcache-load-misses")
        metrics["llc_loads"] = extract_value(r"^\s*([0-9,]+)\s+LLC-loads")
        metrics["llc_misses"] = extract_value(r"^\s*([0-9,]+)\s+LLC-load-misses")
        metrics["instructions"] = extract_value(r"^\s*([0-9,]+)\s+instructions")
        metrics["cycles"] = extract_value(r"^\s*([0-9,]+)\s+cycles")
        metrics["branches"] = extract_value(r"^\s*([0-9,]+)\s+(?:branch-instructions|branches)")
        metrics["branch_misses"] = extract_value(r"^\s*([0-9,]+)\s+branch-misses")

        metrics["l1_miss_rate"] = extract_percentage(r"^\s*([0-9.,]+)%\s+L1-dcache.*misses")
        metrics["ipc"] = extract_percentage(r"^\s*([0-9.,]+)\s+insn\s+per\s+cycle")
        metrics["branch_miss_rate"] = extract_percentage(r"^\s*([0-9.,]+)%\s+branch-misses")
        llc_miss_rate_direct = extract_percentage(r"^\s*([0-9.,]+)%\s+LLC.*misses")
        if not np.isnan(llc_miss_rate_direct): metrics["llc_miss_rate"] = llc_miss_rate_direct

        # --- 计算派生指标 ---
        if np.isnan(metrics["l1_miss_rate"]) and not np.isnan(metrics["l1_loads"]) and metrics["l1_loads"] > 0:
            metrics["l1_miss_rate"] = 100.0 * metrics["l1_misses"] / metrics["l1_loads"]
        if np.isnan(metrics["llc_miss_rate"]):
            if not np.isnan(metrics["llc_loads"]) and metrics["llc_loads"] > 0:
                 metrics["llc_miss_rate"] = 100.0 * metrics["llc_misses"] / metrics["llc_loads"]
            else:
                 cache_refs = extract_value(r"^\s*([0-9,]+)\s+cache-references")
                 cache_misses = extract_value(r"^\s*([0-9,]+)\s+cache-misses")
                 if not np.isnan(cache_refs) and cache_refs > 0 and not np.isnan(cache_misses):
                     metrics["llc_miss_rate"] = 100.0 * cache_misses / cache_refs
                     metrics["llc_loads"] = cache_refs
                     metrics["llc_misses"] = cache_misses
        if np.isnan(metrics["ipc"]) and not np.isnan(metrics["cycles"]) and metrics["cycles"] > 0:
            metrics["ipc"] = metrics["instructions"] / metrics["cycles"]
        if np.isnan(metrics["branch_miss_rate"]) and not np.isnan(metrics["branches"]) and metrics["branches"] > 0:
            metrics["branch_miss_rate"] = 100.0 * metrics["branch_misses"] / metrics["branches"]

    except FileNotFoundError: print(f"警告: 文件未找到 {filepath}")
    except Exception as e: print(f"解析文件 {filepath} 时出错: {e}")
    return metrics

# --- 主逻辑 ---
if __name__ == "__main__":
    # 1. 读取 Recall 和 Latency 数据
    try:
        df_results = pd.read_csv(RESULTS_FILE)
        df_results = df_results.dropna()
        if df_results.empty:
             print(f"错误: {RESULTS_FILE} 为空或无效。请先运行 profile_others.sh。")
             exit(1)
        # 计算 QPS 和召回率百分比
        df_results['qps'] = 1_000_000 / df_results['latency_us']
        df_results['recall_percent'] = df_results['recall'] * 100
    except FileNotFoundError:
        print(f"错误: {RESULTS_FILE} 未找到。请先运行 profile_others.sh。")
        exit(1)
    except Exception as e:
        print(f"读取 {RESULTS_FILE} 时出错: {e}")
        exit(1)

    # 2. 解析所有 perf stat 文件
    perf_stats_list = []
    if not os.path.isdir(STAT_DIR):
        print(f"错误: perf stat 结果目录 '{STAT_DIR}' 不存在。")
        exit(1)

    for filename in os.listdir(STAT_DIR):
        match = FILENAME_PATTERN.match(filename)
        if match:
            algo_name = match.group(1)
            filepath = os.path.join(STAT_DIR, filename)
            stats = parse_perf_stat_file(filepath)
            stats['algorithm'] = algo_name # 添加算法名称键
            perf_stats_list.append(stats)

    if not perf_stats_list:
         print(f"错误: 未在目录 '{STAT_DIR}' 中找到有效的 perf stat 文件。")
         exit(1)

    df_perf = pd.DataFrame(perf_stats_list)

    # 3. 合并数据 (基于 'algorithm' 列)
    try:
        # 确保两个 df 的 'algorithm' 列都是 object/string 类型
        df_results['algorithm'] = df_results['algorithm'].astype(str)
        df_perf['algorithm'] = df_perf['algorithm'].astype(str)
        df_merged = pd.merge(df_results, df_perf, on='algorithm', how='left')
    except Exception as e:
        print("合并 DataFrame 时出错:")
        print("df_results dtypes:\n", df_results.dtypes)
        print("df_perf dtypes:\n", df_perf.dtypes)
        print("错误信息:", e)
        exit(1)

    print("\n--- 合并后的性能数据 ---")
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    display_cols = ['algorithm', 'recall_percent', 'qps', 'ipc',
                    'l1_miss_rate', 'llc_miss_rate', 'branch_miss_rate']
    existing_cols = [col for col in display_cols if col in df_merged.columns]
    print(df_merged[existing_cols])
    print("-" * (len(existing_cols) * 12))


    # 4. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5. 可视化 (使用条形图比较)
    plt.style.use('seaborn-v0_8-whitegrid')
    algo_order = ['flat', 'simd', 'sq'] # 定义条形图顺序
    # 确保数据按此顺序排列，方便绘图
    try:
        df_merged['algorithm'] = pd.Categorical(df_merged['algorithm'], categories=algo_order, ordered=True)
        df_merged = df_merged.sort_values('algorithm')
    except Exception as e:
        print(f"警告: 设置算法顺序时出错 - {e}. 图表可能顺序不一致。")
        algo_order = df_merged['algorithm'].unique() # 使用实际顺序


    plot_configs_bar = [
        # {'y': 'recall_percent', 'y_label': 'Recall@10 (%)', 'filename': 'other_recall.png', 'title': 'Algorithm Comparison: Recall'}, # Recall 都是 1.0 或接近，条形图意义不大
        {'y': 'qps', 'y_label': 'QPS (Queries Per Second)', 'filename': 'other_qps.png', 'title': 'Algorithm Comparison: QPS'},
        {'y': 'ipc', 'y_label': 'Instructions Per Cycle (IPC)', 'filename': 'other_ipc.png', 'title': 'Algorithm Comparison: IPC'},
        {'y': 'l1_miss_rate', 'y_label': 'L1-D Cache Miss Rate (%)', 'filename': 'other_l1_miss.png', 'title': 'Algorithm Comparison: L1 Miss Rate'},
        {'y': 'llc_miss_rate', 'y_label': 'LLC Miss Rate (%)', 'filename': 'other_llc_miss.png', 'title': 'Algorithm Comparison: LLC Miss Rate'},
        {'y': 'branch_miss_rate', 'y_label': 'Branch Miss Rate (%)', 'filename': 'other_branch_miss.png', 'title': 'Algorithm Comparison: Branch Miss Rate'},
    ]

    for config in plot_configs_bar:
        if config['y'] not in df_merged.columns or df_merged[config['y']].isnull().all():
             print(f"警告: 跳过绘图 '{config['title']}'，因为列 '{config['y']}' 不存在或全为空值。")
             continue

        plt.figure(figsize=(8, 5)) # 条形图通常不需要太大
        barplot = sns.barplot(data=df_merged, x='algorithm', y=config['y'], palette='viridis', order=algo_order)

        # 在条形图顶部添加数值标签
        for container in barplot.containers:
             barplot.bar_label(container, fmt='%.3f')

        plt.title(config['title'], fontsize=14)
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel(config['y_label'], fontsize=12)
        plt.xticks(rotation=0) # 保持标签水平
        plt.tight_layout() # 调整布局防止标签重叠
        plt.savefig(os.path.join(OUTPUT_DIR, config['filename']), dpi=300)
        plt.close()

    print(f"\n图表已保存到目录: {OUTPUT_DIR}")


    # 6. 生成 Markdown 分析报告
    print(f"生成分析报告到: {REPORT_FILE} ...")
    try:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write("# Flat, SIMD, SQ 算法 Perf 性能分析报告\n\n")
            k_value = df_merged['recall'].iloc[0] if not df_merged.empty and 'recall' in df_merged.columns else 'N/A' # Assume k is constant
            f.write(f"数据集: DEEP100K, k={k_value}\n")
            f.write(f"测试时间: {pd.Timestamp.now()}\n\n")

            f.write("## 核心性能指标概览\n\n")
            report_cols = ['algorithm', 'recall_percent', 'qps', 'ipc',
                           'l1_miss_rate', 'llc_miss_rate', 'branch_miss_rate']
            existing_report_cols = [col for col in report_cols if col in df_merged.columns]
            f.write(df_merged[existing_report_cols].to_markdown(index=False, floatfmt=".3f", na_rep="N/A"))
            f.write("\n\n")

            f.write("## 性能分析 (基于条形图)\n\n")

            # --- QPS 分析 ---
            f.write("### 1. QPS (查询吞吐量)\n\n")
            if os.path.exists(os.path.join(OUTPUT_DIR, 'other_qps.png')):
                f.write(f"![QPS Comparison]({os.path.join(OUTPUT_DIR, 'other_qps.png')})\n\n")
            f.write("- **观察**: SQ > SIMD > Flat。\n")
            f.write("- **分析**: \n")
            f.write("  - `Flat` 最慢，因为它执行 O(N*D) 的标量浮点运算，计算量最大。\n")
            f.write("  - `SIMD` 利用 NEON 指令并行处理浮点运算，显著减少了计算时间，因此 QPS 大幅高于 `Flat`。\n")
            f.write("  - `SQ` 最快，因为它将数据量化为 `uint8_t`，并使用更快的整数 SIMD 指令进行近似距离计算，计算开销最小。\n\n")

            # --- IPC 分析 ---
            f.write("### 2. IPC (每周期指令数)\n\n")
            if os.path.exists(os.path.join(OUTPUT_DIR, 'other_ipc.png')):
                f.write(f"![IPC Comparison]({os.path.join(OUTPUT_DIR, 'other_ipc.png')})\n\n")
            f.write("- **观察**: SQ 通常具有最高的 IPC，SIMD 次之，Flat 可能最低。\n")
            f.write("- **分析**: \n")
            f.write("  - `SQ` 使用整数 SIMD，这类指令通常延迟较低，且数据依赖性可能较简单，有利于提高 IPC。\n")
            f.write("  - `SIMD` 使用浮点 SIMD，虽然并行度高，但浮点运算延迟可能略长于整数运算。\n")
            f.write("  - `Flat` 执行标量运算，指令级并行性受限，且可能更容易受到内存延迟的影响（如果内存是瓶颈），导致 IPC 较低。\n")
            f.write("  - 高 IPC 表明 CPU 流水线利用率高，停顿少。\n\n")

            # --- L1/LLC 缓存未命中率分析 ---
            f.write("### 3. 缓存未命中率 (L1-D / LLC)\n\n")
            if os.path.exists(os.path.join(OUTPUT_DIR, 'other_l1_miss.png')):
                f.write(f"![L1 Miss Rate Comparison]({os.path.join(OUTPUT_DIR, 'other_l1_miss.png')})\n")
            if os.path.exists(os.path.join(OUTPUT_DIR, 'other_llc_miss.png')):
                f.write(f"![LLC Miss Rate Comparison]({os.path.join(OUTPUT_DIR, 'other_llc_miss.png')})\n\n")
            f.write("- **观察**: Flat 和 SIMD 的缓存未命中率（尤其是 LLC）可能显著高于 SQ。\n")
            f.write("- **分析**: \n")
            f.write("  - `Flat` 和 `SIMD` 都需要访问完整的、未压缩的 `base` 数据（约 37MB）。这个大小通常远超 L1/L2 缓存，甚至可能超过 LLC，导致频繁的缓存未命中和对主内存的访问。\n")
            f.write("  - `SQ` 访问的是量化后的数据 `quantized_base`（约 9.2MB）。虽然也可能超过 L1/L2，但比原始数据小得多，更容易驻留在 LLC 中，从而显著降低 LLC 未命中率。\n")
            f.write("  - L1 未命中率可能差异相对较小，因为三种算法内部计算时都存在一定的数据局部性（例如，SIMD 一次处理 8 个 float），但访问不同基向量时仍会发生 L1 miss。\n")
            f.write("  - 缓存效率是 SQ 相对于 Flat/SIMD 速度优势的重要原因之一。\n\n")

            # --- 分支预测未命中率分析 ---
            f.write("### 4. 分支预测未命中率\n\n")
            if os.path.exists(os.path.join(OUTPUT_DIR, 'other_branch_miss.png')):
                f.write(f"![Branch Miss Rate Comparison]({os.path.join(OUTPUT_DIR, 'other_branch_miss.png')})\n\n")
            f.write("- **观察**: 三种算法的分支预测未命中率可能都比较低，差异可能不大。\n")
            f.write("- **分析**: \n")
            f.write("  - 这些算法的核心都是密集的循环计算（内积或量化距离），循环本身的控制流相对简单且可预测。\n")
            f.write("  - 主要的分支发生在外部循环（遍历基向量）和优先队列维护（如果 `k` 较小，这部分开销占比小）。\n")
            f.write("  - 低的分支预测错误率表明 CPU 流水线因控制流转移而中断的情况较少。\n\n")

            f.write("## 结论与建议\n\n")
            f.write("- **性能排序**: 在速度（QPS）方面，通常 SQ > SIMD > Flat。\n")
            f.write("- **原因**: \n")
            f.write("  - SIMD 通过指令级并行加速计算，优于 Flat。\n")
            f.write("  - SQ 通过数据量化（减少内存占用和带宽需求，提高缓存效率）和更快的整数 SIMD 计算获得最大加速，但牺牲了精度（召回率）。\n")
            f.write("- **瓶颈**: \n")
            f.write("  - Flat 和 SIMD 的主要瓶颈在于计算量和访问大尺寸原始数据的内存带宽/延迟（体现在较高的 LLC 未命中率）。\n")
            f.write("  - SQ 的瓶颈主要在于其固有的量化误差限制了召回率，其计算和内存访问效率相对较高。\n")
            f.write("- **选择**: \n")
            f.write("  - 如果必须保证 100% 召回率，SIMD 是比 Flat 快得多的选择。\n")
            f.write("  - 如果可以接受一定的召回率损失以换取极高的速度，SQ 是最佳选择。\n")
            f.write("  - （与 PQ 对比）PQ 通过参数调整，提供了介于 SQ 和 SIMD/Flat 之间的、更灵活的精度-速度权衡。\n")

    except Exception as e:
        print(f"生成 Markdown 报告时出错: {e}")

    print("分析完成。")
