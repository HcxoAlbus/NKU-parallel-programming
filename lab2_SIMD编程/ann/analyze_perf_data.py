import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

# --- 配置 ---
STAT_DIR = "perf_stat_results"   # perf stat 输出目录
RESULTS_FILE = "pq_results.csv" # C++ 输出的 CSV 文件
OUTPUT_DIR = "perf_analysis_charts" # 图表输出目录
REPORT_FILE = "perf_analysis_report.md" # Markdown 报告文件

# 从文件名提取参数的正则表达式
FILENAME_PATTERN = re.compile(r"perf_stat_nsub(\d+)_rerank(\d+)\.txt")

# --- 数据结构 ---
performance_data = defaultdict(list)

# --- 解析函数 ---

def parse_perf_stat_file(filepath):
    """解析单个 perf stat 输出文件"""
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
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE) # 添加 MULTILINE
            if match:
                try:
                    return int(match.group(1).replace(',', ''))
                except (ValueError, AttributeError): # 更健壮的错误处理
                    return np.nan
            return np.nan

        def extract_percentage(pattern):
             match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE) # 添加 MULTILINE
             if match:
                 try:
                     return float(match.group(1).replace(',', '')) # 处理百分号前的逗号
                 except (ValueError, AttributeError):
                     return np.nan
             return np.nan

        metrics["l1_loads"] = extract_value(r"^\s*([0-9,]+)\s+L1-dcache-loads") # 添加行首匹配和空格处理
        metrics["l1_misses"] = extract_value(r"^\s*([0-9,]+)\s+L1-dcache-load-misses")
        metrics["llc_loads"] = extract_value(r"^\s*([0-9,]+)\s+LLC-loads")
        metrics["llc_misses"] = extract_value(r"^\s*([0-9,]+)\s+LLC-load-misses")
        metrics["instructions"] = extract_value(r"^\s*([0-9,]+)\s+instructions")
        metrics["cycles"] = extract_value(r"^\s*([0-9,]+)\s+cycles")
        metrics["branches"] = extract_value(r"^\s*([0-9,]+)\s+(?:branch-instructions|branches)")
        metrics["branch_misses"] = extract_value(r"^\s*([0-9,]+)\s+branch-misses")

        metrics["l1_miss_rate"] = extract_percentage(r"^\s*([0-9.,]+)%\s+L1-dcache.*misses") # 处理逗号和小数点
        metrics["ipc"] = extract_percentage(r"^\s*([0-9.,]+)\s+insn\s+per\s+cycle")
        metrics["branch_miss_rate"] = extract_percentage(r"^\s*([0-9.,]+)%\s+branch-misses")

        # 尝试提取 LLC miss rate (如果 perf stat 直接提供)
        llc_miss_rate_direct = extract_percentage(r"^\s*([0-9.,]+)%\s+LLC.*misses")
        if not np.isnan(llc_miss_rate_direct):
             metrics["llc_miss_rate"] = llc_miss_rate_direct

        # --- 计算派生指标 (如果直接提取失败) ---
        if np.isnan(metrics["l1_miss_rate"]) and not np.isnan(metrics["l1_loads"]) and metrics["l1_loads"] > 0:
            metrics["l1_miss_rate"] = 100.0 * metrics["l1_misses"] / metrics["l1_loads"]

        # 修正 LLC Miss Rate 计算：如果 LLC-loads 无效，尝试使用 cache-references
        if np.isnan(metrics["llc_miss_rate"]):
            if not np.isnan(metrics["llc_loads"]) and metrics["llc_loads"] > 0:
                 metrics["llc_miss_rate"] = 100.0 * metrics["llc_misses"] / metrics["llc_loads"]
            else:
                 # 尝试备用事件 cache-references
                 cache_refs = extract_value(r"^\s*([0-9,]+)\s+cache-references")
                 cache_misses = extract_value(r"^\s*([0-9,]+)\s+cache-misses") # 通常 cache-misses 对应 LLC misses
                 if not np.isnan(cache_refs) and cache_refs > 0 and not np.isnan(cache_misses):
                     metrics["llc_miss_rate"] = 100.0 * cache_misses / cache_refs
                     # 更新原始计数值以便调试
                     metrics["llc_loads"] = cache_refs # 将 cache-references 视为 LLC loads
                     metrics["llc_misses"] = cache_misses

        if np.isnan(metrics["ipc"]) and not np.isnan(metrics["cycles"]) and metrics["cycles"] > 0:
            metrics["ipc"] = metrics["instructions"] / metrics["cycles"]
        if np.isnan(metrics["branch_miss_rate"]) and not np.isnan(metrics["branches"]) and metrics["branches"] > 0:
            metrics["branch_miss_rate"] = 100.0 * metrics["branch_misses"] / metrics["branches"]

    except FileNotFoundError:
        print(f"警告: 文件未找到 {filepath}")
    except Exception as e:
        print(f"解析文件 {filepath} 时出错: {e}")

    return metrics

# --- 主逻辑 ---
if __name__ == "__main__":
    # 1. 读取 Recall 和 Latency 数据
    try:
        df_results = pd.read_csv(RESULTS_FILE)
        df_results = df_results.dropna()
        if df_results.empty:
             print(f"错误: {RESULTS_FILE} 为空或无效。请先运行 profile_pq.sh。")
             exit(1)
        # ***确保结果文件中的键列是整数类型***
        df_results['nsub'] = df_results['nsub'].astype('int64')
        df_results['rerank_k'] = df_results['rerank_k'].astype('int64')
        # 计算 QPS 和召回率百分比
        df_results['qps'] = 1_000_000 / df_results['latency_us']
        df_results['recall_percent'] = df_results['recall'] * 100
    except FileNotFoundError:
        print(f"错误: {RESULTS_FILE} 未找到。请先运行 profile_pq.sh。")
        exit(1)
    except Exception as e:
        print(f"读取 {RESULTS_FILE} 时出错: {e}")
        exit(1)

    # 2. 解析所有 perf stat 文件
    perf_stats_list = []
    file_count = 0
    if not os.path.isdir(STAT_DIR):
        print(f"错误: perf stat 结果目录 '{STAT_DIR}' 不存在。")
        exit(1)

    for filename in os.listdir(STAT_DIR):
        match = FILENAME_PATTERN.match(filename)
        if match:
            file_count += 1
            nsub = int(match.group(1))
            rerank_k = int(match.group(2))
            filepath = os.path.join(STAT_DIR, filename)
            stats = parse_perf_stat_file(filepath)
            stats['nsub'] = nsub
            stats['rerank_k'] = rerank_k
            perf_stats_list.append(stats)

    if not perf_stats_list:
         print(f"错误: 未在目录 '{STAT_DIR}' 中找到有效的 perf stat 文件。")
         print("请确保 profile_pq.sh 已运行且生成了正确的 .txt 文件。")
         exit(1)

    df_perf = pd.DataFrame(perf_stats_list)
    # ***确保 perf 数据中的键列也是整数类型***
    df_perf['nsub'] = df_perf['nsub'].astype('int64')
    df_perf['rerank_k'] = df_perf['rerank_k'].astype('int64')


    # 3. 合并数据
    # 现在两个 DataFrame 的 'nsub' 和 'rerank_k' 都应该是 int64 类型
    try:
        df_merged = pd.merge(df_results, df_perf, on=['nsub', 'rerank_k'], how='left')
    except Exception as e:
        print("合并 DataFrame 时出错:")
        print("df_results dtypes:\n", df_results.dtypes)
        print("df_perf dtypes:\n", df_perf.dtypes)
        print("错误信息:", e)
        exit(1)

    # *** 在合并后，为了绘图，再将 nsub 转为字符串 ***
    df_merged['nsub_str'] = df_merged['nsub'].astype(str) # 使用新列进行绘图分组

    print("\n--- 合并后的性能数据 ---")
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    # 选择要显示的列
    display_cols = ['nsub', 'rerank_k', 'recall_percent', 'qps', 'ipc',
                    'l1_miss_rate', 'llc_miss_rate', 'branch_miss_rate']
    # 检查列是否存在
    existing_cols = [col for col in display_cols if col in df_merged.columns]
    print(df_merged[existing_cols].head())
    print("-" * (len(existing_cols) * 12)) # 调整分隔线长度


    # 4. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5. 可视化 (修改 hue 和 style 使用 'nsub_str')
    plt.style.use('seaborn-v0_8-whitegrid')

    plot_configs = [
        {'x': 'recall_percent', 'y': 'qps', 'y_label': 'QPS (Log Scale)', 'y_scale': 'log', 'filename': 'pq_recall_vs_qps.png', 'title': 'PQ: Recall vs QPS'},
        {'x': 'recall_percent', 'y': 'ipc', 'y_label': 'IPC', 'y_scale': 'linear', 'filename': 'pq_ipc_vs_recall.png', 'title': 'PQ: IPC vs Recall'},
        {'x': 'recall_percent', 'y': 'l1_miss_rate', 'y_label': 'L1-D Cache Miss Rate (%)', 'y_scale': 'linear', 'filename': 'pq_l1_miss_vs_recall.png', 'title': 'PQ: L1 Miss Rate vs Recall'},
        {'x': 'recall_percent', 'y': 'llc_miss_rate', 'y_label': 'LLC Miss Rate (%)', 'y_scale': 'linear', 'filename': 'pq_llc_miss_vs_recall.png', 'title': 'PQ: LLC Miss Rate vs Recall'},
        {'x': 'recall_percent', 'y': 'branch_miss_rate', 'y_label': 'Branch Miss Rate (%)', 'y_scale': 'linear', 'filename': 'pq_branch_miss_vs_recall.png', 'title': 'PQ: Branch Miss Rate vs Recall'},
    ]

    for config in plot_configs:
        if config['y'] not in df_merged.columns or df_merged[config['y']].isnull().all():
             print(f"警告: 跳过绘图 '{config['title']}'，因为列 '{config['y']}' 不存在或全为空值。")
             continue

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_merged, x=config['x'], y=config['y'], hue='nsub_str', style='nsub_str', # 使用 nsub_str
                     marker='o', markersize=8, palette='viridis', legend='full')

        if config['y_scale'] == 'log':
            plt.yscale('log')

        plt.title(f"{config['title']} (DEEP100K, k=10)", fontsize=16)
        plt.xlabel('Recall@10 (%)', fontsize=12)
        plt.ylabel(config['y_label'], fontsize=12)
        plt.legend(title='nsub', title_fontsize='13', fontsize='11')
        plt.grid(True, which='major', linestyle='--', linewidth=0.7)
        if config['y_scale'] == 'log':
             plt.grid(True, which='minor', linestyle=':', linewidth=0.5)
        else:
             plt.grid(False, which='minor') # 线性刻度通常不需要次网格

        plt.savefig(os.path.join(OUTPUT_DIR, config['filename']), dpi=300, bbox_inches='tight')
        plt.close() # 关闭图像，避免在循环中累积

    print(f"\n图表已保存到目录: {OUTPUT_DIR}")


    # 6. 生成 Markdown 分析报告 (使用 f-string 格式化 NaN)
    print(f"生成分析报告到: {REPORT_FILE} ...")
    try:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write("# PQ 算法 Perf 性能分析报告\n\n")
            # 尝试获取 k 值，如果失败则显示 N/A
            k_value = df_merged['recall'].iloc[0] if not df_merged.empty and 'recall' in df_merged.columns else 'N/A'
            f.write(f"数据集: DEEP100K, k={k_value}\n")
            f.write(f"测试时间: {pd.Timestamp.now()}\n\n")

            f.write("## 核心性能指标概览\n\n")
            # 定义报告中要包含的列
            report_cols = ['nsub', 'rerank_k', 'recall_percent', 'qps', 'ipc',
                           'l1_miss_rate', 'llc_miss_rate', 'branch_miss_rate']
            existing_report_cols = [col for col in report_cols if col in df_merged.columns]

            # 定义格式化函数处理 NaN
            def format_float(x):
                return f"{x:.3f}" if pd.notna(x) else "N/A"

            # 使用格式化函数生成 markdown 表格
           # f.write(df_merged[existing_report_cols].to_markdown(index=False, floatfmt=".3f", na_rep="N/A"))
            f.write(
                df_merged[existing_report_cols]
                .fillna("N/A")
                .to_markdown(index=False, floatfmt=".3f")
            )

            f.write("\n\n")

            f.write("## 性能分析\n\n")
            f.write("### 1. Recall vs. QPS (主要权衡)\n\n")
            f.write(f"![Recall vs QPS]({os.path.join(OUTPUT_DIR, 'pq_recall_vs_qps.png')})\n\n") # 使用相对路径或确保路径正确
            f.write("- **趋势**: 对所有 `nsub` 值，随着 `rerank_k` 增加（通常对应曲线上从左上到右下的点），召回率提高，但 QPS 下降。\n")
            f.write("- **`nsub` 影响**: 较高的 `nsub` (如 32) 通常能达到更高的召回率上限，但其 QPS 普遍低于较低的 `nsub` (如 8 或 16)。\n")
            f.write("- **`rerank_k` 影响**: 增加 `rerank_k` 对召回率的提升存在边际效益递减，但对 QPS 的负面影响（延迟增加）持续存在。\n")
            f.write("- **分析**: `rerank_k` 控制精确计算的范围，直接影响延迟。`nsub` 影响量化精度和 ADC 阶段的计算量。选择最佳配置需要在两者间权衡。\n\n")

            # 为其他图表添加分析（仅包含存在的图表）
            if os.path.exists(os.path.join(OUTPUT_DIR, 'pq_ipc_vs_recall.png')):
                f.write("### 2. IPC (Instructions Per Cycle)\n\n")
                f.write(f"![IPC vs Recall]({os.path.join(OUTPUT_DIR, 'pq_ipc_vs_recall.png')})\n\n")
                f.write("- **趋势**: IPC 随 `rerank_k` 增加（召回率提高）而变化的趋势可能不单一。\n")
                f.write("  - ADC 阶段（`rerank_k` 较小）可能因查表和整数运算有较高 IPC。\n")
                f.write("  - Rerank 阶段（`rerank_k` 较大）涉及浮点 SIMD 计算 (`compute_l2_sq_neon`)，其 IPC 可能不同，且受内存访问影响。\n")
                f.write("- **`nsub` 影响**: 不同 `nsub` 的 IPC 可能相似或有细微差别，取决于 ADC 和 Rerank 各阶段的相对耗时。\n")
                f.write("- **分析**: IPC 反映 CPU 执行效率。较低的 IPC 可能意味着更多的 CPU 停顿（例如等待内存）。PQ 算法混合了查表、整数和浮点 SIMD，IPC 是综合表现。\n\n")

            if os.path.exists(os.path.join(OUTPUT_DIR, 'pq_l1_miss_vs_recall.png')) and os.path.exists(os.path.join(OUTPUT_DIR, 'pq_llc_miss_vs_recall.png')):
                f.write("### 3. 缓存未命中率 (L1-D / LLC)\n\n")
                f.write(f"![L1 Miss Rate vs Recall]({os.path.join(OUTPUT_DIR, 'pq_l1_miss_vs_recall.png')})\n")
                f.write(f"![LLC Miss Rate vs Recall]({os.path.join(OUTPUT_DIR, 'pq_llc_miss_vs_recall.png')})\n\n")
                f.write("- **趋势**: \n")
                f.write("  - **L1-D Miss Rate**: 可能随 `rerank_k` 增加而略有变化。ADC 阶段访问 `distance_table` (小，可能 L1 命中率高)，Rerank 阶段访问原始 `base` 数据（大，可能 L1 命中率低）。\n")
                f.write("  - **LLC Miss Rate**: 主要受 Rerank 阶段影响，因为需要访问较大的原始 `base` 数据子集。随着 `rerank_k` 增加，访问的数据量略增，LLC Miss Rate 可能略微上升或保持平稳（取决于 `rerank_k` 对应的向量是否已在 LLC 中）。\n")
                f.write("- **`nsub` 影响**: `nsub` 对缓存命中率的直接影响可能不大，主要通过改变 ADC 和 Rerank 的相对耗时来间接影响。\n")
                f.write("- **分析**: 缓存性能对 PQ 很重要。ADC 阶段受益于小数据（代码、距离表）的高缓存命中率。Rerank 阶段则面临访问大原始数据的挑战。整体 Miss Rate 会低于 Flat/SIMD 搜索。\n\n")

            if os.path.exists(os.path.join(OUTPUT_DIR, 'pq_branch_miss_vs_recall.png')):
                f.write("### 4. 分支预测未命中率\n\n")
                f.write(f"![Branch Miss Rate vs Recall]({os.path.join(OUTPUT_DIR, 'pq_branch_miss_vs_recall.png')})\n\n")
                f.write("- **趋势**: 分支预测错误率通常与代码中条件分支（if/else, 循环）的预测难度有关。PQ 的核心循环（ADC, Rerank）结构相对固定，Branch Miss Rate 可能变化不大，或随 `rerank_k` 增加略有变化（Rerank 循环次数增加）。\n")
                f.write("- **分析**: 较低的分支预测错误率有助于维持流水线效率。如果此比率异常高，可能指示代码中存在难以预测的分支，可以考虑优化（如循环展开、分支避免）。\n\n")

            f.write("## 结论与建议\n\n")
            f.write("- PQ 算法通过调整 `nsub` 和 `rerank_k` 提供了灵活的性能-精度权衡。\n")
            f.write("- 增加 `rerank_k` 能显著提高召回率，但会增加延迟（降低 QPS），需关注边际效益。\n")
            f.write("- 增加 `nsub` 可能提高召回率上限，但会增加 ADC 阶段的计算开销。\n")
            f.write("- 缓存性能（尤其是 Rerank 阶段对原始数据的访问）和 CPU 执行效率（IPC）是影响最终性能的关键因素。\n")
            f.write("- 建议根据具体应用场景选择合适的 `nsub` 和 `rerank_k` 组合。例如，对于召回率要求极高的场景，选择 `nsub=16/32` 和较高的 `rerank_k`；对于延迟更敏感的场景，选择 `nsub=8/16` 和适中的 `rerank_k`。\n")

    except Exception as e:
        print(f"生成 Markdown 报告时出错: {e}")

    print("分析完成。")
