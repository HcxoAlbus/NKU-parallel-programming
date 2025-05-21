import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# --- 配置 ---
RESULTS_DIR = "./perf_analysis_results/perf_stat_reports" # 修改为新的结果目录
OUTPUT_PLOTS_DIR = "./plots_per_algorithm"
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# (parse_perf_stat 函数与之前版本相同，这里省略以保持简洁，请从之前的回答中复制过来)
def parse_perf_stat(filepath): # 修改此函数
    perf_data = {
        "IPC": None, "L1_dcache_load_miss_rate": None, "branch_miss_rate": None,
        "cycles": None, "instructions": None,
        "L1_dcache_loads": None, "L1_dcache_load_misses": None,
        "LLC_loads": None, "LLC_load_misses": None, "LLC_load_miss_rate": None, # 新增 LLC
        "branch_instructions": None, "branch_misses": None,
    }
    if not os.path.exists(filepath):
        print(f"ERROR: Perf stat file '{filepath}' not found.")
        return perf_data
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read() # Read all content for easier multiline regex if needed

        # Regex to find value and event name, and optionally the percentage for misses
        # Handles cases like:
        #    1,234,567,890      cycles                    #    1.23  GHz
        #      123,456,789      L1-dcache-load-misses     #    5.67% of all L1-dcache accesses
        #      123,456,789      LLC-load-misses           #    0.12% of all LL-cache accesses  (LLC sometimes named LL-cache)
        #      123,456,789      instructions              #    1.50  insn per cycle
        
        patterns = {
            "cycles": r"([\d,]+)\s+cycles",
            "instructions": r"([\d,]+)\s+instructions(?:.*#\s*([\d\.]+)\s*insn per cycle)?", # Optional IPC
            "L1_dcache_loads": r"([\d,]+)\s+L1-dcache-loads",
            "L1_dcache_load_misses": r"([\d,]+)\s+L1-dcache-load-misses(?:.*#\s*([\d\.]+)\%)?", # Optional miss rate
            "LLC_loads": r"([\d,]+)\s+(?:LLC-loads|LL-cache loads)", # perf output can vary
            "LLC_load_misses": r"([\d,]+)\s+(?:LLC-load-misses|LL-cache misses)(?:.*#\s*([\d\.]+)\%)?", # Optional miss rate
            "branch_instructions": r"([\d,]+)\s+(?:branch-instructions|branches)",
            "branch_misses": r"([\d,]+)\s+branch-misses(?:.*#\s*([\d\.]+)\%)?", # Optional miss rate
        }

        for key, pattern_str in patterns.items():
            match = re.search(pattern_str, content, re.MULTILINE)
            if match:
                value_str = match.group(1).replace(',', '')
                perf_data[key] = int(value_str)
                if key == "instructions" and match.group(2):
                    perf_data["IPC"] = float(match.group(2))
                elif (key == "L1_dcache_load_misses" or \
                      key == "LLC_load_misses" or \
                      key == "branch_misses") and \
                     match.group(2):
                    rate_key = key.replace("_misses", "_miss_rate")
                    if key == "L1_dcache_load_misses": rate_key = "L1_dcache_load_miss_rate" # explicit
                    perf_data[rate_key] = float(match.group(2)) / 100.0
            
        # Fallback calculations if percentages/IPC not directly parsed by regex
        if perf_data["IPC"] is None and perf_data["instructions"] and perf_data["cycles"] and perf_data["cycles"] > 0:
            perf_data["IPC"] = perf_data["instructions"] / perf_data["cycles"]
        
        if perf_data.get("L1_dcache_load_miss_rate") is None and \
           perf_data["L1_dcache_load_misses"] is not None and \
           perf_data["L1_dcache_loads"] is not None and perf_data["L1_dcache_loads"] > 0:
            perf_data["L1_dcache_load_miss_rate"] = perf_data["L1_dcache_load_misses"] / perf_data["L1_dcache_loads"]
        
        if perf_data.get("LLC_load_miss_rate") is None and \
           perf_data["LLC_load_misses"] is not None and \
           perf_data["LLC_loads"] is not None and perf_data["LLC_loads"] > 0:
            perf_data["LLC_load_miss_rate"] = perf_data["LLC_load_misses"] / perf_data["LLC_loads"]

        if perf_data.get("branch_miss_rate") is None and \
           perf_data["branch_misses"] is not None and \
           perf_data["branch_instructions"] is not None and perf_data["branch_instructions"] > 0:
            perf_data["branch_miss_rate"] = perf_data["branch_misses"] / perf_data["branch_instructions"]

    except Exception as e:
        print(f"Error parsing perf stat file '{filepath}': {e}")
    return perf_data

def parse_single_program_output(filepath):
    """
    Parses the program output for a single test run.
    Assumes the output format for a single run is simple:
    === METHOD_NAME ===
    平均召回率: X.XXXX
    平均延迟 (us): Y.YYY
    """
    result = {"Method_Full": None, "Recall": None, "Latency_us": None, "QPS": None}
    if not os.path.exists(filepath):
        print(f"ERROR: Program output file '{filepath}' not found.")
        return result

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    method_match = re.search(r"=== (.*?) ===", content)
    recall_match = re.search(r"平均召回率: (\d+\.\d+)", content)
    latency_match = re.search(r"平均延迟 \(us\): (\d+\.\d+)", content)

    if method_match:
        result["Method_Full"] = method_match.group(1).strip()
    if recall_match:
        result["Recall"] = float(recall_match.group(1))
    if latency_match:
        result["Latency_us"] = float(latency_match.group(1))
        if result["Latency_us"] > 0:
            result["QPS"] = 1_000_000.0 / result["Latency_us"]
        else:
            result["QPS"] = 0
    
    # Try to extract base method and params from Method_Full or tag for better grouping
    # This part might need refinement based on your actual Method_Full string from single runs
    if result["Method_Full"]:
        if "IVFADC" in result["Method_Full"]: result["Method_Base"] = "IVFADC"
        elif "IVF" in result["Method_Full"]: result["Method_Base"] = "IVF"
        elif "Flat" in result["Method_Full"]: result["Method_Base"] = "Flat"
        elif "SIMD" in result["Method_Full"]: result["Method_Base"] = "SIMD"
        elif "PQ" in result["Method_Full"]: result["Method_Base"] = "PQ"
        elif "SQ" in result["Method_Full"]: result["Method_Base"] = "SQ"
        else: result["Method_Base"] = "Unknown"

        nprobe_match = re.search(r"nprobe=(\d+)", result["Method_Full"])
        if nprobe_match: result["nprobe"] = int(nprobe_match.group(1))
        
        rerank_match = re.search(r"rerank_k=(\d+)", result["Method_Full"]) # For IVFADC
        if not rerank_match: rerank_match = re.search(r"rerank=(\d+)", result["Method_Full"]) # For PQ
        if rerank_match: result["rerank_k"] = int(rerank_match.group(1))


    return result

def load_all_per_algorithm_data(results_dir):
    all_data = []
    program_stdout_files = glob.glob(os.path.join(results_dir, "program_stdout_*.txt"))

    for stdout_file in program_stdout_files:
        tag = os.path.basename(stdout_file).replace("program_stdout_", "").replace(".txt", "")
        perf_file = os.path.join(results_dir, f"perf_report_{tag}.txt")

        program_metrics = parse_single_program_output(stdout_file)
        perf_metrics = parse_perf_stat(perf_file)

        if program_metrics.get("Recall") is not None: # Check if parsing was successful
            combined_metrics = {"Tag": tag, **program_metrics, **perf_metrics}
            all_data.append(combined_metrics)
        else:
            print(f"Warning: Could not parse program output for tag '{tag}' from {stdout_file}")


    df = pd.DataFrame(all_data)
    # Attempt to extract more structured info from Tag if Method_Full is too generic
    if not df.empty:
        for idx, row in df.iterrows():
            if pd.isna(row.get('nprobe')) and 'np' in row['Tag']:
                 np_match = re.search(r"np(\d+)", row['Tag'])
                 if np_match: df.loc[idx, 'nprobe'] = int(np_match.group(1))
            if pd.isna(row.get('rerank_k')) and ('rrk' in row['Tag'] or 'rerank' in row['Tag']):
                 rrk_match = re.search(r"(?:rrk|rerank)(\d+)", row['Tag'])
                 if rrk_match: df.loc[idx, 'rerank_k'] = int(rrk_match.group(1))
            if pd.isna(row.get('Method_Base')) or row.get('Method_Base') == 'Unknown':
                if 'ivfadc' in row['Tag'].lower(): df.loc[idx, 'Method_Base'] = "IVFADC"
                elif 'ivf' in row['Tag'].lower(): df.loc[idx, 'Method_Base'] = "IVF"
                elif 'flat' in row['Tag'].lower(): df.loc[idx, 'Method_Base'] = "Flat"
                elif 'simd' in row['Tag'].lower(): df.loc[idx, 'Method_Base'] = "SIMD"
                elif 'pq' in row['Tag'].lower(): df.loc[idx, 'Method_Base'] = "PQ"
                elif 'sq' in row['Tag'].lower(): df.loc[idx, 'Method_Base'] = "SQ"


    return df

def plot_perf_metrics_comparison(df, metrics_to_plot, title_prefix, output_dir):
    if df.empty:
        print(f"DataFrame is empty, skipping {title_prefix} plots.")
        return
    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics_to_plot:
        if metric not in df.columns or df[metric].isnull().all():
            print(f"Metric '{metric}' not found in data or all values are null, skipping plot.")
            continue

        plt.figure(figsize=(max(12, len(df) * 0.5), 8)) # Dynamic width
        
        # Sort by the metric for better visualization, handle NaNs by putting them last
        # Create a temporary column for sorting that puts NaNs last
        sort_metric_col = f"{metric}_sortable"
        df[sort_metric_col] = df[metric].apply(lambda x: float('-inf') if pd.isna(x) else x)
        plot_df = df.sort_values(by=sort_metric_col, ascending=False if metric != "L1_dcache_load_miss_rate" and metric != "branch_miss_rate" else True) # Higher is better for IPC, lower for miss rates
        df.drop(columns=[sort_metric_col], inplace=True)
        
        sns.barplot(x=metric, y="Tag", data=plot_df, palette="viridis", hue="Tag", dodge=False, legend=False)
        plt.title(f"{title_prefix}: {metric} Comparison")
        plt.xlabel(metric)
        plt.ylabel("Algorithm / Configuration (Tag)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{title_prefix.lower().replace(' ', '_')}_{metric.lower()}_comparison.png"))
        plt.close()
    print(f"{title_prefix} comparison plots saved to '{output_dir}'")


# (plot_recall_latency_curves and plot_qps_comparison can be reused from the previous script,
#  just ensure they use the new DataFrame structure and 'Tag' or 'Method_Full' for labels.
#  You might need to adjust them slightly if the 'Method_Base', 'nprobe', 'rerank_k' columns
#  are not consistently populated by parse_single_program_output and load_all_per_algorithm_data.
#  The version from the previous response is a good starting point.)

def plot_recall_latency_curves(df, output_dir): # Adapted from previous
    if df.empty:
        print("DataFrame is empty, skipping recall-latency plots.")
        return
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # IVF Plot
    ivf_df = df[df["Method_Base"] == "IVF"].copy()
    if not ivf_df.empty and 'nprobe' in ivf_df.columns and ivf_df['nprobe'].notna().any():
        ivf_df.dropna(subset=['nprobe', 'Latency_us', 'Recall'], inplace=True)
        ivf_df['nprobe'] = ivf_df['nprobe'].astype(float).astype(int) # Ensure nprobe is int for sorting/labeling
        ivf_df.sort_values(by="nprobe", inplace=True)
        plt.figure(figsize=(10, 6))
        plt.plot(ivf_df["Latency_us"], ivf_df["Recall"], marker='o', linestyle='-', label="IVF")
        for _, row in ivf_df.iterrows():
            if pd.notna(row['nprobe']):
                plt.annotate(f"np={int(row['nprobe'])}", (row["Latency_us"], row["Recall"]), textcoords="offset points", xytext=(0,5), ha='center')
        plt.title("IVF: Recall vs. Latency (us)")
        plt.xlabel("Average Latency (us)")
        plt.ylabel("Recall@k")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "ivf_recall_latency.png"))
        plt.close()

    # IVFADC Plot
    ivfadc_df = df[df["Method_Base"] == "IVFADC"].copy()
    if not ivfadc_df.empty and 'nprobe' in ivfadc_df.columns and ivfadc_df['nprobe'].notna().any() \
        and 'rerank_k' in ivfadc_df.columns and ivfadc_df['rerank_k'].notna().any():
        ivfadc_df.dropna(subset=['nprobe', 'rerank_k', 'Latency_us', 'Recall'], inplace=True)
        ivfadc_df['nprobe'] = ivfadc_df['nprobe'].astype(float).astype(int)
        ivfadc_df['rerank_k'] = ivfadc_df['rerank_k'].astype(float).astype(int)
        
        plt.figure(figsize=(12, 7))
        for rerank_val, group in ivfadc_df.groupby('rerank_k'):
            group = group.sort_values(by="nprobe")
            label = f"IVFADC (rerank_k={rerank_val})"
            plt.plot(group["Latency_us"], group["Recall"], marker='o', linestyle='-', label=label)
            for _, row in group.iterrows():
                 if pd.notna(row['nprobe']):
                    plt.annotate(f"np={int(row['nprobe'])}", (row["Latency_us"], row["Recall"]), textcoords="offset points", xytext=(0,5), ha='center')
        
        plt.title("IVFADC: Recall vs. Latency (us)")
        plt.xlabel("Average Latency (us)")
        plt.ylabel("Recall@k")
        plt.legend(title="Parameters", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.80, 1]) # Adjust for legend
        plt.savefig(os.path.join(output_dir, "ivfadc_recall_latency.png"))
        plt.close()

    # Combined plot
    plt.figure(figsize=(14, 8)) # Increased figure size
    unique_methods = df['Method_Base'].dropna().unique()
    
    for method_base in ["Flat", "SIMD", "SQ"]:
        method_data = df[df["Method_Base"] == method_base]
        if not method_data.empty:
            plt.scatter(method_data["Latency_us"], method_data["Recall"], label=method_base, s=100, alpha=0.7, zorder=3)
            for _, row in method_data.iterrows():
                 plt.annotate(method_base, (row["Latency_us"], row["Recall"]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)

    pq_df = df[df["Method_Base"] == "PQ"].copy()
    if not pq_df.empty and 'rerank_k' in pq_df.columns and pq_df['rerank_k'].notna().any():
        pq_df.dropna(subset=['rerank_k', 'Latency_us', 'Recall'], inplace=True)
        pq_df['rerank_k'] = pq_df['rerank_k'].astype(float).astype(int)
        for rerank_val, group in pq_df.groupby('rerank_k'):
            label=f"PQ (rerank_k={rerank_val})"
            plt.scatter(group["Latency_us"], group["Recall"], label=label, s=100, alpha=0.7, zorder=3)
            for _, row in group.iterrows():
                 plt.annotate(f"PQ rrk{int(row['rerank_k'])}", (row["Latency_us"], row["Recall"]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)

    if not ivf_df.empty: plt.plot(ivf_df["Latency_us"], ivf_df["Recall"], marker='.', linestyle='--', label="IVF (varying nprobe)", alpha=0.7, zorder=2)
    if not ivfadc_df.empty:
        for rerank_val, group in ivfadc_df.groupby('rerank_k'):
             plt.plot(group["Latency_us"], group["Recall"], marker='.', linestyle='--', label=f"IVFADC (rerank_k={rerank_val}, varying nprobe)", alpha=0.7, zorder=2)

    plt.title("Overall Recall vs. Latency (us)")
    plt.xlabel("Average Latency (us) - Log Scale")
    plt.ylabel("Recall@k")
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.78, 1]) # Adjust for legend
    plt.savefig(os.path.join(output_dir, "all_methods_recall_latency.png"))
    plt.close()
    print(f"Recall-Latency plots saved to '{output_dir}'")


# --- 主逻辑 ---
if __name__ == "__main__":
    combined_df = load_all_per_algorithm_data(RESULTS_DIR)

    if not combined_df.empty:
        print("\n--- Combined Per-Algorithm Data (first 5 rows) ---")
        print(combined_df.head())
        combined_df.to_csv(os.path.join(RESULTS_DIR, "parsed_all_algorithms_data_with_perf.csv"), index=False)

        # Plot Recall-Latency and QPS (similar to before, using the new combined_df)
        plot_recall_latency_curves(combined_df, OUTPUT_PLOTS_DIR)
        # plot_qps_comparison(combined_df, OUTPUT_PLOTS_DIR) # You can adapt this from previous script

        # Plot Perf Metrics Comparison
        perf_metrics_to_compare = ["IPC", "L1_dcache_load_miss_rate", "branch_miss_rate", "cycles", "instructions"]
        plot_perf_metrics_comparison(combined_df, perf_metrics_to_compare, "Perf Counters", OUTPUT_PLOTS_DIR)
        
        # You can also print a summary table to console
        print("\n--- Summary Table (Recall, Latency, QPS, IPC, L1 Miss Rate, Branch Miss Rate) ---")
        summary_cols = ["Tag", "Method_Base", "nprobe", "rerank_k", "Recall", "Latency_us", "QPS", "IPC", "L1_dcache_load_miss_rate", "branch_miss_rate"]
        # Filter out columns that might not exist if parsing failed for some rows
        existing_summary_cols = [col for col in summary_cols if col in combined_df.columns]
        print(combined_df[existing_summary_cols].to_string(index=False))

    else:
        print(f"No data loaded. Ensure '{RESULTS_DIR}' contains valid output files.")

    print(f"\nProcessing complete. Plots are in '{OUTPUT_PLOTS_DIR}', parsed data in '{RESULTS_DIR}'.")
