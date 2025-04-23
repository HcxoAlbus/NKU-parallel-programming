import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 配置 ---
RESULTS_FILE = "pq_results.csv"
OUTPUT_PLOT_FILE = "pq_performance_curve.png"
PLOT_TITLE = "PQ Performance on DEEP100K (k=10, IP Distance)"
X_LABEL = "Recall@10 (%)"
Y_LABEL = "QPS (Queries Per Second)"

# --- 数据加载和处理 ---
try:
    # 读取 CSV 文件
    df = pd.read_csv(RESULTS_FILE)

    # 过滤掉可能存在的失败行 (假设 latency 为 0 或 recall 为 0)
    df_valid = df[(df['latency_us'] > 0) & (df['recall'] >= 0)].copy() # recall=0 可能是有效结果

    if df_valid.empty:
        print(f"错误: 未在 {RESULTS_FILE} 中找到有效的性能数据。")
        exit(1)

    # 计算 QPS (Queries Per Second)
    # 1 秒 = 1,000,000 微秒
    df_valid['qps'] = 1_000_000 / df_valid['latency_us']

    # 将 Recall 转换为百分比
    df_valid['recall_percent'] = df_valid['recall'] * 100

    # 将 nsub 转为分类或字符串，以便 seaborn 正确分组和添加图例
    df_valid['nsub'] = df_valid['nsub'].astype(str)


    print("加载并处理后的数据:")
    print(df_valid)

    # --- 绘图 ---
    plt.style.use('seaborn-v0_8-whitegrid') # 使用 seaborn 风格
    plt.figure(figsize=(10, 6)) # 设置图像大小

    # 使用 seaborn 绘制线图，按 nsub 分组
    sns.lineplot(
        data=df_valid,
        x='recall_percent',
        y='qps',
        hue='nsub',          # 按 nsub 分色
        style='nsub',        # 按 nsub 分线型 (可选)
        marker='o',          # 添加点标记
        markersize=8,
        palette='viridis',   # 选择一个色板
        legend='full'
    )

    # 设置 Y 轴为对数刻度
    plt.yscale('log')

    # 设置标题和轴标签
    plt.title(PLOT_TITLE, fontsize=16)
    plt.xlabel(X_LABEL, fontsize=12)
    plt.ylabel(Y_LABEL, fontsize=12)

    # 添加图例
    plt.legend(title='nsub', title_fontsize='13', fontsize='11')

    # 优化网格线
    plt.grid(True, which='major', linestyle='--', linewidth=0.7)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5)

    # (可选) 添加 rerank_k 值的文本标注
    # 注意：如果点很密集，这可能会让图变得混乱
    # for i, point in df_valid.iterrows():
    #     plt.text(point['recall_percent'] + 0.1, point['qps'], str(int(point['rerank_k'])), fontsize=8)

    # 保存图像
    plt.savefig(OUTPUT_PLOT_FILE, dpi=300, bbox_inches='tight')
    print(f"\n性能曲线图已保存到: {OUTPUT_PLOT_FILE}")

    # 显示图像
    plt.show()

except FileNotFoundError:
    print(f"错误: 结果文件 '{RESULTS_FILE}' 未找到。")
    print("请先运行 profile_pq.sh 脚本生成结果。")
except Exception as e:
    print(f"处理或绘图时发生错误: {e}")
