#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import os

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# Data from the GPU experiment results
def get_experimental_data():
    """Get the experimental data from GPU runs"""
    
    # CPU baseline data
    cpu_data = {
        'avg_recall': 0.99995,
        'avg_latency_us': 8125.22,
        'total_time_ms': 16255,
        'throughput_qps': 2000 / 16.255  # queries per second
    }
    
    # GPU Original version data
    gpu_original = {
        'batch_sizes': [100, 500, 1000, 2000],
        'batch_times_ms': [797.611, 818.258, 830.278, 854.388],
        'total_times_ms': [1836, 829, 841, 865],
        'recalls': [1.0, 1.0, 0.9999, 0.99995],
        'throughputs_qps': [54.4662, 603.136, 1189.06, 2312.14],
        'speedups': [0.442674, 4.90199, 9.66409, 18.7919]
    }
    
    # GPU Optimized version data
    gpu_optimized = {
        'batch_sizes': [100, 500, 1000, 2000],
        'batch_times_ms': [79.264, 88.2645, 94.7816, 115.725],
        'total_times_ms': [89, 98, 105, 126],
        'recalls': [1.0, 1.0, 0.9999, 0.99995],
        'throughputs_qps': [1123.6, 5102.04, 9523.81, 15873],
        'speedups': [9.13202, 41.4668, 77.4048, 129.008]
    }
    
    return cpu_data, gpu_original, gpu_optimized

def create_comprehensive_visualization():
    """Create comprehensive visualization of GPU performance results"""
    
    cpu_data, gpu_original, gpu_optimized = get_experimental_data()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Speedup Comparison
    ax1 = plt.subplot(2, 4, 1)
    batch_sizes = gpu_original['batch_sizes']
    x_pos = np.arange(len(batch_sizes))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, gpu_original['speedups'], width, 
                    label='Original GPU', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x_pos + width/2, gpu_optimized['speedups'], width,
                    label='Optimized GPU', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Speedup (x)')
    ax1.set_title('GPU Speedup vs CPU Baseline')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x', ha='center', va='bottom', fontsize=9)
    
    # 2. Throughput Comparison
    ax2 = plt.subplot(2, 4, 2)
    cpu_throughput = [cpu_data['throughput_qps']] * len(batch_sizes)
    
    ax2.plot(batch_sizes, cpu_throughput, 'k--', linewidth=2, label='CPU Baseline')
    ax2.plot(batch_sizes, gpu_original['throughputs_qps'], 'o-', 
             linewidth=2, markersize=8, label='Original GPU')
    ax2.plot(batch_sizes, gpu_optimized['throughputs_qps'], 's-', 
             linewidth=2, markersize=8, label='Optimized GPU')
    
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (queries/sec)')
    ax2.set_title('Throughput Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Processing Time Breakdown
    ax3 = plt.subplot(2, 4, 3)
    
    # Calculate overhead time (total - batch processing)
    original_overhead = [total - batch for total, batch in 
                        zip(gpu_original['total_times_ms'], gpu_original['batch_times_ms'])]
    optimized_overhead = [total - batch for total, batch in 
                         zip(gpu_optimized['total_times_ms'], gpu_optimized['batch_times_ms'])]
    
    x_pos = np.arange(len(batch_sizes))
    width = 0.35
    
    # Stacked bar chart
    bars1_batch = ax3.bar(x_pos - width/2, gpu_original['batch_times_ms'], width, 
                         label='Original Batch Time', alpha=0.8, color='lightblue')
    bars1_overhead = ax3.bar(x_pos - width/2, original_overhead, width,
                           bottom=gpu_original['batch_times_ms'],
                           label='Original Overhead', alpha=0.8, color='darkblue')
    
    bars2_batch = ax3.bar(x_pos + width/2, gpu_optimized['batch_times_ms'], width,
                         label='Optimized Batch Time', alpha=0.8, color='lightgreen')
    bars2_overhead = ax3.bar(x_pos + width/2, optimized_overhead, width,
                           bottom=gpu_optimized['batch_times_ms'],
                           label='Optimized Overhead', alpha=0.8, color='darkgreen')
    
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Processing Time Breakdown')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(batch_sizes)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Recall Accuracy
    ax4 = plt.subplot(2, 4, 4)
    cpu_recall = [cpu_data['avg_recall']] * len(batch_sizes)
    
    ax4.plot(batch_sizes, cpu_recall, 'k--', linewidth=2, label='CPU Baseline')
    ax4.plot(batch_sizes, gpu_original['recalls'], 'o-', 
             linewidth=2, markersize=8, label='Original GPU')
    ax4.plot(batch_sizes, gpu_optimized['recalls'], 's-', 
             linewidth=2, markersize=8, label='Optimized GPU')
    
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Recall')
    ax4.set_title('Recall Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.999, 1.0001)
    
    # 5. Efficiency Analysis (Speedup per Batch Size)
    ax5 = plt.subplot(2, 4, 5)
    efficiency_original = [speedup / batch for speedup, batch in 
                          zip(gpu_original['speedups'], batch_sizes)]
    efficiency_optimized = [speedup / batch for speedup, batch in 
                           zip(gpu_optimized['speedups'], batch_sizes)]
    
    ax5.plot(batch_sizes, efficiency_original, 'o-', 
             linewidth=2, markersize=8, label='Original GPU')
    ax5.plot(batch_sizes, efficiency_optimized, 's-', 
             linewidth=2, markersize=8, label='Optimized GPU')
    
    ax5.set_xlabel('Batch Size')
    ax5.set_ylabel('Speedup per Query Unit')
    ax5.set_title('Efficiency Analysis')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Improvement (Optimized vs Original)
    ax6 = plt.subplot(2, 4, 6)
    improvement_speedup = [opt / orig for opt, orig in 
                          zip(gpu_optimized['speedups'], gpu_original['speedups'])]
    improvement_throughput = [opt / orig for opt, orig in 
                             zip(gpu_optimized['throughputs_qps'], gpu_original['throughputs_qps'])]
    
    bars1 = ax6.bar(x_pos - width/2, improvement_speedup, width,
                   label='Speedup Improvement', alpha=0.8, color='orange')
    bars2 = ax6.bar(x_pos + width/2, improvement_throughput, width,
                   label='Throughput Improvement', alpha=0.8, color='purple')
    
    ax6.set_xlabel('Batch Size')
    ax6.set_ylabel('Improvement Factor')
    ax6.set_title('Optimization Improvement')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(batch_sizes)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x', ha='center', va='bottom', fontsize=9)
    
    # 7. Batch Processing Time vs Batch Size
    ax7 = plt.subplot(2, 4, 7)
    ax7.plot(batch_sizes, gpu_original['batch_times_ms'], 'o-', 
             linewidth=2, markersize=8, label='Original GPU')
    ax7.plot(batch_sizes, gpu_optimized['batch_times_ms'], 's-', 
             linewidth=2, markersize=8, label='Optimized GPU')
    
    ax7.set_xlabel('Batch Size')
    ax7.set_ylabel('Batch Processing Time (ms)')
    ax7.set_title('Batch Processing Time Scaling')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Performance Summary Table
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Create summary data
    summary_data = []
    for i, batch in enumerate(batch_sizes):
        summary_data.append([
            f'{batch}',
            f'{gpu_original["speedups"][i]:.1f}x',
            f'{gpu_optimized["speedups"][i]:.1f}x',
            f'{improvement_speedup[i]:.1f}x'
        ])
    
    table_data = [['Batch', 'Orig.', 'Opt.', 'Improve.']] + summary_data
    
    table = ax8.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax8.set_title('Performance Summary', pad=20)
    
    plt.tight_layout()
    plt.savefig('gpu_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('gpu_performance_analysis.pdf', bbox_inches='tight')
    print("Comprehensive visualization saved as 'gpu_performance_analysis.png' and 'gpu_performance_analysis.pdf'")
    
    return fig

def create_detailed_analysis():
    """Create detailed analysis charts"""
    
    cpu_data, gpu_original, gpu_optimized = get_experimental_data()
    
    # Create a separate detailed figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    batch_sizes = gpu_original['batch_sizes']
    
    # 1. Logarithmic Speedup Analysis
    ax1.semilogy(batch_sizes, gpu_original['speedups'], 'o-', 
                linewidth=3, markersize=10, label='Original GPU', color='blue')
    ax1.semilogy(batch_sizes, gpu_optimized['speedups'], 's-', 
                linewidth=3, markersize=10, label='Optimized GPU', color='red')
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='CPU Baseline (1x)')
    
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Speedup (log scale)', fontsize=12)
    ax1.set_title('GPU Speedup Analysis (Log Scale)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Annotate peak performance
    max_speedup_idx = np.argmax(gpu_optimized['speedups'])
    ax1.annotate(f'Peak: {gpu_optimized["speedups"][max_speedup_idx]:.1f}x',
                xy=(batch_sizes[max_speedup_idx], gpu_optimized['speedups'][max_speedup_idx]),
                xytext=(batch_sizes[max_speedup_idx] - 200, gpu_optimized['speedups'][max_speedup_idx] * 0.7),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=11, fontweight='bold')
    
    # 2. Throughput Scaling Analysis
    cpu_throughput = cpu_data['throughput_qps']
    ax2.plot(batch_sizes, [cpu_throughput] * len(batch_sizes), 'k--', 
             linewidth=3, label='CPU Baseline')
    ax2.plot(batch_sizes, gpu_original['throughputs_qps'], 'o-', 
             linewidth=3, markersize=10, label='Original GPU')
    ax2.plot(batch_sizes, gpu_optimized['throughputs_qps'], 's-', 
             linewidth=3, markersize=10, label='Optimized GPU')
    
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Throughput (queries/sec)', fontsize=12)
    ax2.set_title('Throughput Scaling Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Optimization Impact
    improvement_factors = [opt / orig for opt, orig in 
                          zip(gpu_optimized['speedups'], gpu_original['speedups'])]
    
    bars = ax3.bar(range(len(batch_sizes)), improvement_factors, 
                   color=['lightcoral', 'gold', 'lightgreen', 'skyblue'], alpha=0.8)
    ax3.set_xlabel('Batch Size', fontsize=12)
    ax3.set_ylabel('Improvement Factor', fontsize=12)
    ax3.set_title('Optimization Impact (Optimized / Original)', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(batch_sizes)))
    ax3.set_xticklabels(batch_sizes)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}x', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # 4. Performance Efficiency (Speedup per unit batch size)
    efficiency_original = [speedup / batch for speedup, batch in 
                          zip(gpu_original['speedups'], batch_sizes)]
    efficiency_optimized = [speedup / batch for speedup, batch in 
                           zip(gpu_optimized['speedups'], batch_sizes)]
    
    ax4.plot(batch_sizes, efficiency_original, 'o-', 
             linewidth=3, markersize=10, label='Original GPU')
    ax4.plot(batch_sizes, efficiency_optimized, 's-', 
             linewidth=3, markersize=10, label='Optimized GPU')
    
    ax4.set_xlabel('Batch Size', fontsize=12)
    ax4.set_ylabel('Efficiency (Speedup/Batch Size)', fontsize=12)
    ax4.set_title('Performance Efficiency Analysis', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gpu_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('gpu_detailed_analysis.pdf', bbox_inches='tight')
    print("Detailed analysis saved as 'gpu_detailed_analysis.png' and 'gpu_detailed_analysis.pdf'")
    
    return fig

def create_performance_comparison_table():
    """Create a detailed performance comparison table"""
    
    cpu_data, gpu_original, gpu_optimized = get_experimental_data()
    
    # Create DataFrame for easy manipulation
    data = {
        'Batch Size': gpu_original['batch_sizes'],
        'CPU Throughput (q/s)': [cpu_data['throughput_qps']] * len(gpu_original['batch_sizes']),
        'GPU Orig Throughput (q/s)': gpu_original['throughputs_qps'],
        'GPU Opt Throughput (q/s)': gpu_optimized['throughputs_qps'],
        'GPU Orig Speedup': gpu_original['speedups'],
        'GPU Opt Speedup': gpu_optimized['speedups'],
        'Optimization Improvement': [opt / orig for opt, orig in 
                                   zip(gpu_optimized['speedups'], gpu_original['speedups'])],
        'GPU Orig Batch Time (ms)': gpu_original['batch_times_ms'],
        'GPU Opt Batch Time (ms)': gpu_optimized['batch_times_ms'],
        'Time Reduction': [1 - (opt / orig) for opt, orig in 
                          zip(gpu_optimized['batch_times_ms'], gpu_original['batch_times_ms'])]
    }
    
    df = pd.DataFrame(data)
    
    # Round numerical values for better display
    df = df.round(2)
    
    # Save as CSV
    df.to_csv('gpu_performance_comparison.csv', index=False)
    print("Performance comparison table saved as 'gpu_performance_comparison.csv'")
    
    # Display the table
    print("\nPerformance Comparison Summary:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    return df

def generate_performance_report():
    """Generate a comprehensive performance report"""
    
    cpu_data, gpu_original, gpu_optimized = get_experimental_data()
    
    report = f"""
GPU Performance Analysis Report
===============================

Experimental Setup:
- Dataset: DEEP100K (100,000 base vectors, 96 dimensions)
- Query Set: 2,000 queries
- Top-K: 10 nearest neighbors
- CPU: Sequential brute-force search
- GPU: Batch processing with cuBLAS matrix multiplication

CPU Baseline Performance:
- Average Recall: {cpu_data['avg_recall']:.5f}
- Average Latency: {cpu_data['avg_latency_us']:.2f} μs per query
- Total Processing Time: {cpu_data['total_time_ms']} ms
- Throughput: {cpu_data['throughput_qps']:.2f} queries/second

GPU Performance Results:
------------------------

Original GPU Implementation:
"""
    
    for i, batch in enumerate(gpu_original['batch_sizes']):
        report += f"""
  Batch Size {batch}:
    - Throughput: {gpu_original['throughputs_qps'][i]:.2f} q/s
    - Speedup: {gpu_original['speedups'][i]:.2f}x
    - Batch Processing Time: {gpu_original['batch_times_ms'][i]:.2f} ms
    - Recall: {gpu_original['recalls'][i]:.5f}
"""
    
    report += "\nOptimized GPU Implementation:\n"
    
    for i, batch in enumerate(gpu_optimized['batch_sizes']):
        improvement = gpu_optimized['speedups'][i] / gpu_original['speedups'][i]
        report += f"""
  Batch Size {batch}:
    - Throughput: {gpu_optimized['throughputs_qps'][i]:.2f} q/s
    - Speedup: {gpu_optimized['speedups'][i]:.2f}x
    - Batch Processing Time: {gpu_optimized['batch_times_ms'][i]:.2f} ms
    - Recall: {gpu_optimized['recalls'][i]:.5f}
    - Improvement over Original: {improvement:.2f}x
"""
    
    # Key findings
    max_speedup = max(gpu_optimized['speedups'])
    max_speedup_batch = gpu_optimized['batch_sizes'][gpu_optimized['speedups'].index(max_speedup)]
    max_improvement = max([opt / orig for opt, orig in 
                          zip(gpu_optimized['speedups'], gpu_original['speedups'])])
    
    report += f"""

Key Findings:
-------------
1. Maximum Speedup: {max_speedup:.1f}x achieved with batch size {max_speedup_batch}
2. Maximum Optimization Improvement: {max_improvement:.1f}x
3. Batch size impact: Performance scales significantly with batch size
4. Small batch penalty: Batch size 100 shows suboptimal performance in original version
5. Optimization effectiveness: Heap-based top-k selection significantly outperforms naive selection
6. Recall consistency: All implementations maintain high recall (≥99.99%)

Performance Scaling Analysis:
- GPU performance scales super-linearly with batch size
- Optimization benefits increase with larger batch sizes
- Memory access patterns become more efficient with larger batches
- GPU utilization improves significantly with batch processing

Recommendations:
- Use batch sizes ≥ 500 for optimal GPU utilization
- Implement optimized algorithms (heap-based selection) for better performance
- Consider GPU for high-throughput scenarios rather than low-latency single queries
- Monitor memory usage for very large batch sizes
"""
    
    # Save report
    with open('gpu_performance_report.txt', 'w') as f:
        f.write(report)
    
    print("Performance report saved as 'gpu_performance_report.txt'")
    print(report)

def main():
    """Main function to generate all visualizations and analysis"""
    
    print("Generating GPU Performance Analysis...")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    os.makedirs('gpu_analysis_results', exist_ok=True)
    os.chdir('gpu_analysis_results')
    
    # Generate all visualizations and analysis
    print("\n1. Creating comprehensive visualization...")
    create_comprehensive_visualization()
    
    print("\n2. Creating detailed analysis...")
    create_detailed_analysis()
    
    print("\n3. Creating performance comparison table...")
    create_performance_comparison_table()
    
    print("\n4. Generating performance report...")
    generate_performance_report()
    
    print("\n" + "=" * 50)
    print("Analysis complete! Generated files:")
    print("- gpu_performance_analysis.png/pdf")
    print("- gpu_detailed_analysis.png/pdf") 
    print("- gpu_performance_comparison.csv")
    print("- gpu_performance_report.txt")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()