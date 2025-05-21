import matplotlib.pyplot as plt
import matplotlib as mpl

# Set Chinese font
mpl.rcParams['font.sans-serif'] = ['SimHei']  # Or any other Chinese font like 'Microsoft YaHei'
mpl.rcParams['axes.unicode_minus'] = False  # To display negative signs correctly

# IVF (Pthread) Data
nprobe_ivf = [1, 2, 4, 8, 16, 32]
recall_ivf = [0.57525, 0.72980, 0.85170, 0.92390, 0.96960, 0.99020]
latency_ivf = [375.288, 416.984, 442.451, 568.385, 678.527, 909.113]

plt.figure(figsize=(10, 6))

# Plot Recall vs. Latency
plt.plot(latency_ivf, recall_ivf, marker='o', linestyle='-', color='b', label='IVF (Pthread)')
for i, txt in enumerate(nprobe_ivf):
    plt.annotate(f'nprobe={txt}', (latency_ivf[i], recall_ivf[i]), textcoords="offset points", xytext=(5,5), ha='left')

plt.title('IVF (Pthread) 性能: 召回率 vs. 延迟 (clusters=256)')
plt.xlabel('平均延迟 (μs)')
plt.ylabel('平均召回率')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('ivf_performance.pdf')
# plt.show()

print("IVF performance plot saved as ivf_performance.pdf")