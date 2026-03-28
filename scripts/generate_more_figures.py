import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

os.makedirs('/Users/tom/Documents/Research Paper/manuscript/figures', exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# 1. Bubble Chart: Accuracy vs Throughput (Size = RAM)
models = ['TinyLlama-1.1B', 'Phi-3-mini-3.8B', 'Qwen2.5-3B', 'Mistral-7B']
acc = [22, 86, 80, 85]
tps = [17.40, 4.54, 5.58, 3.51]
ram = [1.36, 4.30, 2.73, 5.40] # GB

plt.figure(figsize=(10, 6))
scatter = plt.scatter(tps, acc, s=[r * 500 for r in ram], alpha=0.6, c=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], edgecolors='w', linewidth=2)

for i, model in enumerate(models):
    plt.annotate(model, (tps[i], acc[i]), xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold')

plt.title('Accuracy vs Throughput Trade-off\n(Bubble size represents Peak RAM in GB)', fontsize=14, fontweight='bold')
plt.xlabel('Inference Throughput (Tokens/second)')
plt.ylabel('Classification Accuracy (%)')
plt.ylim(0, 100)
plt.xlim(0, 20)
plt.tight_layout()
plt.savefig('/Users/tom/Documents/Research Paper/manuscript/figures/acc_vs_tps_bubble.png', dpi=300)
plt.close()

# 2. Agent Overhead Grouped Bar Chart
labels = ['TinyLlama-1.1B', 'Phi-3-mini-3.8B', 'Qwen2.5-3B', 'Mistral-7B']
single_expected = [6.57 * 3, 12.81 * 3, 8.09 * 3, 21.60 * 3] # Expected (3x Single Step)
agent_actual = [17.52, 39.97, 20.40, 73.74] # Actual 3-Step Chain

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, single_expected, width, label='Expected Linear (3 × Single)', color='#aec7e8')
rects2 = ax.bar(x + width/2, agent_actual, width, label='Actual 3-Step Agent Chain', color='#1f77b4')

ax.set_ylabel('End-to-End Latency (Seconds)', fontsize=12)
ax.set_title('Agent Chain Latency: Expected vs Actual', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend()

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('/Users/tom/Documents/Research Paper/manuscript/figures/agent_overhead_bar.png', dpi=300)
plt.close()

# 3. Cold vs Warm TTFT Reduction
cold = [2505.3, 6204.1, 5673.5, 11395.3]
warm = [211.3, 257.4, 424.5, 426.0]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, cold, width, label='Cold TTFT (ms)', color='#ff9896')
ax.bar(x + width/2, warm, width, label='Warm TTFT (ms)', color='#d62728')

ax.set_ylabel('Time To First Token (Milliseconds)', fontsize=12)
ax.set_title('Impact of Caching on Time-to-First-Token (TTFT)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('/Users/tom/Documents/Research Paper/manuscript/figures/ttft_reduction.png', dpi=300)
plt.close()

print('Successfully generated 3 publication-ready figures!')
