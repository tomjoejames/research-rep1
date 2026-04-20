#!/usr/bin/env python3
"""
Generate all publication-ready figures for the research paper.
Uses CORRECTED accuracy values verified against raw CSV data.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import csv
import os

FIGURES_DIR = '/Users/tom/Documents/Research Paper/manuscript/figures'
RESULTS_DIR = '/Users/tom/Documents/Research Paper/results'
os.makedirs(FIGURES_DIR, exist_ok=True)

# === Global style ===
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# === CORRECTED DATA (verified against CSV) ===
MODELS = ['TinyLlama-1.1B', 'Phi-3-mini-3.8B', 'Qwen2.5-3B', 'Mistral-7B']
MODEL_SHORT = ['TinyLlama\n1.1B', 'Phi-3-mini\n3.8B', 'Qwen2.5\n3B', 'Mistral\n7B']
COLORS = ['#5B8DEF', '#FF8C42', '#2ECC71', '#E74C3C']
COLORS_LIGHT = ['#A8C5F7', '#FFBE8E', '#82E0AA', '#F1948A']

ACC = [27, 89, 80, 93]  # CORRECTED
TPS = [16.35, 4.57, 5.64, 3.49]  # warm means from deep_analysis
RAM_GB = [1.36, 4.31, 2.74, 5.41]
PEAK_RAM_MB = [1360, 4306, 2735, 5407]
LOAD_TIME = [1.81, 12.76, 12.98, 17.09]

# E3 data
SINGLE_STEP = [6.70, 12.78, 8.01, 21.52]
AGENT_CHAIN = [17.52, 39.97, 20.40, 73.74]
LINEAR_EXPECTED = [s * 3 for s in SINGLE_STEP]

STEP1 = [5.54, 11.78, 6.64, 23.16]
STEP2 = [5.65, 12.23, 6.52, 27.08]
STEP3 = [6.35, 15.99, 7.19, 23.97]

# E6 data
COLD_TTFT = [2505, 6204, 5674, 11395]
WARM_TTFT = [189, 237, 412, 408]

# DI data
DI_STD = [0.7612, 0.7188, 0.7719, 0.6926]
DI_FLOOR = [0.0, 0.7188, 0.7719, 0.6926]

# Energy data
ENERGY_J = [212.8, 744.8, 451.2, 969.2]
PPR = [0.127, 0.120, 0.177, 0.096]

# Per-category F1
CATEGORIES = ['Inquiry', 'Complaint', 'Feedback', 'Spam', 'Urgent']
F1_MATRIX = {
    'TinyLlama-1.1B': [0.12, 0.49, 0.52, 0.00, 0.52],
    'Phi-3-mini-3.8B': [0.92, 0.85, 0.82, 0.95, 0.91],
    'Qwen2.5-3B':     [0.83, 0.73, 0.79, 0.89, 0.80],
    'Mistral-7B':      [0.97, 0.85, 0.82, 1.00, 1.00],
}

# ══════════════════════════════════════════════════════════════
# FIGURE 1: Baseline Throughput Bar Chart
# ══════════════════════════════════════════════════════════════
def fig1_throughput_bar():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(MODEL_SHORT, TPS, color=COLORS, edgecolor='white', linewidth=1.5, width=0.6)
    
    for bar, val in zip(bars, TPS):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Tokens per Second (tok/s)')
    ax.set_title('Baseline Inference Throughput by Model\n(Q4_K_M, Device 1, warm runs)', fontweight='bold')
    ax.set_ylim(0, 20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    plt.savefig(f'{FIGURES_DIR}/throughput_bar.png')
    plt.close()
    print('  ✅ throughput_bar.png')


# ══════════════════════════════════════════════════════════════
# FIGURE 2: Accuracy vs Throughput Bubble Chart (CORRECTED)
# ══════════════════════════════════════════════════════════════
def fig2_acc_vs_tps():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    for i, model in enumerate(MODELS):
        ax.scatter(TPS[i], ACC[i], s=RAM_GB[i] * 450, alpha=0.7,
                   c=COLORS[i], edgecolors='#333', linewidth=1.5, zorder=5)
        
        offset_x, offset_y = 12, 8
        if model == 'Qwen2.5-3B':
            offset_x, offset_y = -80, 12
        elif model == 'TinyLlama-1.1B':
            offset_x, offset_y = -90, -15
        
        ax.annotate(f'{model}\n({ACC[i]}%, {RAM_GB[i]}GB)',
                    (TPS[i], ACC[i]),
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#666', lw=0.8) if abs(offset_x) > 15 else None)
    
    # Cloud baseline reference line
    ax.axhline(y=85, color='#999', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(17, 86.5, 'GPT-3.5 Baseline (~85-90%)', fontsize=8, color='#666', style='italic')
    
    ax.set_xlabel('Inference Throughput (tok/s)')
    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_title('Accuracy vs Throughput Trade-off\n(Bubble size ∝ Peak RAM usage)', fontweight='bold')
    ax.set_ylim(10, 100)
    ax.set_xlim(0, 20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2)
    
    plt.savefig(f'{FIGURES_DIR}/acc_vs_tps_bubble.png')
    plt.close()
    print('  ✅ acc_vs_tps_bubble.png (CORRECTED)')


# ══════════════════════════════════════════════════════════════
# FIGURE 3: Agent Overhead Bar Chart (CORRECTED)
# ══════════════════════════════════════════════════════════════
def fig3_agent_overhead():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODELS))
    w = 0.32
    
    bars1 = ax.bar(x - w/2, LINEAR_EXPECTED, w, label='Expected Linear (3 × Single)',
                   color=COLORS_LIGHT, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + w/2, AGENT_CHAIN, w, label='Actual 3-Step Agent Chain',
                   color=COLORS, edgecolor='white', linewidth=1)
    
    # Add overhead ratio labels
    for i in range(len(MODELS)):
        ratio = AGENT_CHAIN[i] / LINEAR_EXPECTED[i]
        y_pos = max(LINEAR_EXPECTED[i], AGENT_CHAIN[i]) + 1.5
        ax.text(x[i], y_pos, f'{ratio:.2f}×', ha='center', fontsize=10,
                fontweight='bold', color='#333')
    
    ax.set_ylabel('End-to-End Latency (seconds)')
    ax.set_title('Agent Chain Latency: Expected vs Actual\n(Overhead ratios shown above)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_SHORT)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    plt.savefig(f'{FIGURES_DIR}/agent_overhead_bar.png')
    plt.close()
    print('  ✅ agent_overhead_bar.png (CORRECTED)')


# ══════════════════════════════════════════════════════════════
# FIGURE 4: Cold vs Warm TTFT
# ══════════════════════════════════════════════════════════════
def fig4_ttft():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODELS))
    w = 0.32
    
    ax.bar(x - w/2, COLD_TTFT, w, label='Cold TTFT', color=COLORS_LIGHT, edgecolor='white')
    ax.bar(x + w/2, WARM_TTFT, w, label='Warm TTFT', color=COLORS, edgecolor='white')
    
    # Add reduction percentages
    for i in range(len(MODELS)):
        reduction = (1 - WARM_TTFT[i] / COLD_TTFT[i]) * 100
        ax.text(x[i], COLD_TTFT[i] + 200, f'↓{reduction:.0f}%',
                ha='center', fontsize=10, fontweight='bold', color='#27AE60')
    
    ax.set_ylabel('Time to First Token (ms)')
    ax.set_title('Cold-Start vs Warm TTFT\n(Percentage reduction shown above cold bars)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_SHORT)
    ax.set_yscale('log')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(f'{FIGURES_DIR}/ttft_reduction.png')
    plt.close()
    print('  ✅ ttft_reduction.png')


# ══════════════════════════════════════════════════════════════
# FIGURE 5: DI Heatmap with Sensitivity
# ══════════════════════════════════════════════════════════════
def fig5_di_heatmap():
    # Compute DI for a grid of w1 (accuracy weight) and w2 (latency weight)
    # w3 = 0.15 fixed, w4 = 1 - w1 - w2 - w3
    A_baseline = 85.0
    fastest = min(TPS)  # Actually we need latency, not TPS
    latencies = [6.08, 21.28, 12.89, 27.69]
    fastest_lat = min(latencies)
    smallest_ram = min(PEAK_RAM_MB)
    
    w1_range = np.arange(0.1, 0.6, 0.05)
    w2_range = np.arange(0.1, 0.6, 0.05)
    
    # For each (w1, w2), find which model ranks #1
    winner_grid = np.zeros((len(w2_range), len(w1_range)))
    
    for i, w2 in enumerate(w2_range):
        for j, w1 in enumerate(w1_range):
            w3 = 0.10
            w4 = 1.0 - w1 - w2 - w3
            if w4 < 0.05:
                winner_grid[i, j] = np.nan
                continue
            
            best_di, best_model = -1, -1
            for k in range(4):
                if ACC[k] < 50:  # accuracy floor
                    continue
                a_comp = min(1.0, ACC[k] / A_baseline) * w1
                l_comp = (fastest_lat / latencies[k]) * w2
                m_comp = (smallest_ram / PEAK_RAM_MB[k]) * w3
                cr_comp = 1.0 * w4
                di = a_comp + l_comp + m_comp + cr_comp
                if di > best_di:
                    best_di = di
                    best_model = k
            winner_grid[i, j] = best_model
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    cmap = matplotlib.colors.ListedColormap(['#FF8C42', '#2ECC71', '#E74C3C'])
    bounds = [0.5, 1.5, 2.5, 3.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(winner_grid, cmap=cmap, norm=norm, aspect='auto',
                   extent=[w1_range[0]-0.025, w1_range[-1]+0.025,
                           w2_range[-1]+0.025, w2_range[0]-0.025])
    
    ax.set_xlabel('Accuracy Weight (w₁)', fontsize=12)
    ax.set_ylabel('Latency Weight (w₂)', fontsize=12)
    ax.set_title('DI Winner Map: Optimal Model by Weight Configuration\n(w₃=0.10 fixed, w₄=1-w₁-w₂-w₃, accuracy floor=50%)', fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF8C42', label='Phi-3-mini-3.8B'),
        Patch(facecolor='#2ECC71', label='Qwen2.5-3B'),
        Patch(facecolor='#E74C3C', label='Mistral-7B'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.savefig(f'{FIGURES_DIR}/di_heatmap.png')
    plt.close()
    print('  ✅ di_heatmap.png (CORRECTED)')


# ══════════════════════════════════════════════════════════════
# FIGURE 6 (NEW): Per-Step Latency Decomposition
# ══════════════════════════════════════════════════════════════
def fig6_per_step_latency():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODELS))
    w = 0.22
    
    bars1 = ax.bar(x - w, STEP1, w, label='Step 1 (Extract)', color=COLORS_LIGHT, edgecolor='white')
    bars2 = ax.bar(x,     STEP2, w, label='Step 2 (Compute)', color=[c + 'BB' if len(c) < 7 else c for c in COLORS], edgecolor='white')
    bars3 = ax.bar(x + w, STEP3, w, label='Step 3 (Format)',  color=COLORS, edgecolor='white')
    
    # Add step3/step1 ratio
    for i in range(len(MODELS)):
        ratio = STEP3[i] / STEP1[i]
        symbol = '⚠️' if ratio > 1.15 else '✓'
        y_pos = max(STEP1[i], STEP2[i], STEP3[i]) + 0.5
        ax.text(x[i], y_pos, f'S3/S1={ratio:.2f}×', ha='center', fontsize=9, fontweight='bold',
                color='#E74C3C' if ratio > 1.15 else '#27AE60')
    
    ax.set_ylabel('Per-Step Latency (seconds)')
    ax.set_title('Per-Step Agent Latency Decomposition\n(Context accumulation effect shown as S3/S1 ratio)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_SHORT)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    plt.savefig(f'{FIGURES_DIR}/per_step_latency.png')
    plt.close()
    print('  ✅ per_step_latency.png (NEW)')


# ══════════════════════════════════════════════════════════════
# FIGURE 7 (NEW): Per-Category F1 Heatmap
# ══════════════════════════════════════════════════════════════
def fig7_f1_heatmap():
    fig, ax = plt.subplots(figsize=(9, 5))
    
    data = np.array([F1_MATRIX[m] for m in MODELS])
    
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks(np.arange(len(CATEGORIES)))
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_xticklabels(CATEGORIES, fontsize=11)
    ax.set_yticklabels(MODELS, fontsize=11)
    
    # Annotate cells
    for i in range(len(MODELS)):
        for j in range(len(CATEGORIES)):
            val = data[i, j]
            color = 'white' if val < 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)
    
    ax.set_title('Per-Category F1-Score by Model\n(Q4_K_M, Email Classification, n=100)', fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('F1-Score', fontsize=11)
    
    plt.savefig(f'{FIGURES_DIR}/f1_heatmap.png')
    plt.close()
    print('  ✅ f1_heatmap.png (NEW)')


# ══════════════════════════════════════════════════════════════
# FIGURE 8 (NEW): Energy & PPR Comparison
# ══════════════════════════════════════════════════════════════
def fig8_energy():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(len(MODELS))
    
    # Left: Energy per inference
    bars = ax1.bar(MODEL_SHORT, ENERGY_J, color=COLORS, edgecolor='white', width=0.55)
    for bar, val in zip(bars, ENERGY_J):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                 f'{val:.0f}J', ha='center', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Energy per Inference (Joules)')
    ax1.set_title('Estimated Energy Consumption\n(100-token inference, ~35W avg power)', fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: PPR (Performance-to-Power Ratio)
    bars2 = ax2.bar(MODEL_SHORT, PPR, color=COLORS, edgecolor='white', width=0.55)
    for bar, val in zip(bars2, PPR):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)
    ax2.set_ylabel('PPR (Accuracy% / Joules)')
    ax2.set_title('Performance-to-Power Ratio\n(Higher = more accuracy per unit energy)', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3)
    
    # Highlight best PPR
    best_idx = PPR.index(max(PPR))
    bars2[best_idx].set_edgecolor('#FFD700')
    bars2[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/energy_ppr.png')
    plt.close()
    print('  ✅ energy_ppr.png (NEW)')


# ══════════════════════════════════════════════════════════════
# FIGURE 9 (NEW): DI Floor Comparison
# ══════════════════════════════════════════════════════════════
def fig9_di_floor():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODELS))
    w = 0.32
    
    bars1 = ax.bar(x - w/2, DI_STD, w, label='Standard DI', color=COLORS_LIGHT, edgecolor='white')
    bars2 = ax.bar(x + w/2, DI_FLOOR, w, label='DI with 50% Accuracy Floor', color=COLORS, edgecolor='white')
    
    # Add values
    for i in range(len(MODELS)):
        ax.text(x[i] - w/2, DI_STD[i] + 0.01, f'{DI_STD[i]:.3f}',
                ha='center', fontsize=9, fontweight='bold')
        val = DI_FLOOR[i]
        label = f'{val:.3f}' if val > 0 else 'DI = 0'
        color = '#E74C3C' if val == 0 else '#333'
        ax.text(x[i] + w/2, max(val, 0.02) + 0.01, label,
                ha='center', fontsize=9, fontweight='bold', color=color)
    
    ax.set_ylabel('Deployability Index Score')
    ax.set_title('Effect of Minimum Accuracy Floor on DI Rankings\n(TinyLlama eliminated by 50% floor)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_SHORT)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    plt.savefig(f'{FIGURES_DIR}/di_floor_comparison.png')
    plt.close()
    print('  ✅ di_floor_comparison.png (NEW)')


# ══════════════════════════════════════════════════════════════  
# FIGURE 10 (NEW): RAM Utilization Stacked
# ══════════════════════════════════════════════════════════════
def fig10_ram():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    system_baseline = 680  # approx system RAM
    model_delta = [r - system_baseline for r in PEAK_RAM_MB]
    remaining = [16384 - r for r in PEAK_RAM_MB]
    
    x = np.arange(len(MODELS))
    w = 0.5
    
    ax.bar(x, [system_baseline]*4, w, label='System Baseline', color='#BDC3C7', edgecolor='white')
    ax.bar(x, model_delta, w, bottom=[system_baseline]*4, label='Model Weight', color=COLORS, edgecolor='white')
    ax.bar(x, remaining, w, bottom=PEAK_RAM_MB, label='Available RAM', color='#ECF0F1', edgecolor='#BDC3C7', linewidth=0.5)
    
    # Add percentage labels
    for i in range(len(MODELS)):
        pct = PEAK_RAM_MB[i] / 16384 * 100
        ax.text(x[i], PEAK_RAM_MB[i] + 200, f'{pct:.0f}% used\n({PEAK_RAM_MB[i]:,} MB)',
                ha='center', fontsize=9, fontweight='bold')
    
    ax.axhline(y=16384, color='#E74C3C', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(3.6, 16600, '16 GB Ceiling', color='#E74C3C', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('RAM (MB)')
    ax.set_title('RAM Utilization by Model\n(Q4_K_M, all models within 16GB envelope)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_SHORT)
    ax.set_ylim(0, 18000)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(f'{FIGURES_DIR}/ram_utilization.png')
    plt.close()
    print('  ✅ ram_utilization.png (NEW)')


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating all publication figures...\n')
    
    fig1_throughput_bar()
    fig2_acc_vs_tps()
    fig3_agent_overhead()
    fig4_ttft()
    fig5_di_heatmap()
    fig6_per_step_latency()
    fig7_f1_heatmap()
    fig8_energy()
    fig9_di_floor()
    fig10_ram()
    
    print(f'\n✅ All 10 figures generated in {FIGURES_DIR}/')
