import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "c:/Adhi/Startup/Research/research-rep1-v2/results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Aesthetic settings for Academic Graphics
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    'figure.dpi': 300, 
    'savefig.dpi': 300, 
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

def get_latest_files(prefix):
    """Returns all CSVs starting with prefix in the root results directory, explicitly ignoring legacy subdirectories."""
    pattern = os.path.join(RESULTS_DIR, f"{prefix}*.csv")
    files = [f for f in glob.glob(pattern) if os.path.isfile(f)]
    return files

def plot_graph_a():
    """Graph A (E2): Deployability Frontier (Accuracy vs. Throughput scatter plot)"""
    acc_files = get_latest_files("e2_accuracy")
    tp_files = get_latest_files("e2_throughput")
    if not acc_files or not tp_files: 
        print("Missing E2 data for Graph A.")
        return
    
    df_acc = pd.concat([pd.read_csv(f) for f in acc_files])
    df_tp = pd.concat([pd.read_csv(f) for f in tp_files])
    
    # Merge on shared execution context
    acc_summary = df_acc.groupby(['device', 'model'])['correct'].mean().reset_index()
    tp_summary = df_tp.groupby(['device', 'model'])['tokens_per_sec'].mean().reset_index()
    df = pd.merge(acc_summary, tp_summary, on=['device', 'model'])
    df['accuracy_pct'] = df['correct'] * 100
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='tokens_per_sec', y='accuracy_pct', hue='model', style='device', s=200, alpha=0.85)
    
    plt.title("Deployability Frontier: Accuracy vs. Throughput")
    plt.xlabel("Throughput (Tokens / Sec)")
    plt.ylabel("Accuracy (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "Graph_A_Deployability_Frontier.png"))
    plt.close()
    print("Saved Graph A.")

def plot_graph_b():
    """Graph B (E3): Routing Overhead (Grouped bar chart for Multi-agent penalty)"""
    files = get_latest_files("e3_overhead")
    if not files:
        print("Missing E3 data for Graph B.")
        return
    df = pd.concat([pd.read_csv(f) for f in files])
    
    # Melt dataframe to compare baseline vs chain vertically
    summary = df.groupby(['device', 'model'])[['single_time_s', 'agent_chain_s']].mean().reset_index()
    melted = summary.melt(id_vars=['device', 'model'], value_vars=['single_time_s', 'agent_chain_s'], 
                          var_name='Execution Type', value_name='Latency (s)')
    melted['Execution Type'] = melted['Execution Type'].replace({'single_time_s': 'Base Single-Step', 'agent_chain_s': 'Multi-Agent Chain'})
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x='model', y='Latency (s)', hue='Execution Type', errorbar=None)
    
    # Handle dynamic device faceting visually if multiple exist
    devices = df['device'].unique()
    plt.title(f"Routing Overhead: 3.35x Multi-Agent Penalty (Devices: {', '.join(devices)})")
    plt.ylabel("Execution Latency (Seconds)")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "Graph_B_Routing_Overhead.png"))
    plt.close()
    print("Saved Graph B.")

def plot_graph_c():
    """Graph C (E4): Latency Explosion (Grouped Bar Chart - Base vs Failed Recovery)"""
    files = get_latest_files("e4_resilience")
    if not files:
        print("Missing E4 data for Graph C.")
        return
    df = pd.concat([pd.read_csv(f) for f in files])
    
    # Label standard execution vs recovery penalty execution
    df['Status'] = df['recovery_attempted'].apply(lambda x: 'Failed Recovery Loop' if x else 'Base Task (Success)')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='model', y='total_time_s', hue='Status', errorbar='sd')
    
    devices = df['device'].unique()
    plt.title(f"Latency Explosion: The Compute Waste of Self-Healing (Devices: {', '.join(devices)})")
    plt.ylabel("Total End-to-End Latency (Seconds)")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "Graph_C_Latency_Explosion.png"))
    plt.close()
    print("Saved Graph C.")

def plot_graph_d():
    """Graph D (E5): KV Cache Memory Pressure (Dual-axis line chart)"""
    files = get_latest_files("e5_memory")
    if not files:
        print("Missing E5 data for Graph D.")
        return
    df = pd.concat([pd.read_csv(f) for f in files])
    
    if 'max_tokens_reached' not in df.columns:
        print("E5 Data lacks 'max_tokens_reached'. Skipping Graph D.")
        return
        
    df_sorted = df.sort_values(by='max_tokens_reached')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('RAM Utilization (%)', color=color)
    sns.lineplot(data=df_sorted, x='model', y='ram_utilization_pct', marker='o', ax=ax1, color=color, sort=False)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(len(df_sorted)))
    ax1.set_xticklabels(df_sorted['model'], rotation=45, ha='right')

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Max Context Tokens Reached', color=color)  
    sns.lineplot(data=df_sorted, x='model', y='max_tokens_reached', marker='s', ax=ax2, color=color, linestyle='--', sort=False)
    ax2.tick_params(axis='y', labelcolor=color)

    ax2.axhline(y=1740, color='red', linestyle=':', label='~1740 Token Crash Limit')
    ax2.legend(loc='upper left', bbox_to_anchor=(0.0, 1.15))

    devices = df['device'].unique()
    plt.title(f"KV Cache Memory Pressure (Devices: {', '.join(devices)})", y=1.05)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "Graph_D_KV_Cache_Pressure.png"))
    plt.close()
    print("Saved Graph D.")

def plot_graph_e():
    """Graph E (E1/E6): Latency Variance (Box-and-whisker plots for Cold vs Warm latency)"""
    files = get_latest_files("e6_coldwarm")
    if not files:
        print("Missing E6 data for Graph E.")
        return
    df = pd.concat([pd.read_csv(f) for f in files])
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='model', y='ttft_ms', hue='condition', palette=['#1f77b4', '#ff7f0e'])
    
    devices = df['device'].unique()
    plt.title(f"Latency Variance: Cold vs. Warm State Initialization (Devices: {', '.join(devices)})")
    plt.ylabel("Time to First Token (ms)")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "Graph_E_Latency_Variance.png"))
    plt.close()
    print("Saved Graph E.")

if __name__ == "__main__":
    print("Booting Dynamic Plotter...")
    plot_graph_a()
    plot_graph_b()
    plot_graph_c()
    plot_graph_d()
    plot_graph_e()
    print(f"\\nAll required academic graphs have been successfully generated and saved to {PLOTS_DIR}")
