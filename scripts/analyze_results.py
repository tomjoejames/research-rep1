#!/usr/bin/env python3
"""
Analyze experiment results and generate tables + figures for the paper.
Run: python analyze_results.py results/experiment_run_XXXXX.json
"""

import json
import sys
import os
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("matplotlib not found — tables will be generated but no figures.")


def load_results(filepath):
    with open(filepath) as f:
        return pd.DataFrame(json.load(f))


def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def compute_deployability_index(row, baseline_acc=85.0):
    """
    DI = w1*(A/A_baseline) + w2*(1/L_norm) + w3*(1/M_norm) + w4*CR
    Weights: accuracy=0.35, latency=0.25, memory=0.15, completion=0.25
    """
    w1, w2, w3, w4 = 0.35, 0.25, 0.15, 0.25
    acc_ratio = min(row.get("accuracy", 0) / baseline_acc, 1.0) if baseline_acc > 0 else 0
    lat_norm = max(row.get("latency_norm", 1), 0.01)
    mem_norm = max(row.get("memory_norm", 1), 0.01)
    cr = row.get("completion_rate", 0)
    return round(w1 * acc_ratio + w2 * (1 / lat_norm) + w3 * (1 / mem_norm) + w4 * cr, 4)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_json_file>")
        print("Example: python analyze_results.py results/experiment_run_20260325_143000.json")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    df = load_results(filepath)
    out_dir = os.path.dirname(filepath) or "results"
    os.makedirs(f"{out_dir}/figures", exist_ok=True)

    print(f"Loaded {len(df)} data points from {filepath}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Tasks: {sorted(df['task'].unique())}")

    # ════════════════════════════════════════════════════════════
    # TABLE 1: Throughput and Latency
    # ════════════════════════════════════════════════════════════
    print_section("TABLE 1: THROUGHPUT AND LATENCY")

    perf = df.groupby("model").agg(
        tok_s_mean=("tokens_per_sec", "mean"),
        tok_s_std=("tokens_per_sec", "std"),
        ttft_mean_ms=("ttft_ms", "mean"),
        ttft_std_ms=("ttft_ms", "std"),
        latency_p50=("total_time_s", "median"),
        latency_p90=("total_time_s", lambda x: np.percentile(x.dropna(), 90)),
        latency_p95=("total_time_s", lambda x: np.percentile(x.dropna(), 95)),
        n_samples=("total_time_s", "count"),
    ).round(2)
    print(perf.to_string())
    perf.to_csv(f"{out_dir}/table1_throughput_latency.csv")
    print(f"\nSaved: {out_dir}/table1_throughput_latency.csv")

    # ════════════════════════════════════════════════════════════
    # TABLE 2: Resource Consumption
    # ════════════════════════════════════════════════════════════
    print_section("TABLE 2: RESOURCE CONSUMPTION")

    resources = df.groupby("model").agg(
        peak_ram_mb=("peak_ram_mb", "max"),
        avg_ram_mb=("avg_ram_mb", "mean"),
        avg_cpu_pct=("avg_cpu_pct", "mean"),
        peak_cpu_pct=("peak_cpu_pct", "max"),
    ).round(1)
    print(resources.to_string())
    resources.to_csv(f"{out_dir}/table2_resources.csv")
    print(f"\nSaved: {out_dir}/table2_resources.csv")

    # Load times (one per model)
    if "load_time_s" in df.columns:
        load_times = df.groupby("model")["load_time_s"].first().round(2)
        print("\nModel Load Times (cold start):")
        print(load_times.to_string())

    # ════════════════════════════════════════════════════════════
    # TABLE 3: Task Accuracy
    # ════════════════════════════════════════════════════════════
    print_section("TABLE 3: TASK ACCURACY")

    # Classification
    cls_df = df[df["task"] == "classification"]
    if not cls_df.empty:
        cls_acc = cls_df.groupby("model")["correct"].mean().round(4) * 100
        print("Classification Accuracy (%):")
        print(cls_acc.to_string())
        print()

    # Extraction
    ext_df = df[df["task"] == "extraction"]
    if not ext_df.empty:
        ext_acc = ext_df.groupby("model")["extraction_accuracy"].mean().round(4) * 100
        print("Extraction Accuracy (%):")
        print(ext_acc.to_string())
        print()

    # Agent
    agent_df = df[df["task"] == "agent"]
    if not agent_df.empty:
        agent_stats = agent_df.groupby("model").agg(
            answer_acc_pct=("answer_correct", lambda x: round(x.mean() * 100, 1)),
            completion_rate_pct=("chain_completed", lambda x: round(x.mean() * 100, 1)),
            avg_chain_time_s=("chain_time_s", "mean"),
        ).round(2)
        print("Agent Task Results:")
        print(agent_stats.to_string())

    # Combined accuracy table
    accuracy_table = pd.DataFrame(index=sorted(df["model"].unique()))
    if not cls_df.empty:
        accuracy_table["classification_%"] = cls_acc
    if not ext_df.empty:
        accuracy_table["extraction_%"] = ext_acc
    if not agent_df.empty:
        accuracy_table["agent_acc_%"] = agent_df.groupby("model")["answer_correct"].mean() * 100
        accuracy_table["agent_completion_%"] = agent_df.groupby("model")["chain_completed"].mean() * 100
    accuracy_table = accuracy_table.round(1)
    accuracy_table.to_csv(f"{out_dir}/table3_accuracy.csv")
    print(f"\nSaved: {out_dir}/table3_accuracy.csv")

    # ════════════════════════════════════════════════════════════
    # TABLE 4: Agent Chain Depth Analysis
    # ════════════════════════════════════════════════════════════
    if not agent_df.empty and "steps_required" in agent_df.columns:
        print_section("TABLE 4: AGENT CHAIN DEPTH ANALYSIS")

        depth_stats = agent_df.groupby(["model", "steps_required"]).agg(
            accuracy=("answer_correct", lambda x: round(x.mean() * 100, 1)),
            completion=("chain_completed", lambda x: round(x.mean() * 100, 1)),
            avg_time_s=("chain_time_s", "mean"),
        ).round(2)
        print(depth_stats.to_string())
        depth_stats.to_csv(f"{out_dir}/table4_chain_depth.csv")
        print(f"\nSaved: {out_dir}/table4_chain_depth.csv")

    # ════════════════════════════════════════════════════════════
    # TABLE 5: Deployability Index
    # ════════════════════════════════════════════════════════════
    print_section("TABLE 5: DEPLOYABILITY INDEX")

    di_data = []
    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model]

        # Overall accuracy (average across tasks)
        accs = []
        m_cls = mdf[mdf["task"] == "classification"]
        if not m_cls.empty:
            accs.append(m_cls["correct"].mean() * 100)
        m_ext = mdf[mdf["task"] == "extraction"]
        if not m_ext.empty:
            accs.append(m_ext["extraction_accuracy"].mean() * 100)
        m_agent = mdf[mdf["task"] == "agent"]
        if not m_agent.empty:
            accs.append(m_agent["answer_correct"].mean() * 100)

        overall_acc = np.mean(accs) if accs else 0

        # Latency normalized (1 = fastest model)
        avg_latency = mdf["total_time_s"].mean()

        # Memory normalized (fraction of 16GB)
        peak_mem = mdf["peak_ram_mb"].max()

        # Completion rate
        cr = 1.0
        if not m_agent.empty:
            cr = m_agent["chain_completed"].mean()

        di_data.append({
            "model": model,
            "accuracy": overall_acc,
            "avg_latency_s": avg_latency,
            "peak_ram_mb": peak_mem,
            "completion_rate": cr,
        })

    di_df = pd.DataFrame(di_data)

    # Normalize latency and memory
    min_latency = di_df["avg_latency_s"].min()
    di_df["latency_norm"] = di_df["avg_latency_s"] / min_latency if min_latency > 0 else 1
    di_df["memory_norm"] = di_df["peak_ram_mb"] / (16 * 1024)  # fraction of 16GB

    di_df["DI"] = di_df.apply(compute_deployability_index, axis=1)
    di_df = di_df.sort_values("DI", ascending=False)
    di_df["rank"] = range(1, len(di_df) + 1)

    print(di_df[["model", "accuracy", "avg_latency_s", "peak_ram_mb", "completion_rate", "DI", "rank"]].to_string(index=False))
    di_df.to_csv(f"{out_dir}/table5_deployability_index.csv", index=False)
    print(f"\nSaved: {out_dir}/table5_deployability_index.csv")

    # ════════════════════════════════════════════════════════════
    # FIGURES
    # ════════════════════════════════════════════════════════════
    if HAS_PLOT:
        print_section("GENERATING FIGURES")

        plt.style.use("seaborn-v0_8-whitegrid")
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63",
                  "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
                  "#795548", "#3F51B5", "#8BC34A", "#FFC107"]

        # Figure 2: Tokens/second bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        models_sorted = perf.sort_values("tok_s_mean", ascending=True).index
        bars = ax.barh(range(len(models_sorted)),
                       perf.loc[models_sorted, "tok_s_mean"],
                       xerr=perf.loc[models_sorted, "tok_s_std"],
                       color=colors[:len(models_sorted)], edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(models_sorted)))
        ax.set_yticklabels(models_sorted, fontsize=10)
        ax.set_xlabel("Tokens per Second", fontsize=12)
        ax.set_title("Inference Throughput by Model Configuration", fontsize=14, fontweight="bold")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(f"{out_dir}/figures/fig2_throughput.png", dpi=150)
        print(f"  Saved: fig2_throughput.png")
        plt.close()

        # Figure 3: Latency box plot
        fig, ax = plt.subplots(figsize=(12, 6))
        box_data = [df[df["model"] == m]["total_time_s"].dropna().values for m in models_sorted]
        bp = ax.boxplot(box_data, vert=False, labels=models_sorted, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xlabel("Response Time (seconds)", fontsize=12)
        ax.set_title("Latency Distribution by Model Configuration", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/figures/fig3_latency_distribution.png", dpi=150)
        print(f"  Saved: fig3_latency_distribution.png")
        plt.close()

        # Figure 5: RAM consumption
        fig, ax = plt.subplots(figsize=(12, 6))
        ram_sorted = resources.sort_values("peak_ram_mb", ascending=True)
        ax.barh(range(len(ram_sorted)), ram_sorted["peak_ram_mb"],
                color=colors[:len(ram_sorted)], edgecolor="white")
        ax.axvline(x=16*1024, color="red", linestyle="--", label="16GB System RAM")
        ax.set_yticks(range(len(ram_sorted)))
        ax.set_yticklabels(ram_sorted.index, fontsize=10)
        ax.set_xlabel("Peak RAM Usage (MB)", fontsize=12)
        ax.set_title("Memory Consumption by Model Configuration", fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/figures/fig5_ram_usage.png", dpi=150)
        print(f"  Saved: fig5_ram_usage.png")
        plt.close()

        # Figure 6: Deployability Index heatmap
        if len(di_df) > 0:
            # Parse model name and quantization for heatmap axes
            di_for_heatmap = di_df.copy()
            di_for_heatmap["base_model"] = di_for_heatmap["model"].apply(lambda x: x.rsplit("-", 1)[0])
            di_for_heatmap["quant"] = di_for_heatmap["model"].apply(lambda x: x.rsplit("-", 1)[-1])

            pivot = di_for_heatmap.pivot_table(index="base_model", columns="quant", values="DI")
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, fontsize=11)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index, fontsize=11)
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        val = pivot.values[i, j]
                        if not np.isnan(val):
                            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=12, fontweight="bold")
                ax.set_title("Deployability Index (DI) Heatmap", fontsize=14, fontweight="bold")
                ax.set_xlabel("Quantization Level")
                ax.set_ylabel("Model")
                plt.colorbar(im, ax=ax, label="DI Score")
                plt.tight_layout()
                plt.savefig(f"{out_dir}/figures/fig6_DI_heatmap.png", dpi=150)
                print(f"  Saved: fig6_DI_heatmap.png")
                plt.close()

        print(f"\nAll figures saved to {out_dir}/figures/")

    # ════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════
    print_section("EXPERIMENT SUMMARY")
    print(f"Total data points:     {len(df)}")
    print(f"Models tested:         {df['model'].nunique()}")
    print(f"Task types:            {df['task'].nunique()}")
    print(f"Best throughput:       {perf['tok_s_mean'].max():.1f} tok/s ({perf['tok_s_mean'].idxmax()})")
    print(f"Lowest latency (p50):  {perf['latency_p50'].min():.2f}s ({perf['latency_p50'].idxmin()})")
    print(f"Best DI score:         {di_df['DI'].max():.4f} ({di_df.iloc[0]['model']})")


if __name__ == "__main__":
    main()
