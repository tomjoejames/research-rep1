#!/usr/bin/env python3
"""
Deep analysis of experiment results for the research paper.
Computes: per-category F1, per-step agent latency, energy estimates,
confusion matrices, DI with accuracy floor, and summary statistics.
"""

import csv
import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Helpers ───────────────────────────────────────────────────────────────────

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def std(vals):
    m = mean(vals)
    return (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5 if vals else 0.0

def p50(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n else 0

def p90(vals):
    s = sorted(vals)
    idx = int(0.9 * len(s))
    return s[min(idx, len(s) - 1)] if s else 0

def p95(vals):
    s = sorted(vals)
    idx = int(0.95 * len(s))
    return s[min(idx, len(s) - 1)] if s else 0


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

# ─── 1. Per-Category F1 Breakdown (E2 Accuracy) ──────────────────────────────

def analyze_e2_accuracy():
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Per-Category Email Classification Breakdown")
    print("=" * 70)

    files = sorted(RESULTS_DIR.glob("e2_accuracy_device1*.csv"))
    if not files:
        print("  No E2 accuracy files found!")
        return

    rows = read_csv(files[0])

    # Only analyze models that actually produced output (Q4_K_M base models)
    q4_models = ["tinyllama", "phi3:mini", "qwen2.5:3b", "mistral:7b"]
    categories = ["inquiry", "complaint", "feedback", "spam", "urgent"]

    results = {}

    for model in q4_models:
        model_rows = [r for r in rows if r["model"] == model]
        if not model_rows:
            continue

        # Build confusion matrix
        confusion = defaultdict(lambda: defaultdict(int))
        per_cat_correct = defaultdict(int)
        per_cat_total = defaultdict(int)

        for r in model_rows:
            expected = r["expected"].strip()
            predicted = r.get("predicted", "").strip().lower()
            correct = r.get("correct", "").strip() == "True"

            per_cat_total[expected] += 1
            if correct:
                per_cat_correct[expected] += 1

            # Map predicted to category
            pred_cat = None
            for cat in categories:
                if cat in predicted:
                    pred_cat = cat
                    break
            if pred_cat is None:
                pred_cat = "other"

            confusion[expected][pred_cat] += 1

        overall_correct = sum(per_cat_correct.values())
        overall_total = sum(per_cat_total.values())
        overall_acc = overall_correct / overall_total * 100 if overall_total else 0

        # Compute per-category precision, recall, F1
        cat_metrics = {}
        for cat in categories:
            tp = confusion[cat].get(cat, 0)
            fp = sum(confusion[other].get(cat, 0) for other in categories if other != cat)
            fn = per_cat_total[cat] - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            cat_metrics[cat] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "correct": per_cat_correct[cat],
                "total": per_cat_total[cat],
                "accuracy": per_cat_correct[cat] / per_cat_total[cat] * 100 if per_cat_total[cat] else 0,
            }

        results[model] = {
            "overall_accuracy": overall_acc,
            "overall_correct": overall_correct,
            "overall_total": overall_total,
            "per_category": cat_metrics,
            "confusion": dict(confusion),
        }

        # Print results
        print(f"\n  ### {model} (Accuracy: {overall_acc:.1f}%, {overall_correct}/{overall_total})")
        print(f"  {'Category':<12} {'Acc%':>6} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Correct':>8}")
        print(f"  {'─'*12} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
        for cat in categories:
            m = cat_metrics[cat]
            print(f"  {cat:<12} {m['accuracy']:>5.1f}% {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['correct']:>4}/{m['total']:<3}")

        # Confusion matrix
        print(f"\n  Confusion Matrix (rows=expected, cols=predicted):")
        header = "  " + f"{'':>12}" + "".join(f"{c[:6]:>8}" for c in categories + ["other"])
        print(header)
        for exp_cat in categories:
            row_vals = "".join(f"{confusion[exp_cat].get(c, 0):>8}" for c in categories + ["other"])
            print(f"  {exp_cat:>12}{row_vals}")

    return results


# ─── 2. Per-Step Agent Latency Analysis (E3) ─────────────────────────────────

def analyze_e3_per_step():
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Per-Step Agent Latency (E3)")
    print("=" * 70)

    files = sorted(RESULTS_DIR.glob("e3_overhead_device1*.csv"))
    if not files:
        print("  No E3 files found!")
        return

    rows = read_csv(files[0])
    models = ["tinyllama", "phi3:mini", "qwen2.5:3b", "mistral:7b"]

    results = {}
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        if not model_rows:
            continue

        # Exclude run 1 (cold start)
        warm_rows = [r for r in model_rows if int(r["run"]) > 1]

        step1_times = [float(r["step1_s"]) for r in warm_rows]
        step2_times = [float(r["step2_s"]) for r in warm_rows]
        step3_times = [float(r["step3_s"]) for r in warm_rows]
        overhead_ratios = [float(r["overhead_ratio"]) for r in warm_rows]
        single_times = [float(r["single_time_s"]) for r in warm_rows]

        results[model] = {
            "step1_mean": mean(step1_times),
            "step2_mean": mean(step2_times),
            "step3_mean": mean(step3_times),
            "step1_std": std(step1_times),
            "step2_std": std(step2_times),
            "step3_std": std(step3_times),
            "overhead_mean": mean(overhead_ratios),
            "overhead_std": std(overhead_ratios),
            "single_mean": mean(single_times),
        }

        r = results[model]
        print(f"\n  ### {model}")
        print(f"  Single-step mean: {r['single_mean']:.2f}s")
        print(f"  Step 1: {r['step1_mean']:.2f} ± {r['step1_std']:.2f}s")
        print(f"  Step 2: {r['step2_mean']:.2f} ± {r['step2_std']:.2f}s")
        print(f"  Step 3: {r['step3_mean']:.2f} ± {r['step3_std']:.2f}s")
        print(f"  Overhead ratio: {r['overhead_mean']:.3f} ± {r['overhead_std']:.3f}")

        # Check for progressive slowdown
        if r['step3_mean'] > r['step1_mean'] * 1.1:
            pct = (r['step3_mean'] - r['step1_mean']) / r['step1_mean'] * 100
            print(f"  ⚠️  Step 3 is {pct:.1f}% slower than Step 1 (context accumulation effect)")
        else:
            print(f"  ✅ No significant progressive slowdown detected")

    return results


# ─── 3. Energy Estimation ────────────────────────────────────────────────────

def analyze_energy():
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Energy Estimation (i5-1235U)")
    print("=" * 70)
    print("  Using i5-1235U TDP: PBP=15W, MTP=55W")
    print("  Estimated inference power ≈ 35W (mid-range under ~48% CPU load)")

    EST_POWER_W = 35  # Estimated average during inference

    # E1 baseline latency (warm runs, 100 tokens)
    e1_data = {
        "TinyLlama-1.1B": {"latency_s": 6.08, "tokens": 100},
        "Phi-3-mini-3.8B": {"latency_s": 21.28, "tokens": 100},
        "Qwen2.5-3B": {"latency_s": 12.89, "tokens": 100},
        "Mistral-7B": {"latency_s": 27.69, "tokens": 100},
    }

    # Accuracy from E2
    accuracy = {
        "TinyLlama-1.1B": 22,
        "Phi-3-mini-3.8B": 86,
        "Qwen2.5-3B": 80,
        "Mistral-7B": 85,
    }

    print(f"\n  {'Model':<20} {'Latency':>8} {'Energy(J)':>10} {'J/token':>8} {'PPR(%/J)':>9} {'kWh/1K':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*8} {'─'*9} {'─'*8}")

    results = {}
    for model, data in e1_data.items():
        energy_j = EST_POWER_W * data["latency_s"]
        j_per_token = energy_j / data["tokens"]
        ppr = accuracy[model] / energy_j  # Performance-to-Power Ratio (%/J)
        kwh_per_1k = (j_per_token * 1000) / 3600000 * 1000  # mWh per token * 1000

        results[model] = {
            "energy_j": energy_j,
            "j_per_token": j_per_token,
            "ppr": ppr,
            "kwh_per_1k_tasks": kwh_per_1k,
        }

        print(f"  {model:<20} {data['latency_s']:>7.2f}s {energy_j:>9.1f}J {j_per_token:>7.2f} {ppr:>8.4f} {kwh_per_1k:>7.4f}")

    # Comparison with Huang & Wang RTX A6000 data
    print(f"\n  Comparison with Huang & Wang (IJCNN 2025) RTX A6000 data:")
    print(f"  qwen2.5:3b on A6000: ~558J, PPR=0.163 %/J, F1=91.0%")
    print(f"  qwen2.5:3b on i5 CPU: ~{results['Qwen2.5-3B']['energy_j']:.0f}J, PPR={results['Qwen2.5-3B']['ppr']:.4f} %/J, Acc=80%")
    print(f"  → CPU uses ~{results['Qwen2.5-3B']['energy_j']/558:.1f}× less energy per inference (lower power, longer duration)")

    return results


# ─── 4. DI with Minimum Accuracy Floor ───────────────────────────────────────

def analyze_di_with_floor():
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Deployability Index with Minimum Accuracy Floor")
    print("=" * 70)

    A_baseline = 85.0
    w1, w2, w3, w4 = 0.35, 0.25, 0.15, 0.25
    MIN_ACC_FLOOR = 50.0  # Models below this get DI = 0

    configs = [
        {"model": "TinyLlama-1.1B", "accuracy": 22, "latency": 6.08, "peak_ram": 1360, "completion": 100},
        {"model": "Phi-3-mini-3.8B", "accuracy": 86, "latency": 21.28, "peak_ram": 4306, "completion": 100},
        {"model": "Qwen2.5-3B", "accuracy": 80, "latency": 12.89, "peak_ram": 2735, "completion": 100},
        {"model": "Mistral-7B", "accuracy": 85, "latency": 27.69, "peak_ram": 5407, "completion": 100},
    ]

    fastest_latency = min(c["latency"] for c in configs)
    max_ram = 16384  # 16 GB

    print(f"\n  Standard DI (no floor) vs DI with {MIN_ACC_FLOOR}% accuracy floor:")
    print(f"  {'Model':<20} {'DI (std)':>10} {'DI (floor)':>12} {'Rank(std)':>10} {'Rank(floor)':>12}")
    print(f"  {'─'*20} {'─'*10} {'─'*12} {'─'*10} {'─'*12}")

    for c in configs:
        a_norm = c["accuracy"] / A_baseline
        l_norm = c["latency"] / fastest_latency
        m_norm = c["peak_ram"] / max_ram
        cr = c["completion"] / 100.0

        di_std = w1 * a_norm + w2 * (1 / l_norm) + w3 * (1 / m_norm) + w4 * cr

        if c["accuracy"] < MIN_ACC_FLOOR:
            di_floor = 0.0
        else:
            di_floor = di_std

        c["di_std"] = di_std
        c["di_floor"] = di_floor

    # Rank
    by_std = sorted(configs, key=lambda x: x["di_std"], reverse=True)
    by_floor = sorted(configs, key=lambda x: x["di_floor"], reverse=True)

    for c in configs:
        rank_std = [x["model"] for x in by_std].index(c["model"]) + 1
        rank_floor = [x["model"] for x in by_floor].index(c["model"]) + 1
        print(f"  {c['model']:<20} {c['di_std']:>10.4f} {c['di_floor']:>12.4f} {rank_std:>10} {rank_floor:>12}")

    print(f"\n  Key change: TinyLlama drops from rank {[x['model'] for x in by_std].index('TinyLlama-1.1B')+1} → {[x['model'] for x in by_floor].index('TinyLlama-1.1B')+1} with accuracy floor")
    print(f"  This prevents gaming the DI with a fast-but-useless model.")


# ─── 5. E1 Summary Statistics (Enhanced) ─────────────────────────────────────

def analyze_e1_enhanced():
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Enhanced E1 Baseline Statistics")
    print("=" * 70)

    files = sorted(RESULTS_DIR.glob("e1_baseline_device1*.csv"))
    if not files:
        print("  No E1 files found!")
        return

    rows = read_csv(files[0])
    models = ["tinyllama", "phi3:mini", "qwen2.5:3b", "mistral:7b"]

    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        if not model_rows:
            continue

        # Warm runs (exclude run 1)
        warm = [r for r in model_rows if int(r["run"]) > 1]

        tps_vals = [float(r["tokens_per_sec"]) for r in warm if float(r["tokens_per_sec"]) > 0]
        ttft_vals = [float(r["ttft_ms"]) for r in warm]
        latency_vals = [float(r["total_time_s"]) for r in warm]
        ram_vals = [float(r["peak_ram_mb"]) for r in warm]
        cpu_vals = [float(r["avg_cpu_pct"]) for r in warm]

        print(f"\n  ### {model} (n={len(warm)} warm runs)")
        print(f"  Throughput:  {mean(tps_vals):.2f} ± {std(tps_vals):.2f} tok/s  (CV={std(tps_vals)/mean(tps_vals)*100:.1f}%)")
        print(f"  TTFT:        {mean(ttft_vals):.1f} ± {std(ttft_vals):.1f} ms  [p50={p50(ttft_vals):.1f}, p90={p90(ttft_vals):.1f}, p95={p95(ttft_vals):.1f}]")
        print(f"  Latency:     {mean(latency_vals):.2f} ± {std(latency_vals):.2f} s  [p50={p50(latency_vals):.2f}, p90={p90(latency_vals):.2f}, p95={p95(latency_vals):.2f}]")
        print(f"  Peak RAM:    {mean(ram_vals):.0f} ± {std(ram_vals):.0f} MB")
        print(f"  Avg CPU:     {mean(cpu_vals):.1f} ± {std(cpu_vals):.1f}%")

        # Cold start (run 1)
        cold = [r for r in model_rows if int(r["run"]) == 1]
        if cold:
            cold_ttft = float(cold[0]["ttft_ms"])
            warm_ttft = mean(ttft_vals)
            reduction = (cold_ttft - warm_ttft) / cold_ttft * 100
            print(f"  Cold TTFT:   {cold_ttft:.1f} ms → Warm: {warm_ttft:.1f} ms ({reduction:.1f}% reduction)")


# ─── 6. E6 Cold vs Warm Summary ─────────────────────────────────────────────

def analyze_e6_summary():
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Cold vs Warm TTFT Summary (E6)")
    print("=" * 70)

    files = sorted(RESULTS_DIR.glob("e6_coldwarm_device1*.csv"))
    if not files:
        print("  No E6 files found!")
        return

    rows = read_csv(files[0])
    models = ["tinyllama", "phi3:mini", "qwen2.5:3b", "mistral:7b"]

    for model in models:
        cold_rows = [r for r in rows if r["model"] == model and r["condition"] == "cold"]
        warm_rows = [r for r in rows if r["model"] == model and r["condition"] == "warm"]

        if not cold_rows or not warm_rows:
            continue

        cold_ttft = [float(r["ttft_ms"]) for r in cold_rows]
        warm_ttft = [float(r["ttft_ms"]) for r in warm_rows[1:]]  # Exclude warm run 1
        cold_tps = [float(r["tokens_per_sec"]) for r in cold_rows]
        warm_tps = [float(r["tokens_per_sec"]) for r in warm_rows[1:]]

        print(f"\n  ### {model}")
        print(f"  Cold TTFT: {mean(cold_ttft):.0f} ± {std(cold_ttft):.0f} ms (n={len(cold_ttft)})")
        print(f"  Warm TTFT: {mean(warm_ttft):.0f} ± {std(warm_ttft):.0f} ms (n={len(warm_ttft)})")
        print(f"  Reduction: {(1 - mean(warm_ttft)/mean(cold_ttft))*100:.1f}%")
        print(f"  Cold tok/s: {mean(cold_tps):.2f}  |  Warm tok/s: {mean(warm_tps):.2f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Deep Analysis: LLM Agent Deployability Research Paper             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    e2_results = analyze_e2_accuracy()
    e3_results = analyze_e3_per_step()
    energy_results = analyze_energy()
    analyze_di_with_floor()
    analyze_e1_enhanced()
    analyze_e6_summary()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
