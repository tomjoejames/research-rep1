#!/usr/bin/env python3
"""
E5 — Memory Pressure Analysis
Tests: RAM time-series during model load + inference for all 12 configs.
       Detects swap usage, OOM risk, and steady-state vs peak memory.
Device: Device 1 only

Samples RAM/CPU at 100ms during:
  1. Pre-load baseline (2s)
  2. Model load phase
  3. Active inference (5 runs)
  4. Post-unload recovery (5s)

Usage:
  python experiments/e5_memory.py --device device1
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lib import *

DEFAULT_MODELS = [
    "tinyllama", "tinyllama-q5", "tinyllama-q8",
    "phi3:mini",  "phi3-q5",      "phi3-q8",
    "qwen2.5:3b", "qwen-q5",      "qwen-q8",
    "mistral:7b",  "mistral-q5",   "mistral-q8",
]

INFERENCE_PROMPT = "Explain quantum computing in 5 sentences."
INFERENCE_RUNS = 5


def get_swap_mb():
    """Get current swap usage in MB."""
    swap = psutil.swap_memory()
    return round(swap.used / (1024**2), 1)


def main():
    parser = make_parser("E5: Memory Pressure Analysis")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    device = args.device

    if not check_ollama():
        print("ERROR: Ollama not running."); sys.exit(1)

    print_header(f"E5: MEMORY PRESSURE — {device}")
    total_ram = round(psutil.virtual_memory().total / (1024**2), 0)
    print(f"System RAM: {total_ram} MB")
    print(f"Models: {len(models)} configs")

    summary_rows = []
    all_timeseries = {}

    for model in models:
        print_header(f"Model: {model}")

        # ── Phase 1: Pre-load baseline ───────────────────
        print("  Phase 1: Baseline measurement (2s)...")
        baseline_ram = psutil.virtual_memory().used / (1024**2)
        baseline_swap = get_swap_mb()
        time.sleep(2)

        # ── Phase 2: Model load with monitoring ──────────
        print("  Phase 2: Model load...")
        mon_load = ResourceMonitor(); mon_load.start()
        load_time = load_model(model)
        mon_load.stop()

        if load_time < 0:
            print(f"  SKIPPING {model}")
            continue

        load_stats = mon_load.get_stats()
        post_load_ram = psutil.virtual_memory().used / (1024**2)
        post_load_swap = get_swap_mb()

        # ── Phase 3: Inference with monitoring ───────────
        print(f"  Phase 3: Inference ({INFERENCE_RUNS} runs)...")
        mon_infer = ResourceMonitor(); mon_infer.start()
        inference_times = []

        for i in range(INFERENCE_RUNS):
            result = run_inference(model, INFERENCE_PROMPT, max_tokens=100, ctx_size=2048)
            inference_times.append(result["total_time_s"])
            print(f"    Run {i+1}: {result['tokens_per_sec']:.1f} tok/s | {result['total_time_s']:.2f}s")

        mon_infer.stop()
        infer_stats = mon_infer.get_stats()
        peak_infer_swap = get_swap_mb()

        # ── Phase 4: Unload and recovery ─────────────────
        print("  Phase 4: Unload + recovery (5s)...")
        unload_model(model)
        time.sleep(5)
        post_unload_ram = psutil.virtual_memory().used / (1024**2)
        post_unload_swap = get_swap_mb()

        # ── Record summary ───────────────────────────────
        model_ram_delta = post_load_ram - baseline_ram
        swap_used = peak_infer_swap > baseline_swap + 10  # >10MB swap = swap is active

        summary_rows.append({
            "experiment": "E5",
            "device": device,
            "model": model,
            "baseline_ram_mb": round(baseline_ram, 1),
            "post_load_ram_mb": round(post_load_ram, 1),
            "model_ram_delta_mb": round(model_ram_delta, 1),
            "peak_inference_ram_mb": infer_stats["peak_ram_mb"],
            "avg_inference_ram_mb": infer_stats["avg_ram_mb"],
            "peak_inference_cpu_pct": infer_stats["peak_cpu_pct"],
            "avg_inference_cpu_pct": infer_stats["avg_cpu_pct"],
            "baseline_swap_mb": baseline_swap,
            "peak_swap_mb": peak_infer_swap,
            "swap_active": swap_used,
            "load_time_s": round(load_time, 2),
            "avg_inference_s": round(sum(inference_times) / len(inference_times), 4),
            "post_unload_ram_mb": round(post_unload_ram, 1),
            "ram_leaked_mb": round(post_unload_ram - baseline_ram, 1),
            "system_total_ram_mb": total_ram,
            "ram_utilization_pct": round(infer_stats["peak_ram_mb"] / total_ram * 100, 1),
        })

        # Store time-series for this model
        all_timeseries[model] = {
            "load_phase": mon_load.get_timeseries(),
            "inference_phase": mon_infer.get_timeseries(),
        }

        print(f"  → RAM delta: {model_ram_delta:.0f}MB | Peak: {infer_stats['peak_ram_mb']:.0f}MB "
              f"| Swap: {'YES ⚠️' if swap_used else 'No'} | Utilization: {infer_stats['peak_ram_mb']/total_ram*100:.0f}%")

        time.sleep(10)

    ts = get_timestamp()
    save_csv(summary_rows, f"{args.output_dir}/e5_memory_{device}_{ts}.csv")
    save_json(all_timeseries, f"{args.output_dir}/e5_timeseries_{device}_{ts}.json")
    print_header("E5 COMPLETE")


if __name__ == "__main__":
    main()
