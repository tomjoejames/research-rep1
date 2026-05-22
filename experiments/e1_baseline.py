#!/usr/bin/env python3
"""
E1 — Baseline Inference Benchmark
Tests: Throughput (tok/s), TTFT, latency across 4 models at Q4_K_M default.
Device: BOTH (run on each device separately, set --device flag)
Runs: 20 per model (configurable)

Usage:
  python experiments/e1_baseline.py --device device1
  python experiments/e1_baseline.py --device device2
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lib import *

FIXED_PROMPT = "Explain in 3 sentences why artificial intelligence is important for healthcare."
DEFAULT_MODELS = ["tinyllama", "phi3:mini", "qwen2.5:3b", "mistral:7b"]
DEFAULT_RUNS = 20


def main():
    parser = make_parser("E1: Baseline Inference Benchmark")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    runs = args.runs or DEFAULT_RUNS
    device = args.device

    if not check_ollama():
        print("ERROR: Ollama not running. Start with: ollama serve &")
        sys.exit(1)

    print_header(f"E1: BASELINE INFERENCE — {device}")
    print(f"Models: {models}")
    print(f"Runs per model: {runs}")

    rows = []

    for model in models:
        print_header(f"Model: {model}")
        load_time = load_model(model)
        if load_time < 0:
            continue

        warmup(model, n=3)

        for i in range(runs):
            mon = ResourceMonitor()
            mon.start()
            result = run_inference(model, FIXED_PROMPT, max_tokens=100, ctx_size=2048)
            mon.stop()
            stats = mon.get_stats()

            row = {
                "experiment": "E1",
                "device": device,
                "model": model,
                "run": i + 1,
                "total_time_s": result["total_time_s"],
                "ttft_ms": result["ttft_ms"],
                "tokens": result["tokens"],
                "tokens_per_sec": result["tokens_per_sec"],
                "peak_ram_mb": stats["peak_ram_mb"],
                "avg_cpu_pct": stats["avg_cpu_pct"],
                "load_time_s": round(load_time, 2),
            }
            rows.append(row)
            print(f"  Run {i+1:02d}/{runs}: {result['tokens_per_sec']:.1f} tok/s | "
                  f"TTFT {result['ttft_ms']:.0f}ms | {result['total_time_s']:.2f}s")

        unload_model(model)
        time.sleep(10)

    outfile = f"{args.output_dir}/e1_baseline_{device}_{get_timestamp()}.csv"
    save_csv(rows, outfile)
    print_header("E1 COMPLETE")


if __name__ == "__main__":
    main()
