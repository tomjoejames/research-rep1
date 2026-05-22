#!/usr/bin/env python3
"""
E6 — Cold-Start vs Warm Inference
Tests: Is TTFT significantly lower for warm inferences vs cold first inference?
Device: BOTH

Protocol:
  For each model:
    1. Unload model fully
    2. Cold-start inference (record TTFT and latency) — repeat 5 times (full unload between each)
    3. Load model once, then do 20 warm inferences (record TTFT and latency)
    4. Compare cold vs warm distributions

Usage:
  python experiments/e6_coldwarm.py --device device1
  python experiments/e6_coldwarm.py --device device2
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lib import *

DEFAULT_MODELS = ["tinyllama", "phi3:mini", "qwen2.5:3b", "mistral:7b"]
COLD_RUNS = 5
WARM_RUNS = 20
PROMPT = "What are three benefits of renewable energy?"


def main():
    parser = make_parser("E6: Cold-Start vs Warm Inference")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    device = args.device

    if not check_ollama():
        print("ERROR: Ollama not running."); sys.exit(1)

    print_header(f"E6: COLD vs WARM — {device}")

    rows = []

    for model in models:
        print_header(f"Model: {model}")

        # ── Cold-start runs ──────────────────────────────
        print(f"\n  [Cold Start] {COLD_RUNS} runs (full unload between each)")
        for i in range(COLD_RUNS):
            # Ensure model is fully unloaded
            unload_model(model)
            time.sleep(10)

            # Cold inference — model load + first inference combined
            mon = ResourceMonitor(); mon.start()
            cold_start = time.perf_counter()

            # This forces a cold load
            result = run_inference(model, PROMPT, max_tokens=80, ctx_size=2048)

            cold_total = time.perf_counter() - cold_start
            mon.stop(); stats = mon.get_stats()

            rows.append({
                "experiment": "E6",
                "device": device,
                "model": model,
                "condition": "cold",
                "run": i + 1,
                "total_time_s": round(cold_total, 4),
                "ttft_ms": result["ttft_ms"],
                "tokens": result["tokens"],
                "tokens_per_sec": result["tokens_per_sec"],
                "peak_ram_mb": stats["peak_ram_mb"],
                "avg_cpu_pct": stats["avg_cpu_pct"],
            })
            print(f"    Cold {i+1}: TTFT={result['ttft_ms']:.0f}ms | Total={cold_total:.2f}s")

        # ── Warm runs ────────────────────────────────────
        print(f"\n  [Warm] {WARM_RUNS} runs (model already loaded)")
        # Ensure model is loaded
        load_model(model)
        warmup(model, n=3)

        for i in range(WARM_RUNS):
            mon = ResourceMonitor(); mon.start()
            result = run_inference(model, PROMPT, max_tokens=80, ctx_size=2048)
            mon.stop(); stats = mon.get_stats()

            rows.append({
                "experiment": "E6",
                "device": device,
                "model": model,
                "condition": "warm",
                "run": i + 1,
                "total_time_s": result["total_time_s"],
                "ttft_ms": result["ttft_ms"],
                "tokens": result["tokens"],
                "tokens_per_sec": result["tokens_per_sec"],
                "peak_ram_mb": stats["peak_ram_mb"],
                "avg_cpu_pct": stats["avg_cpu_pct"],
            })
            if (i + 1) % 5 == 0:
                print(f"    Warm {i+1}: TTFT={result['ttft_ms']:.0f}ms | {result['tokens_per_sec']:.1f} tok/s")

        unload_model(model)
        time.sleep(10)

    save_csv(rows, f"{args.output_dir}/e6_coldwarm_{device}_{get_timestamp()}.csv")
    print_header("E6 COMPLETE")


if __name__ == "__main__":
    main()
