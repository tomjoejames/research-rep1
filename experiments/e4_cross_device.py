#!/usr/bin/env python3
"""
E4 — Cross-Device Reproducibility
Tests: Does the i5-1334U (13th Gen) yield measurably better throughput than i5-1235U (12th Gen)?
Device: BOTH (must run identically on each device)

Runs 2 models × 30 runs with identical prompts and settings. Results are paired
across devices and analyzed for statistically significant differences.

Usage:
  python experiments/e4_cross_device.py --device device1
  python experiments/e4_cross_device.py --device device2
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lib import *

# Use 2 representative models: one small (phi3) and one large (mistral)
DEFAULT_MODELS = ["phi3:mini", "mistral:7b"]
DEFAULT_RUNS = 30

# Multiple prompts of varying lengths to test throughput at different input sizes
PROMPTS = {
    "short": "What is machine learning?",
    "medium": (
        "Explain the difference between supervised learning, unsupervised learning, "
        "and reinforcement learning. Provide a real-world example for each. "
        "Keep your response under 100 words."
    ),
    "long": (
        "Write a detailed comparison of three popular deep learning frameworks: "
        "TensorFlow, PyTorch, and JAX. For each framework, describe its primary "
        "use case, the type of projects it is best suited for, its main advantages "
        "and disadvantages, and the community support available. "
        "Focus on practical differences that would help a developer choose between them. "
        "Keep your response under 200 words."
    ),
}


def main():
    parser = make_parser("E4: Cross-Device Reproducibility")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    runs = args.runs or DEFAULT_RUNS
    device = args.device

    if not check_ollama():
        print("ERROR: Ollama not running."); sys.exit(1)

    print_header(f"E4: CROSS-DEVICE — {device}")
    print(f"Models: {models}")
    print(f"Runs per model per prompt: {runs}")
    print(f"Prompt lengths: {list(PROMPTS.keys())}")

    rows = []

    for model in models:
        print_header(f"Model: {model}")
        load_time = load_model(model)
        if load_time < 0:
            continue
        warmup(model, n=5)

        for prompt_name, prompt_text in PROMPTS.items():
            print(f"\n  Prompt: {prompt_name} ({len(prompt_text)} chars)")
            for i in range(runs):
                mon = ResourceMonitor(); mon.start()
                result = run_inference(model, prompt_text, max_tokens=150, ctx_size=2048)
                mon.stop(); stats = mon.get_stats()

                rows.append({
                    "experiment": "E4",
                    "device": device,
                    "model": model,
                    "prompt_type": prompt_name,
                    "prompt_chars": len(prompt_text),
                    "run": i + 1,
                    "total_time_s": result["total_time_s"],
                    "ttft_ms": result["ttft_ms"],
                    "tokens": result["tokens"],
                    "tokens_per_sec": result["tokens_per_sec"],
                    "peak_ram_mb": stats["peak_ram_mb"],
                    "avg_cpu_pct": stats["avg_cpu_pct"],
                    "load_time_s": round(load_time, 2),
                })
                if (i + 1) % 10 == 0:
                    print(f"    Run {i+1}/{runs}: {result['tokens_per_sec']:.1f} tok/s")

        unload_model(model)
        time.sleep(15)

    save_csv(rows, f"{args.output_dir}/e4_crossdevice_{device}_{get_timestamp()}.csv")
    print_header("E4 COMPLETE")


if __name__ == "__main__":
    main()
