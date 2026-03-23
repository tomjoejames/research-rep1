#!/usr/bin/env python3
"""
E2 — Quantization Impact
Tests: Accuracy + speed tradeoff across Q4_K_M, Q5_K_M, Q8_0 for all 4 models.
Device: Device 1 only (controlled single-device comparison)

Part A: Throughput (20 runs per config, fixed prompt)
Part B: Classification accuracy (100 emails per config)

Usage:
  python experiments/e2_quantization.py --device device1
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lib import *

FIXED_PROMPT = "Explain in 3 sentences why artificial intelligence is important for healthcare."

DEFAULT_MODELS = [
    "tinyllama", "tinyllama-q5", "tinyllama-q8",
    "phi3:mini",  "phi3-q5",      "phi3-q8",
    "qwen2.5:3b", "qwen-q5",      "qwen-q8",
    "mistral:7b",  "mistral-q5",   "mistral-q8",
]
THROUGHPUT_RUNS = 20


def classification_prompt(email_text):
    return f"""You are an email classifier. Classify the following email into exactly one category.
Categories: inquiry, complaint, feedback, spam, urgent

Respond with ONLY the category label, nothing else.

Email:
{email_text}

Category:"""


def main():
    parser = make_parser("E2: Quantization Impact")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    device = args.device

    if not check_ollama():
        print("ERROR: Ollama not running."); sys.exit(1)

    # Load email dataset
    email_file = os.path.join(os.path.dirname(__file__), "..", "data", "emails.json")
    with open(email_file) as f:
        emails = json.load(f)

    print_header(f"E2: QUANTIZATION IMPACT — {device}")
    print(f"Models: {len(models)} configs")
    print(f"Throughput runs: {THROUGHPUT_RUNS}")
    print(f"Classification samples: {len(emails)}")

    throughput_rows = []
    accuracy_rows = []

    for model in models:
        print_header(f"Model: {model}")
        load_time = load_model(model)
        if load_time < 0:
            print(f"  SKIPPING {model} — not found. Did you create it?")
            continue

        warmup(model, n=3)

        # ── Part A: Throughput ──────────────────────────────
        print(f"\n  [Part A] Throughput ({THROUGHPUT_RUNS} runs)")
        for i in range(THROUGHPUT_RUNS):
            mon = ResourceMonitor(); mon.start()
            result = run_inference(model, FIXED_PROMPT, max_tokens=100, ctx_size=2048)
            mon.stop(); stats = mon.get_stats()
            throughput_rows.append({
                "experiment": "E2-throughput", "device": device, "model": model,
                "run": i + 1,
                "total_time_s": result["total_time_s"],
                "ttft_ms": result["ttft_ms"],
                "tokens": result["tokens"],
                "tokens_per_sec": result["tokens_per_sec"],
                "peak_ram_mb": stats["peak_ram_mb"],
                "avg_cpu_pct": stats["avg_cpu_pct"],
                "load_time_s": round(load_time, 2),
            })
            if (i + 1) % 5 == 0:
                print(f"    Run {i+1}/{THROUGHPUT_RUNS}: {result['tokens_per_sec']:.1f} tok/s")

        # ── Part B: Classification Accuracy ─────────────────
        print(f"\n  [Part B] Classification ({len(emails)} emails)")
        correct_count = 0
        for j, sample in enumerate(emails):
            result = run_inference(model, classification_prompt(sample["text"]), max_tokens=10)
            predicted = result["response"].strip().lower().replace(".", "").replace(",", "")
            is_correct = sample["label"].lower() in predicted
            if is_correct:
                correct_count += 1
            accuracy_rows.append({
                "experiment": "E2-accuracy", "device": device, "model": model,
                "sample_id": sample["id"],
                "expected": sample["label"],
                "predicted": predicted,
                "correct": is_correct,
                "tokens_per_sec": result["tokens_per_sec"],
                "ttft_ms": result["ttft_ms"],
            })
            if (j + 1) % 25 == 0:
                print(f"    {j+1}/{len(emails)}: running accuracy = {correct_count/(j+1)*100:.1f}%")

        acc = correct_count / len(emails) * 100
        print(f"  → {model} accuracy: {acc:.1f}%")

        unload_model(model)
        time.sleep(15)

    # Save results
    ts = get_timestamp()
    save_csv(throughput_rows, f"{args.output_dir}/e2_throughput_{device}_{ts}.csv")
    save_csv(accuracy_rows, f"{args.output_dir}/e2_accuracy_{device}_{ts}.csv")
    print_header("E2 COMPLETE")


if __name__ == "__main__":
    main()
