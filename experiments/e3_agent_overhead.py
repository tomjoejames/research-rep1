#!/usr/bin/env python3
"""
E3 — Agent vs Non-Agent Overhead
Tests: Is 3-step agent latency > 3x single-step latency? (super-linear overhead)
Device: Device 1 (primary), Device 2 for validation of 1 model

Compares:
  - Single-step: one prompt → one response (all information in one prompt)
  - 3-step agent: lookup → calculate → format (sequential chain)

Usage:
  python experiments/e3_agent_overhead.py --device device1
  python experiments/e3_agent_overhead.py --device device2 --models "phi3:mini" --runs 20
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lib import *

DEFAULT_MODELS = ["tinyllama", "phi3:mini", "qwen2.5:3b", "mistral:7b"]
DEFAULT_RUNS = 30

# ── Prompts ────────────────────────────────────────────────────
SINGLE_PROMPT = """You are a billing assistant. A customer bought 5 units of Widget-X at Rs.350 each.
Calculate: subtotal, 18% GST, and total cost.
Respond with ONLY three lines:
Subtotal: [amount]
GST: [amount]
Total: [amount]"""

AGENT_STEP1 = """You are a lookup assistant. A customer is asking about Widget-X.
Product Database:
  Widget-X: Rs.350 per unit

The customer wants 5 units. Look up the price.
Respond with ONLY:
Product: Widget-X
Price: 350
Quantity: 5"""

AGENT_STEP2_TEMPLATE = """You are a calculator assistant.
Product info: {step1}
Calculate: subtotal (quantity × price) and 18% GST.
Respond with ONLY:
Subtotal: [amount]
GST: [amount]"""

AGENT_STEP3_TEMPLATE = """You are a formatter assistant.
Calculation: {step2}
Format these results as a clean one-line invoice summary.
Respond with ONLY the formatted invoice line."""


def main():
    parser = make_parser("E3: Agent vs Non-Agent Overhead")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    runs = args.runs or DEFAULT_RUNS
    device = args.device

    if not check_ollama():
        print("ERROR: Ollama not running."); sys.exit(1)

    print_header(f"E3: AGENT OVERHEAD — {device}")
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
            # ── Single-step inference ────────────────────
            mon1 = ResourceMonitor(); mon1.start()
            single = run_inference(model, SINGLE_PROMPT, max_tokens=50, ctx_size=2048)
            mon1.stop(); s1 = mon1.get_stats()

            # ── 3-step agent chain ───────────────────────
            mon2 = ResourceMonitor(); mon2.start()
            chain_start = time.perf_counter()

            step1 = run_inference(model, AGENT_STEP1, max_tokens=30, ctx_size=2048)
            step2 = run_inference(model,
                AGENT_STEP2_TEMPLATE.format(step1=step1["response"] or "Widget-X, 350, 5"),
                max_tokens=30, ctx_size=2048)
            step3 = run_inference(model,
                AGENT_STEP3_TEMPLATE.format(step2=step2["response"] or "Subtotal: 1750, GST: 315"),
                max_tokens=50, ctx_size=2048)

            chain_total = time.perf_counter() - chain_start
            mon2.stop(); s2 = mon2.get_stats()

            linear_expected = single["total_time_s"] * 3
            overhead_ratio = chain_total / linear_expected if linear_expected > 0 else 0

            # Check if all steps produced output
            chain_ok = all(s["response"] for s in [step1, step2, step3])

            rows.append({
                "experiment": "E3",
                "device": device,
                "model": model,
                "run": i + 1,
                "single_time_s": single["total_time_s"],
                "single_tokens": single["tokens"],
                "single_toks": single["tokens_per_sec"],
                "single_ttft_ms": single["ttft_ms"],
                "agent_chain_s": round(chain_total, 4),
                "step1_s": step1["total_time_s"],
                "step2_s": step2["total_time_s"],
                "step3_s": step3["total_time_s"],
                "linear_expected_s": round(linear_expected, 4),
                "overhead_ratio": round(overhead_ratio, 4),
                "chain_completed": chain_ok,
                "single_ram_mb": s1["peak_ram_mb"],
                "agent_ram_mb": s2["peak_ram_mb"],
            })
            print(f"  Run {i+1:02d}: single={single['total_time_s']:.2f}s | "
                  f"agent={chain_total:.2f}s | ratio={overhead_ratio:.2f}x | ok={chain_ok}")

        unload_model(model)
        time.sleep(15)

    save_csv(rows, f"{args.output_dir}/e3_overhead_{device}_{get_timestamp()}.csv")
    print_header("E3 COMPLETE")


if __name__ == "__main__":
    main()
