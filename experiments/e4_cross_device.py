#!/usr/bin/env python3
"""
E4 — Cross-Device Reproducibility & Agent Resilience
Tests: Does the system accurately self-heal from formatting/math failures?
Device: BOTH (must run identically on each device)

Runs 2 models × 30 runs, injecting a strict formulation prompt to test resilience.
"""

import sys, os, json, time, math
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lib import *

# ── Dynamic Skill Loader ─────────────────────────────────────────
# Cross-platform absolute path resolution, explicitly avoiding relative imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_skill_function(skill_folder_name, function_name):
    skill_path = os.path.join(REPO_ROOT, ".agents", "skills", skill_folder_name, "skill.py")
    if not os.path.exists(skill_path):
        raise FileNotFoundError(f"Missing skill implementation: {skill_path}")
    spec = importlib.util.spec_from_file_location(function_name, skill_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)

analyze_failure = load_skill_function("deep-failure-analyzer", "analyze_failure")
evaluate_recovery = load_skill_function("recovery-accuracy-evaluator", "evaluate_recovery")


DEFAULT_MODELS = ["phi3:mini", "mistral:7b"]
DEFAULT_RUNS = 30

# ── Strict JSON Prompt ─────────────────────────────────────────
PROMPT = "A customer buys 7 units of Widget-Z at $112.50 each. Apply a 6.25% regional tax, then deduct a flat $25 loyalty coupon. Output ONLY valid JSON with keys: 'subtotal', 'tax_amount', 'coupon', and 'final_total'."
EXPECTED_OUTPUT = '{"subtotal": 787.5, "tax_amount": 49.22, "coupon": 25, "final_total": 811.72}'


def check_success(actual_output):
    """Helper to detect strict JSON compliance (fallback for SAFE status)."""
    try:
        clean_output = str(actual_output).strip()
        # Clean markdown code blocks if present
        if clean_output.startswith("```json"):
             clean_output = clean_output.replace("```json", "").replace("```", "").strip()
        elif clean_output.startswith("```"):
             clean_output = clean_output.replace("```", "").strip()
             
        data = json.loads(clean_output)
        sub = float(data.get("subtotal", 0))
        tax = float(data.get("tax_amount", 0))
        coup = float(data.get("coupon", 0))
        fin = float(data.get("final_total", 0))
        
        if (math.isclose(sub, 787.5, abs_tol=0.1) and 
            math.isclose(tax, 49.22, abs_tol=0.1) and 
            math.isclose(coup, 25.0, abs_tol=0.1) and 
            math.isclose(fin, 811.72, abs_tol=0.1)):
            return True
    except Exception:
        pass
    return False


def main():
    parser = make_parser("E4: Agent Resilience & Self-Healing")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    runs = args.runs or DEFAULT_RUNS
    device = args.device

    if not check_ollama():
        print("ERROR: Ollama not running."); sys.exit(1)

    print_header(f"E4: SELF-HEALING — {device}")
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
            
            # ── 1. Initial Inference ──
            result = run_inference(model, PROMPT, max_tokens=150, ctx_size=2048)
            actual_output = result["response"]
            
            # ── 2. Evaluation ──
            if check_success(actual_output):
                initial_error_type = "Success"
            else:
                initial_error_type = analyze_failure(PROMPT, EXPECTED_OUTPUT, actual_output)

            recovery_attempted = False
            recovery_successful = False

            # ── 3. Self-Healing Loop ──
            if initial_error_type not in ("Success", "SAFE"):
                recovery_attempted = True
                reprompt = f"You made a {initial_error_type}. Fix it and output only the correct JSON."
                
                # Trigger second inference
                recovery_result = run_inference(model, reprompt, max_tokens=150, ctx_size=2048)
                reprompted_output = recovery_result["response"]
                
                # Check recovery success
                eval_data = evaluate_recovery(initial_error_type, reprompted_output, EXPECTED_OUTPUT)
                recovery_successful = eval_data.get('recovered', False)
                
                # Add latency and tokens from recovery step to total resource cost
                result["total_time_s"] += recovery_result["total_time_s"]
                result["tokens"] += recovery_result["tokens"]
                result["tokens_per_sec"] = result["tokens"] / result["total_time_s"] if result["total_time_s"] > 0 else 0

            mon.stop()
            stats = mon.get_stats()

            # ── 4. CSV Schema Sync ──
            rows.append({
                "experiment": "E4",
                "device": device,
                "model": model,
                "run": i + 1,
                "total_time_s": round(result["total_time_s"], 4),
                "ttft_ms": result["ttft_ms"],
                "tokens": result["tokens"],
                "tokens_per_sec": round(result["tokens_per_sec"], 2),
                "peak_ram_mb": stats["peak_ram_mb"],
                "avg_cpu_pct": stats["avg_cpu_pct"],
                "load_time_s": round(load_time, 2),
                "initial_error_type": initial_error_type,
                "recovery_attempted": recovery_attempted,
                "recovery_successful": recovery_successful,
                "actual_output": actual_output[:500].replace("\n", " "),
            })
            
            print(f"  Run {i+1:02d}: error={initial_error_type} | recovered={recovery_successful} | time={result['total_time_s']:.2f}s")

        unload_model(model)
        time.sleep(15)

    save_csv(rows, f"{args.output_dir}/e4_resilience_{device}_{get_timestamp()}.csv")
    print_header("E4 COMPLETE")


if __name__ == "__main__":
    main()
