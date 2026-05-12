#!/usr/bin/env python3
"""
E5 — Memory Pressure Analysis & Active KV Cache Monitoring
Tests: Tracks dynamic memory growth as Context Window scales up, 
       automatically intervening when pressure triggers the local logic.
Device: Device 1 & 2
"""

import sys, os, time, psutil
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lib import *

# ── Dynamic Skill Loader ─────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_skill_function(skill_folder_name, function_name):
    skill_path = os.path.join(REPO_ROOT, ".agents", "skills", skill_folder_name, "skill.py")
    if not os.path.exists(skill_path):
        raise FileNotFoundError(f"Missing skill implementation: {skill_path}")
    spec = importlib.util.spec_from_file_location(function_name, skill_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)

monitor_kv_pressure = load_skill_function("kv-cache-pressure-monitor", "monitor_kv_pressure")

# Defaulting to 4 standard test models to prevent exhausting time
DEFAULT_MODELS = [
    "tinyllama", "phi3:mini", "qwen2.5:3b", "mistral:7b"
]

MAX_TURNS = 10
# 300 words dummy block to force KV token inflation rapidly
DUMMY_BLOCK = " ".join(["memory_test_token"] * 300)

def get_swap_mb():
    """Get current swap usage in MB."""
    swap = psutil.swap_memory()
    return round(swap.used / (1024**2), 1)

def main():
    parser = make_parser("E5: KV Cache Pressure Monitoring")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    device = args.device

    if not check_ollama():
        print("ERROR: Ollama not running."); sys.exit(1)

    print_header(f"E5: KV CACHE PRESSURE — {device}")
    total_ram_mb = round(psutil.virtual_memory().total / (1024**2), 0)
    print(f"System RAM: {total_ram_mb} MB")

    summary_rows = []
    all_timeseries = {}

    for model in models:
        print_header(f"Model: {model}")
        
        baseline_ram = psutil.virtual_memory().used / (1024**2)
        baseline_swap = get_swap_mb()

        # ── Phase 2: Load ─────────────────────────────────
        mon_load = ResourceMonitor(); mon_load.start()
        load_time = load_model(model)
        mon_load.stop()

        if load_time < 0:
            print(f"  SKIPPING {model}")
            continue

        load_stats = mon_load.get_stats()
        post_load_ram = psutil.virtual_memory().used / (1024**2)

        # ── Phase 3: Active KV Scaling ─────────────────────
        print(f"  Phase 3: Active KV Scaling Context Growth...")
        mon_infer = ResourceMonitor(); mon_infer.start()
        
        current_prompt = "Explain quantum computing."
        
        tokens_at_flush = 0
        cache_flushed = False
        final_pressure_status = "SAFE"
        max_tokens_reached = 0

        for turn in range(MAX_TURNS):
            # 1. Accumulate Context
            current_prompt += f"\n\nAdditional Context:\n{DUMMY_BLOCK}"
            
            # 2. Active Monitoring Estimation
            word_count = len(current_prompt.split())
            current_tokens = int(word_count * 1.3)
            current_ram_mb = psutil.virtual_memory().used / (1024**2)
            
            max_tokens_reached = max(max_tokens_reached, current_tokens)
            
            # Call monitoring skill
            pressure_data = monitor_kv_pressure(
                current_tokens=current_tokens, 
                max_context_window=2048, 
                current_ram_mb=current_ram_mb, 
                system_max_ram_mb=total_ram_mb
            )
            final_pressure_status = pressure_data.get('status', 'SAFE')
            
            if final_pressure_status == 'CRITICAL':
                print("⚠️ CRITICAL PRESSURE REACHED: FLUSHING CACHE")
                tokens_at_flush = current_tokens
                cache_flushed = True
                break
                
            # 3. Inference Run
            result = run_inference(model, current_prompt, max_tokens=100, ctx_size=2048)
            print(f"    Turn {turn+1}: {current_tokens} estimated tokens | RAM: {current_ram_mb:.0f}MB | Latency: {result['total_time_s']:.2f}s")

        mon_infer.stop()
        infer_stats = mon_infer.get_stats()
        peak_infer_swap = get_swap_mb()

        # ── Phase 4: Unload ────────────────────────────────
        print("  Phase 4: Unloading & recovery...")
        unload_model(model)
        time.sleep(5)

        swap_used = peak_infer_swap > baseline_swap + 10

        # Update CSV Schema
        summary_rows.append({
            "experiment": "E5",
            "device": device,
            "model": model,
            "baseline_ram_mb": round(baseline_ram, 1),
            "peak_inference_ram_mb": infer_stats["peak_ram_mb"],
            "max_tokens_reached": max_tokens_reached,
            "pressure_status": final_pressure_status,
            "cache_flushed": cache_flushed,
            "peak_swap_mb": peak_infer_swap,
            "swap_active": swap_used,
            "ram_utilization_pct": round(infer_stats["peak_ram_mb"] / total_ram_mb * 100, 1),
        })

        all_timeseries[model] = {
            "load_phase": mon_load.get_timeseries(),
            "inference_phase": mon_infer.get_timeseries(),
        }

        print(f"  → Max Tokens: {max_tokens_reached} | Status: {final_pressure_status} | Flushed: {cache_flushed}")
        time.sleep(10)

    ts = get_timestamp()
    save_csv(summary_rows, f"{args.output_dir}/e5_memory_{device}_{ts}.csv")
    save_json(all_timeseries, f"{args.output_dir}/e5_timeseries_{device}_{ts}.json")
    print_header("E5 COMPLETE")

if __name__ == "__main__":
    main()
