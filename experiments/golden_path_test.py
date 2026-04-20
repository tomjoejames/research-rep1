import sys
import os
import json
import re
import math
import time
import importlib.util

# 1. Path Normalization & WSL Readiness
# Assuming the script was moved to experiments/ folder, root is one level up.
# If it is in the root directory, adjust the path logic accordingly.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO_ROOT)

from experiments.lib import ResourceMonitor, run_inference, load_model

# 2. Skill Loading & Modular Import
# Since Python cannot normally import paths with dots (e.g., .agents) or hyphens 
# (e.g., zero-cost-router) via standard `import` syntax, we dynamically load them.
def load_skill_function(skill_folder_name, function_name):
    skill_path = os.path.join(REPO_ROOT, ".agents", "skills", skill_folder_name, "skill.py")
    
    # Check if the module exists before attempting to load (prevent unhandled crashes)
    if not os.path.exists(skill_path):
        raise FileNotFoundError(f"Missing skill implementation: {skill_path}")
        
    spec = importlib.util.spec_from_file_location(function_name, skill_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)

# Dynamically bind references to the 'actual' production skill functions
zero_cost_router = load_skill_function("zero-cost-router", "zero_cost_router")
di_scorer = load_skill_function("di-scorer-skill", "di_scorer")

def run_golden_path():
    task = "Look up prices for Keyboard (1200), Mouse (800), and Monitor (15000). Calculate subtotal and add 18% GST. Format as final invoice."
    expected_total = 20060.0
    
    print("\n--- Golden Path Test: 3-Item Invoice ---")
    
    # Use imported Router
    model = zero_cost_router(task)
    print(f"Decision: {model}")
    
    load_model(model)
    monitor = ResourceMonitor()
    monitor.start()
    
    prompt =f"""Database: {{'Keyboard': 1200, 'Mouse': 800, 'Monitor': 15000}}. 
                Query: {task}. 
                Break down the calculation step-by-step. 
                1. Sum the prices. 
                2. Multiply by 0.18 for GST. 
                3. Add them together.
                Provide the final numeric total clearly as 'Final Total: [number]'."""
    result = run_inference(model, prompt, max_tokens=128)
    
    monitor.stop()
    stats = monitor.get_stats()
    
    # 3. Standardize Accuracy verification
    response = result["response"]
    print(f"\nResponse: {response}")
    raw_matches = re.findall(r"[\d,]+(?:\.\d+)?", response)
    nums = []
    for n in raw_matches:
        clean_n = n.replace(',', '')
        # Only convert if there's actually a digit in there (prevents empty string/comma errors)
        if any(char.isdigit() for char in clean_n):
            try:
                nums.append(float(clean_n))
            except ValueError:
                continue
    # math.isclose logic mapped to robust extraction array
    acc = 100 if any(math.isclose(n, expected_total, abs_tol=0.1) for n in nums) else 0
    
    # Use imported DI Scorer
    di_score = di_scorer(acc, result["total_time_s"], stats["peak_ram_mb"], 1.0)
    
    print("\n--- PERFORMANCE SUMMARY ---")
    print("| Chosen Model | Accuracy | Peak Isolated RAM | Final DI Score |")
    print("| :--- | :--- | :--- | :--- |")
    print(f"| {model} | {acc}% | {stats['peak_ram_mb']} MB | {di_score} |")
    
    if stats['peak_ram_mb'] < 10000:
        print("\nNOTE: Isolated RAM confirmed. System total RAM is usually >15GB.")

if __name__ == "__main__":
    run_golden_path()
