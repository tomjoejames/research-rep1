#!/usr/bin/env python3
"""
Shared experiment library.
All experiment scripts import from here to avoid code duplication.
"""

import json
import time
import os
import sys
import csv
import threading
import argparse
import psutil
import requests
from datetime import datetime

OLLAMA_URL = "http://localhost:11434"

# ── Default models ────────────────────────────────────────────
# Q4 defaults (pulled by Ollama natively)
Q4_MODELS = {
    "tinyllama": "tinyllama",
    "phi3": "phi3:mini",
    "qwen": "qwen2.5:3b",
    "mistral": "mistral:7b",
}

# All 12 model-quant combinations
ALL_MODELS = [
    "tinyllama", "tinyllama-q5", "tinyllama-q8",
    "phi3:mini",  "phi3-q5",      "phi3-q8",
    "qwen2.5:3b", "qwen-q5",      "qwen-q8",
    "mistral:7b",  "mistral-q5",   "mistral-q8",
]

# Model metadata for reference
MODEL_INFO = {
    "tinyllama":    {"params": "1.1B", "family": "TinyLlama"},
    "tinyllama-q5": {"params": "1.1B", "family": "TinyLlama"},
    "tinyllama-q8": {"params": "1.1B", "family": "TinyLlama"},
    "phi3:mini":    {"params": "3.8B", "family": "Phi-3"},
    "phi3-q5":      {"params": "3.8B", "family": "Phi-3"},
    "phi3-q8":      {"params": "3.8B", "family": "Phi-3"},
    "qwen2.5:3b":   {"params": "3.0B", "family": "Qwen2.5"},
    "qwen-q5":      {"params": "3.0B", "family": "Qwen2.5"},
    "qwen-q8":      {"params": "3.0B", "family": "Qwen2.5"},
    "mistral:7b":   {"params": "7.2B", "family": "Mistral"},
    "mistral-q5":   {"params": "7.2B", "family": "Mistral"},
    "mistral-q8":   {"params": "7.2B", "family": "Mistral"},
}


# ── Resource Monitor ──────────────────────────────────────────
class ResourceMonitor:
    """Background thread that samples RAM and CPU at ~100ms intervals."""

    def __init__(self):
        self.ram_samples = []
        self.cpu_samples = []
        self.timestamps = []
        self.running = False
        self._thread = None

    def start(self):
        self.ram_samples = []
        self.cpu_samples = []
        self.timestamps = []
        self.running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _sample(self):
        while self.running:
            self.timestamps.append(time.perf_counter())
            self.ram_samples.append(psutil.virtual_memory().used / (1024**2))
            self.cpu_samples.append(psutil.cpu_percent(interval=0.1))

    def get_stats(self):
        return {
            "peak_ram_mb": round(max(self.ram_samples), 1) if self.ram_samples else 0,
            "avg_ram_mb": round(sum(self.ram_samples) / len(self.ram_samples), 1) if self.ram_samples else 0,
            "peak_cpu_pct": round(max(self.cpu_samples), 1) if self.cpu_samples else 0,
            "avg_cpu_pct": round(sum(self.cpu_samples) / len(self.cpu_samples), 1) if self.cpu_samples else 0,
            "samples_count": len(self.ram_samples),
        }

    def get_timeseries(self):
        """Return full time-series data for detailed analysis (used by E5)."""
        if not self.timestamps:
            return []
        t0 = self.timestamps[0]
        return [
            {"t_s": round(t - t0, 3), "ram_mb": round(r, 1), "cpu_pct": round(c, 1)}
            for t, r, c in zip(self.timestamps, self.ram_samples, self.cpu_samples)
        ]


# ── Ollama Interaction ────────────────────────────────────────
def check_ollama():
    """Check if Ollama is running."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False


def load_model(model_name):
    """Cold-start load a model into memory."""
    print(f"  Loading {model_name}...")
    start = time.perf_counter()
    try:
        requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model_name, "prompt": "Hi", "options": {"num_predict": 1}
        }, timeout=600)
    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Cannot connect to Ollama at {OLLAMA_URL}")
        print(f"  Start it with: ollama serve &")
        return -1
    except requests.exceptions.Timeout:
        print(f"  ERROR: Model load timed out (>10min)")
        return -1
    load_time = time.perf_counter() - start
    print(f"  Loaded in {load_time:.2f}s")
    return load_time


def unload_model(model_name):
    """Unload model from memory."""
    try:
        requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model_name, "keep_alive": 0, "prompt": ""
        }, timeout=30)
    except:
        pass
    time.sleep(5)


def run_inference(model_name, prompt, max_tokens=256, ctx_size=4096):
    """Run a single streaming inference. Returns dict with timing + output."""
    start_time = time.perf_counter()
    first_token_time = None
    full_response = ""
    token_count = 0

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.0,
                    "num_predict": max_tokens,
                    "num_ctx": ctx_size,
                }
            },
            stream=True,
            timeout=600,
        )
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "response" in data and data["response"]:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_response += data["response"]
                token_count += 1
            if data.get("done", False):
                break
    except Exception as e:
        return {
            "response": "",
            "total_time_s": round(time.perf_counter() - start_time, 4),
            "ttft_ms": 0,
            "tokens": 0,
            "tokens_per_sec": 0,
            "error": str(e),
        }

    end_time = time.perf_counter()
    total_time = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else total_time

    return {
        "response": full_response.strip(),
        "total_time_s": round(total_time, 4),
        "ttft_ms": round(ttft * 1000, 2),
        "tokens": token_count,
        "tokens_per_sec": round(token_count / total_time, 2) if total_time > 0 else 0,
    }


def warmup(model_name, n=5):
    """Run n warm-up inferences (results discarded)."""
    print(f"  Warm-up ({n} runs)...")
    for _ in range(n):
        run_inference(model_name, "What is 2+2?", max_tokens=10)


# ── Output Helpers ────────────────────────────────────────────
def save_csv(rows, filepath):
    """Save list of dicts to CSV."""
    if not rows:
        print("  WARNING: No data to save.")
        return
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  ✅ Saved {len(rows)} rows → {filepath}")


def save_json(data, filepath):
    """Save data to JSON."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✅ Saved → {filepath}")


def make_parser(description):
    """Create standard CLI argument parser for experiment scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--device", type=str, default="device1",
                        help="Device identifier (device1 or device2)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Override model list (space-separated)")
    parser.add_argument("--runs", type=int, default=None,
                        help="Override number of runs per model")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    return parser


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")
