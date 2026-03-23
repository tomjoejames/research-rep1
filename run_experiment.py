#!/usr/bin/env python3
"""
Research Experiment: LLM Deployability on Resource-Constrained Systems
Run: python run_experiment.py
"""

import json
import time
import os
import threading
import psutil
import requests
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
RESULTS_DIR = "results"
DATA_DIR = "data"

MODELS = [
    "tinyllama-q4", "tinyllama-q5", "tinyllama-q8",
    "phi3-q4", "phi3-q5", "phi3-q8",
    "qwen-q4", "qwen-q5", "qwen-q8",
    "mistral-q4", "mistral-q5", "mistral-q8",
]


# ── Resource Monitor ───────────────────────────────────────────
class ResourceMonitor:
    """Background thread that samples RAM and CPU every 100ms."""
    def __init__(self):
        self.ram_samples = []
        self.cpu_samples = []
        self.running = False

    def start(self):
        self.ram_samples = []
        self.cpu_samples = []
        self.running = True
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

    def _sample(self):
        while self.running:
            self.ram_samples.append(psutil.virtual_memory().used / (1024**2))
            self.cpu_samples.append(psutil.cpu_percent(interval=0.1))

    def get_stats(self):
        return {
            "peak_ram_mb": round(max(self.ram_samples), 1) if self.ram_samples else 0,
            "avg_ram_mb": round(sum(self.ram_samples)/len(self.ram_samples), 1) if self.ram_samples else 0,
            "peak_cpu_pct": round(max(self.cpu_samples), 1) if self.cpu_samples else 0,
            "avg_cpu_pct": round(sum(self.cpu_samples)/len(self.cpu_samples), 1) if self.cpu_samples else 0,
        }


# ── Ollama Interaction ─────────────────────────────────────────
def load_model(model_name):
    """Load a model into memory (cold start)."""
    print(f"  Loading model {model_name}...")
    start = time.perf_counter()
    try:
        requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model_name,
            "prompt": "Hi",
            "options": {"num_predict": 1}
        }, timeout=300)
    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Cannot connect to Ollama. Is it running? (ollama serve)")
        return -1
    load_time = time.perf_counter() - start
    print(f"  Loaded in {load_time:.2f}s")
    return load_time


def unload_model(model_name):
    """Unload model from memory."""
    try:
        requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model_name,
            "keep_alive": 0,
            "prompt": ""
        }, timeout=30)
    except:
        pass
    time.sleep(5)


def run_inference(model_name, prompt, max_tokens=256):
    """Run a single inference and return timing + output."""
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
                    "num_ctx": 4096,
                }
            },
            stream=True,
            timeout=300
        )

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data and data["response"]:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    full_response += data["response"]
                    token_count += 1
                if data.get("done", False):
                    break
    except Exception as e:
        print(f"    ERROR during inference: {e}")
        return {
            "response": "",
            "total_time_s": time.perf_counter() - start_time,
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


# ── Task Prompts ───────────────────────────────────────────────
def classification_prompt(email_text):
    return f"""You are an email classifier. Classify the following email into exactly one category.
Categories: inquiry, complaint, feedback, spam, urgent

Respond with ONLY the category label, nothing else.

Email:
{email_text}

Category:"""


def extraction_prompt(text):
    return f"""Extract the following fields from the text below. Return ONLY a valid JSON object with these keys: name, date, location, price, category.
If a field is not found, use "N/A".

Text:
{text}

JSON:"""


def agent_step1_prompt(query, lookup_data):
    data_str = json.dumps(lookup_data, indent=2)
    return f"""You are a helpful assistant with access to a product database.

Database:
{data_str}

User query: {query}

Step 1: Look up the relevant product information from the database.
Respond with ONLY the relevant product details in this format:
Product: [name]
Price: [price]
Unit: [unit]"""


def agent_step2_prompt(query, step1_result):
    return f"""You are a calculator assistant.

User query: {query}
Product information: {step1_result}

Step 2: Perform the required calculations (quantity × price, tax, total).
Respond with ONLY the calculation results:
Subtotal: [amount]
Tax: [amount]
Total: [amount]"""


def agent_step3_prompt(query, step2_result):
    return f"""You are a document formatter.

User query: {query}
Calculation results: {step2_result}

Step 3: Format the above as a clean invoice.
Respond with a formatted invoice only."""


# ── Experiment Runners ─────────────────────────────────────────
def run_classification_experiment(model_name, data):
    results = []
    for i, sample in enumerate(data):
        monitor = ResourceMonitor()
        monitor.start()
        result = run_inference(model_name, classification_prompt(sample["text"]), max_tokens=10)
        monitor.stop()

        predicted = result["response"].strip().lower().replace(".", "").replace(",", "")
        correct = sample["label"].lower() in predicted

        results.append({
            "task": "classification",
            "sample_id": sample["id"],
            "model": model_name,
            "correct": correct,
            "predicted": predicted,
            "expected": sample["label"],
            **result,
            **monitor.get_stats(),
        })
        print(f"    [{i+1}/{len(data)}] {'✓' if correct else '✗'} | {result['tokens_per_sec']} tok/s | {result['ttft_ms']}ms TTFT")
    return results


def run_extraction_experiment(model_name, data):
    results = []
    for i, sample in enumerate(data):
        monitor = ResourceMonitor()
        monitor.start()
        result = run_inference(model_name, extraction_prompt(sample["text"]), max_tokens=256)
        monitor.stop()

        fields_correct = 0
        total_fields = 5
        try:
            resp = result["response"]
            json_start = resp.find("{")
            json_end = resp.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(resp[json_start:json_end])
                expected = sample["expected"]
                for key in ["name", "date", "location", "price", "category"]:
                    if key in parsed and str(parsed[key]).strip().lower() == str(expected.get(key, "")).strip().lower():
                        fields_correct += 1
        except (json.JSONDecodeError, KeyError):
            pass

        results.append({
            "task": "extraction",
            "sample_id": sample["id"],
            "model": model_name,
            "fields_correct": fields_correct,
            "total_fields": total_fields,
            "extraction_accuracy": round(fields_correct / total_fields, 2),
            **result,
            **monitor.get_stats(),
        })
        print(f"    [{i+1}/{len(data)}] {fields_correct}/{total_fields} fields | {result['tokens_per_sec']} tok/s")
    return results


def run_agent_experiment(model_name, data):
    results = []
    for i, sample in enumerate(data):
        monitor = ResourceMonitor()
        monitor.start()
        chain_start = time.perf_counter()

        # Step 1: Lookup
        step1 = run_inference(model_name, agent_step1_prompt(
            sample["query"], sample["lookup_data"]
        ), max_tokens=128)

        # Step 2: Calculate
        step2 = run_inference(model_name, agent_step2_prompt(
            sample["query"], step1["response"]
        ), max_tokens=128)

        # Step 3: Format
        step3 = None
        if sample.get("steps", 2) >= 3:
            step3 = run_inference(model_name, agent_step3_prompt(
                sample["query"], step2["response"]
            ), max_tokens=512)

        chain_end = time.perf_counter()
        monitor.stop()

        total_chain_time = chain_end - chain_start
        steps_completed = 3 if step3 else 2
        all_steps_ok = bool(step1["response"] and step2["response"])
        if step3:
            all_steps_ok = all_steps_ok and bool(step3["response"])

        final_response = (step3 or step2)["response"]
        expected_total = str(sample["expected_answer"]["total"])
        answer_correct = expected_total in final_response

        results.append({
            "task": "agent",
            "sample_id": sample["id"],
            "model": model_name,
            "steps_completed": steps_completed,
            "steps_required": sample.get("steps", 3),
            "chain_completed": all_steps_ok,
            "answer_correct": answer_correct,
            "chain_time_s": round(total_chain_time, 4),
            "step1_time_s": step1["total_time_s"],
            "step2_time_s": step2["total_time_s"],
            "step3_time_s": step3["total_time_s"] if step3 else 0,
            "step1_toks": step1["tokens_per_sec"],
            "step2_toks": step2["tokens_per_sec"],
            **monitor.get_stats(),
        })
        print(f"    [{i+1}/{len(data)}] {'✓' if answer_correct else '✗'} | chain: {total_chain_time:.1f}s | completed: {all_steps_ok}")
    return results


# ── Main ───────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    with open(f"{DATA_DIR}/emails.json") as f:
        emails = json.load(f)
    with open(f"{DATA_DIR}/extraction.json") as f:
        extraction = json.load(f)
    with open(f"{DATA_DIR}/agent_tasks.json") as f:
        agent_tasks = json.load(f)

    print(f"  Emails: {len(emails)}, Extraction: {len(extraction)}, Agent: {len(agent_tasks)}")

    all_results = []

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")

        load_time = load_model(model)
        if load_time < 0:
            print(f"  SKIPPING {model} (failed to load)")
            continue

        # Warm-up
        print("  Warm-up (5 runs)...")
        for _ in range(5):
            run_inference(model, "What is 2+2?", max_tokens=10)

        # Task 1
        print(f"\n  [Task 1] Email Classification ({len(emails)} samples)")
        cls_results = run_classification_experiment(model, emails)
        for r in cls_results:
            r["load_time_s"] = load_time
        all_results.extend(cls_results)

        # Task 2
        print(f"\n  [Task 2] Information Extraction ({len(extraction)} samples)")
        ext_results = run_extraction_experiment(model, extraction)
        for r in ext_results:
            r["load_time_s"] = load_time
        all_results.extend(ext_results)

        # Task 3
        print(f"\n  [Task 3] Multi-Step Agent ({len(agent_tasks)} samples)")
        agent_results = run_agent_experiment(model, agent_tasks)
        for r in agent_results:
            r["load_time_s"] = load_time
        all_results.extend(agent_results)

        # Unload
        unload_model(model)
        print(f"\n  Model unloaded. Waiting 30s...")
        time.sleep(30)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{RESULTS_DIR}/experiment_run_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"✅ ALL DONE — {len(all_results)} data points saved to {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
