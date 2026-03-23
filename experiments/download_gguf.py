#!/usr/bin/env python3
"""
Helper to download Q5_K_M and Q8_0 GGUF variants from HuggingFace
and register them with Ollama.

Usage:
  python experiments/download_gguf.py
"""

import os
import subprocess
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Models to download ────────────────────────────────────────
# Format: (repo_id, filename, ollama_name)
DOWNLOADS = [
    # TinyLlama
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf", "tinyllama-q5"),
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",   "tinyllama-q8"),
    # Phi-3-mini
    ("bartowski/Phi-3-mini-4k-instruct-GGUF", "Phi-3-mini-4k-instruct-Q5_K_M.gguf", "phi3-q5"),
    ("bartowski/Phi-3-mini-4k-instruct-GGUF", "Phi-3-mini-4k-instruct-Q8_0.gguf",   "phi3-q8"),
    # Qwen2.5-3B
    ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q5_k_m.gguf", "qwen-q5"),
    ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q8_0.gguf",   "qwen-q8"),
    # Mistral-7B
    ("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "mistral-7b-instruct-v0.2.Q5_K_M.gguf", "mistral-q5"),
    ("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "mistral-7b-instruct-v0.2.Q8_0.gguf",   "mistral-q8"),
]


def main():
    print("=" * 60)
    print("  GGUF Model Downloader")
    print("=" * 60)
    print(f"  Download directory: {os.path.abspath(MODEL_DIR)}")
    print(f"  Models to download: {len(DOWNLOADS)}")
    print()

    for repo_id, filename, ollama_name in DOWNLOADS:
        print(f"\n--- {ollama_name} ---")
        filepath = os.path.join(MODEL_DIR, filename)

        # Download if not cached
        if os.path.exists(filepath):
            print(f"  Already exists: {filepath}")
        else:
            print(f"  Downloading from {repo_id}...")
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=MODEL_DIR,
                )
                filepath = downloaded
                print(f"  Downloaded: {filepath}")
            except Exception as e:
                print(f"  ERROR: {e}")
                print(f"  Try manually downloading from https://huggingface.co/{repo_id}")
                continue

        # Register with Ollama
        print(f"  Registering as '{ollama_name}' in Ollama...")
        modelfile = f"FROM {os.path.abspath(filepath)}"
        modelfile_path = os.path.join(MODEL_DIR, f"Modelfile_{ollama_name}")
        with open(modelfile_path, "w") as f:
            f.write(modelfile)

        try:
            subprocess.run(
                ["ollama", "create", ollama_name, "-f", modelfile_path],
                check=True, capture_output=True, text=True
            )
            print(f"  ✅ Registered: {ollama_name}")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR creating ollama model: {e.stderr}")

    print("\n" + "=" * 60)
    print("  Done! Verify with: ollama list")
    print("=" * 60)


if __name__ == "__main__":
    main()
