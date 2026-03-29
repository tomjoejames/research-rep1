#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Setup script for LLM Deployability Experiments
# Run this on EACH device before starting experiments.
#
# Usage:
#   chmod +x experiments/setup_models.sh
#   ./experiments/setup_models.sh
# ─────────────────────────────────────────────────────────────

set -e

echo "============================================"
echo "  Experiment Setup for LLM Deployability"
echo "============================================"

# ── Step 1: Check Ollama ──────────────────────────────────────
echo ""
echo "[1/4] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "  Ollama not found. Installing..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "  Ollama already installed: $(ollama --version)"
fi

# ── Step 2: Check Python deps ────────────────────────────────
echo ""
echo "[2/4] Checking Python dependencies..."
# pip install --quiet psutil requests pandas matplotlib seaborn numpy huggingface_hub
python3 -m pip install --quiet psutil requests pandas matplotlib seaborn numpy huggingface_hub
echo "  Dependencies OK"

# ── Step 3: Pull Q4 models (Ollama native) ───────────────────
echo ""
echo "[3/4] Pulling Q4_K_M models (Ollama native)..."
echo "  This may take a while on first run."
echo ""

models=("tinyllama" "phi3:mini" "qwen2.5:3b" "mistral:7b")
for model in "${models[@]}"; do
    echo "  Pulling $model..."
    ollama pull "$model"
done

echo ""
echo "  All Q4 models pulled."

# ── Step 4: Instructions for Q5/Q8 variants ─────────────────
echo ""
echo "[4/4] Q5 and Q8 quantization variants"
echo ""
echo "  The Q4_K_M variants are ready. For E2 (quantization experiment),"
echo "  you need Q5_K_M and Q8_0 variants. Download them manually:"
echo ""
echo "  Option A: Use the Python download helper:"
echo "    python experiments/download_gguf.py"
echo ""
echo "  Option B: Manual download from HuggingFace and create Modelfiles:"
echo "    1. Download .gguf files from bartowski/<model>-GGUF repos"
echo '    2. echo "FROM ./path/to/model.gguf" > Modelfile'
echo '    3. ollama create <name> -f Modelfile'
echo ""

# ── Verify ───────────────────────────────────────────────────
echo "============================================"
echo "  Setup Complete! Available models:"
echo "============================================"
ollama list
echo ""
echo "  To run experiments:"
echo "    python experiments/e1_baseline.py --device device1"
echo "    python experiments/e2_quantization.py --device device1"
echo "    python experiments/e3_agent_overhead.py --device device1"
echo "    python experiments/e4_cross_device.py --device device1"
echo "    python experiments/e5_memory.py --device device1"
echo "    python experiments/e6_coldwarm.py --device device1"
echo ""
echo "  Change --device to device2 on the second machine."
