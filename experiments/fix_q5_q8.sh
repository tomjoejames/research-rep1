#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Q5 and Q8 Model Fix Script
# Run this on the Ubuntu experiment devices (Device 1 & 2)
# to fix the missing Q5 and Q8 Ollama models.
# ─────────────────────────────────────────────────────────────

set -e

echo "============================================"
echo "  Fixing Q5/Q8 Quantization Models"
echo "============================================"

# Check if huggingface_hub is installed, if not try to install it or warn
if ! python3 -c "import huggingface_hub" &> /dev/null; then
  echo "Installing huggingface_hub..."
  python3 -m pip install huggingface_hub
fi

# We use the existing download_gguf.py to fetch and register the models
if [ -f "experiments/download_gguf.py" ]; then
    echo "Running Python GGUF downloader..."
    python3 experiments/download_gguf.py
else
    echo "ERROR: experiments/download_gguf.py not found. Are you in the project root?"
    exit 1
fi

echo ""
echo "============================================"
echo "  Models should now be registered!"
echo "  Please verify that 'ollama list' shows:"
echo "   - tinyllama-q5 / tinyllama-q8"
echo "   - phi3-q5 / phi3-q8"
echo "   - qwen-q5 / qwen-q8"
echo "   - mistral-q5 / mistral-q8"
echo "============================================"
echo ""
echo "Next step: Re-run E2 to collect the missing data:"
echo "  python experiments/e2_quantization.py --device device1"
