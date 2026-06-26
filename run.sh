#!/bin/bash

set -e

# ─────────────────────────────────────────────
# Register Device
# ─────────────────────────────────────────────

if [ ! -f ".device_id" ]; then

    echo "====================================="
    echo " First-time device registration"
    echo "====================================="
    echo ""

    echo "Enter your invite code:"
    read INVITE

    DEVICE_ID=$(python3 - <<EOF
import json

with open("registry/devices.json") as f:
    data = json.load(f)

print(data.get("$INVITE", "INVALID"))
EOF
)

    if [ "$DEVICE_ID" = "INVALID" ]; then
        echo ""
        echo "❌ Invalid invite code"
        exit 1
    fi

    echo "$DEVICE_ID" > .device_id

    echo ""
    echo "✅ Registered as: $DEVICE_ID"

else
    DEVICE_ID=$(cat .device_id)
fi


# ─────────────────────────────────────────────
# Start
# ─────────────────────────────────────────────

echo ""
echo "====================================="
echo " Running Experiments on $DEVICE_ID"
echo "====================================="
echo ""

CURRENT_DATE=$(date +%F)
BRANCH_NAME="results-$DEVICE_ID-$CURRENT_DATE"

echo "Branch name:"
echo "$BRANCH_NAME"
echo ""


# ─────────────────────────────────────────────
# Install Git if missing (Linux/WSL)
# ─────────────────────────────────────────────

if ! command -v git &> /dev/null; then
    echo "Installing Git..."
    sudo apt-get update -qq && sudo apt-get install -y git
fi


# ─────────────────────────────────────────────
# Install Ollama if missing
# ─────────────────────────────────────────────

if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "Ollama already installed: $(ollama --version)"
fi


# ─────────────────────────────────────────────
# Guard: Q4 models must already be pulled
# ─────────────────────────────────────────────

# Run setup_models.sh once before run.sh if this check fails.
echo ""
echo "Checking Q4 models..."
for model in tinyllama phi3:mini qwen2.5:3b mistral:7b; do
    if ! ollama list | grep -q "$model"; then
        echo ""
        echo "❌ Model '$model' not found. Run setup first:"
        echo "   chmod +x setup_models.sh && ./setup_models.sh"
        exit 1
    fi
done
echo "✅ All Q4 models present."


# ─────────────────────────────────────────────
# Python venv + dependencies
# ─────────────────────────────────────────────

# || true suppresses error if .venv already exists
echo ""
echo "Installing virtual environment... skipping if it already exists."
python3 -m venv .venv 2>/dev/null || true
. .venv/bin/activate

echo ""
echo "Inside virtual environment '.venv'..."

echo ""
# Check if requirements are met, hide the error output, and run pip only if it fails
if ! pip check -r requirements.txt > /dev/null 2>&1; then
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
else
    echo "Dependencies already satisfied. Skipping."
fi

# ─────────────────────────────────────────────
# Guard: All models must be present before experiments start
# ─────────────────────────────────────────────

# Q4 models: run setup_models.sh if any are missing.
# Q5/Q8 models: run fix_q5_q8.sh if any are missing.
echo ""
echo "Checking Q4 models..."
for model in tinyllama phi3:mini qwen2.5:3b mistral:7b; do
    if ! ollama list | grep -q "$model"; then
        echo ""
        echo "❌ Q4 model '$model' not found. Run setup first:"
        echo "   chmod +x setup_models.sh && ./setup_models.sh"
        exit 1
    fi
done
echo "✅ All Q4 models present."

echo ""
echo "Checking Q5/Q8 models..."
for model in tinyllama-q5 tinyllama-q8 phi3-q5 phi3-q8 qwen-q5 qwen-q8 mistral-q5 mistral-q8; do
    if ! ollama list | grep -q "$model"; then
        echo ""
        echo "❌ Q5/Q8 model '$model' not found. Run the quantization fix first:"
        echo "   chmod +x fix_q5_q8.sh && ./fix_q5_q8.sh"
        exit 1
    fi
done
echo "✅ All Q5/Q8 models present."


# ─────────────────────────────────────────────
# Start Ollama
# ─────────────────────────────────────────────

echo ""
echo "Starting Ollama..."
ollama serve &
OLLAMA_PID=$!
sleep 5


# ─────────────────────────────────────────────
# Run Experiments
# ─────────────────────────────────────────────

echo ""
echo "Running E1..."
python experiments/e1_baseline.py --device $DEVICE_ID

echo ""
echo "Running E2..."
python experiments/e2_quantization.py --device $DEVICE_ID

echo ""
echo "Running E3..."
python experiments/e3_agent_overhead.py --device $DEVICE_ID

# E4 (cross-device resilience) is excluded here — must be run manually by the
# finalist experimenter after both devices have completed E1–E6.

echo ""
echo "Running E5..."
python experiments/e5_memory.py --device $DEVICE_ID

echo ""
echo "Running E6..."
python experiments/e6_coldwarm.py --device $DEVICE_ID


# ─────────────────────────────────────────────
# Stop Ollama
# ─────────────────────────────────────────────

echo ""
echo "Stopping Ollama..."
kill $OLLAMA_PID || true
pkill ollama || true


# ─────────────────────────────────────────────
# Git Push
# ─────────────────────────────────────────────

echo ""
echo "Setting origin..."
git fetch origin
git checkout main
git pull origin main

echo ""
echo "Preparing Git branch..."
git checkout -b $BRANCH_NAME
git add results/
git commit -m "Experiment results from $DEVICE_ID"
git push origin $BRANCH_NAME


# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────

echo ""
echo "====================================="
echo "✅ Results pushed successfully"
echo "====================================="
echo "Branch:"
echo "$BRANCH_NAME"
echo ""
