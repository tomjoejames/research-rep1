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
# Install Git if missing
# ─────────────────────────────────────────────

if ! command -v git &> /dev/null; then
    echo "Installing Git..."
    winget install --id Git.Git -e --source winget
fi


# ─────────────────────────────────────────────
# Start Ollama
# ─────────────────────────────────────────────

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
