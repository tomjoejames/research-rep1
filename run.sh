#!/bin/bash

echo "🚀 Starting full experiment run..."

# Start Ollama (background)
echo "Starting Ollama..."
ollama serve &
sleep 5

# Run experiments
echo "Running E1..."
python experiments/e1_baseline.py --device device1

echo "Running E2..."
python experiments/e2_quantization.py --device device1

echo "Running E3..."
python experiments/e3_agent_overhead.py --device device1

echo "Running E5..."
python experiments/e5_memory.py --device device1

echo "Running E6..."
python experiments/e6_coldwarm.py --device device1

echo "✅ Experiments completed"

# Kill Ollama 
echo "🛑 Stopping Ollama..."
kill $OLLAMA_PID

# Backup kill
pkill ollama


# Create branch name with timestamp
BRANCH_NAME="results-device1"

echo "🌿 Creating branch: $BRANCH_NAME"
git checkout -b $BRANCH_NAME

# Add results
git add .

# Commit
git commit -m "Add experiment results for device1 - $BRANCH_NAME"

# Push
git push origin $BRANCH_NAME

echo "🚀 Pushed results to branch: $BRANCH_NAME"

current_date=$(date +%F)

# Push to main as well
git switch main
git add .
git commit -am "Device1 results - $current_date"
git push origin main