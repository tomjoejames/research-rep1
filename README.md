# Deployability of LLM-Based Agent Workflows on Resource-Constrained CPU-Only Systems

**An Empirical Evaluation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Authors

Sahana, Fazil, Adwaith

## Abstract

This research evaluates the feasibility of deploying LLM-based agent workflows on resource-constrained, CPU-only consumer hardware. We benchmark four small language models—TinyLlama-1.1B, Phi-3-mini-3.8B, Qwen2.5-3B-Instruct, and Mistral-7B-Instruct-v0.3—across three GGUF quantization levels (Q4_K_M, Q5_K_M, Q8_0) on an Intel Core i5-1235U system with 16GB RAM and no discrete GPU.

## Repository Structure

```
.
├── README.md                          # This file
├── paper_v2.md                        # Full research paper
├── draft_v1.md                        # Original draft (archived)
├── literature_review_references.md    # Collected references (19 papers)
├── run_experiment.py                  # Main experiment benchmark script
├── analyze_results.py                 # Results analysis & figure generation
├── data/
│   ├── emails.json                    # Email classification dataset (100 samples)
│   ├── extraction.json                # Information extraction dataset (50 samples)
│   └── agent_tasks.json               # Multi-step agent tasks (50 samples)
└── results/                           # Experiment output (auto-generated)
    └── figures/                       # Generated plots (auto-generated)
```

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| CPU | Intel Core i5-1235U (12th Gen) or equivalent |
| RAM | 16 GB minimum |
| Storage | ~20 GB free (for model files) |
| GPU | Not required (CPU-only inference) |
| OS | Linux / macOS / Windows with WSL |

## Quick Start

### 1. Install Dependencies

```bash
# Install Ollama (inference runtime)
curl -fsSL https://ollama.ai/install.sh | sh

# Install Python packages
pip install psutil requests pandas matplotlib seaborn
```

### 2. Download Models

```bash
ollama pull tinyllama
ollama pull phi3:mini
ollama pull qwen2.5:3b
ollama pull mistral:7b
```

### 3. Run Experiments

```bash
# Ensure Ollama is running
ollama serve &

# Run the full benchmark suite
python run_experiment.py
```

### 4. Analyze Results

```bash
python analyze_results.py results/experiment_run_*.json
```

This generates:
- **5 CSV tables** (throughput, resources, accuracy, chain depth, deployability index)
- **4 figures** (throughput bar chart, latency box plot, RAM usage, DI heatmap)

## Experiment Overview

### Models Evaluated

| Model | Parameters | Size (Q4_K_M) |
|-------|-----------|---------------|
| TinyLlama-1.1B | 1.1B | ~700 MB |
| Phi-3-mini-3.8B | 3.8B | ~2.3 GB |
| Qwen2.5-3B-Instruct | 3.0B | ~1.9 GB |
| Mistral-7B-Instruct-v0.3 | 7.2B | ~4.1 GB |

### Tasks

1. **Email Classification** — 5-class categorization (100 samples)
2. **Information Extraction** — Text → structured JSON (50 samples)
3. **Multi-Step Agent Workflow** — Lookup → Calculate → Format (50 samples)

### Metrics Captured

- Tokens/second (throughput)
- Time-to-first-token (TTFT)
- End-to-end latency (p50, p90, p95)
- Peak and steady-state RAM
- CPU utilization
- Task accuracy and completion rate
- Deployability Index (DI) — composite score

## Research Paper

The full paper is available at [`paper_v2.md`](paper_v2.md) and includes:

- Literature review of 19 papers (2023–2026)
- 4 formal hypotheses
- Full factorial experimental design (72 configurations)
- Novel Deployability Index (DI) metric
- 32 academic references

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{sahana2026deployability,
  title={Deployability of LLM-Based Agent Workflows on Resource-Constrained CPU-Only Systems: An Empirical Evaluation},
  author={Sahana and Fazil and Adwaith},
  year={2026}
}
```
