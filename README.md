# Deployability of LLM-Based Agent Workflows on Resource-Constrained CPU-Only Systems

**An Empirical Evaluation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Authors

Sahana, Fazil, Adwaith

## Abstract

This research evaluates the feasibility of deploying LLM-based agent workflows on resource-constrained, CPU-only consumer hardware. We benchmark four small language models—TinyLlama-1.1B, Phi-3-mini-3.8B, Qwen2.5-3B-Instruct, and Mistral-7B-Instruct-v0.3—across three GGUF quantization levels (Q4_K_M, Q5_K_M, Q8_0) on two Intel Core i5 systems (12th/13th Gen) with 16GB RAM and no discrete GPU.

## Repository Structure

```
.
├── README.md                          # This file
├── paper_v2.md                        # Full research paper
├── draft_v1.md                        # Original draft (archived)
├── literature_review_references.md    # Collected references (19 papers)
├── requirements.txt                   # Python dependencies
├── run_experiment.py                  # Legacy combined benchmark script
├── analyze_results.py                 # Results analysis & figure generation
├── experiments/                       # ⭐ Experiment pipeline
│   ├── README.md                      #   Execution guide & schedule
│   ├── lib.py                         #   Shared library (monitors, Ollama wrappers)
│   ├── setup_models.sh                #   One-command environment setup
│   ├── download_gguf.py               #   Q5/Q8 GGUF downloader
│   ├── e1_baseline.py                 #   E1: Baseline inference throughput
│   ├── e2_quantization.py             #   E2: Quantization impact (speed + accuracy)
│   ├── e3_agent_overhead.py           #   E3: Agent vs non-agent latency
│   ├── e4_cross_device.py             #   E4: Cross-device reproducibility
│   ├── e5_memory.py                   #   E5: Memory pressure & swap detection
│   └── e6_coldwarm.py                 #   E6: Cold-start vs warm TTFT
├── data/
│   ├── emails.json                    # Email classification dataset (100 samples)
│   ├── extraction.json                # Information extraction dataset (50 samples)
│   └── agent_tasks.json               # Multi-step agent tasks (50 samples)
└── results/                           # Experiment output (auto-generated)
```

## Test Devices

| Device | CPU | RAM | GPU | Role |
|--------|-----|-----|-----|------|
| Device 1 | Intel i5-1235U (12th Gen) | 16 GB | None | Primary — all 6 experiments |
| Device 2 | Intel i5-1334U (13th Gen) | 16 GB | None | Validation — E1, E4, E6 |

## Quick Start

### 1. Setup (run on each device)

```bash
chmod +x experiments/setup_models.sh
./experiments/setup_models.sh
```

### 2. Run Experiments

```bash
ollama serve &

# Device 1 — full pipeline
python experiments/e1_baseline.py --device device1
python experiments/e2_quantization.py --device device1
python experiments/e3_agent_overhead.py --device device1
python experiments/e4_cross_device.py --device device1
python experiments/e5_memory.py --device device1
python experiments/e6_coldwarm.py --device device1

# Device 2 — validation subset
python experiments/e1_baseline.py --device device2
python experiments/e4_cross_device.py --device device2
python experiments/e6_coldwarm.py --device device2
```

### 3. Analyze Results

```bash
python analyze_results.py results/experiment_run_*.json
```

## Experiment Overview

| # | Experiment | Tests | Est. Time |
|---|-----------|-------|-----------|
| E1 | Baseline Inference | Throughput vs model size | 2h × 2 |
| E2 | Quantization Impact | Q4/Q5/Q8 accuracy-speed tradeoff | 6h |
| E3 | Agent Overhead | Multi-step chain latency penalty | 3h |
| E4 | Cross-Device | i5-12th vs i5-13th Gen comparison | 2h × 2 |
| E5 | Memory Pressure | RAM/swap behavior under load | 2h |
| E6 | Cold vs Warm | TTFT reduction after model caching | 1h × 2 |

### Models Evaluated

| Model | Parameters | Size (Q4_K_M) |
|-------|-----------|---------------|
| TinyLlama-1.1B | 1.1B | ~700 MB |
| Phi-3-mini-3.8B | 3.8B | ~2.3 GB |
| Qwen2.5-3B-Instruct | 3.0B | ~1.9 GB |
| Mistral-7B-Instruct-v0.3 | 7.2B | ~4.1 GB |

### Metrics Captured

- Tokens/second (throughput)
- Time-to-first-token (TTFT)
- End-to-end latency (p50, p90, p95)
- Peak and steady-state RAM + swap detection
- CPU utilization
- Task accuracy and completion rate
- Agent overhead ratio
- Cold vs warm TTFT delta
- Deployability Index (DI) — composite score

## Research Paper

The full paper is at [`paper_v2.md`](paper_v2.md) and includes:

- Literature review of 19 papers (2023–2026)
- 4 formal hypotheses
- 6 experiments across 2 devices
- Novel Deployability Index (DI) metric
- 32 academic references

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{sahana2026deployability,
  title={Deployability of LLM-Based Agent Workflows on Resource-Constrained CPU-Only Systems: An Empirical Evaluation},
  author={Sahana and Fazil and Adwaith},
  year={2026}
}
```
