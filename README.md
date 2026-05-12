# Deployability of LLM-Based Agent Workflows on Resource-Constrained CPU-Only Systems

**An Empirical Evaluation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Authors

Sahana, Adwaith, Fazil 

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
# Deployability of LLM-Based Agent Workflows on Resource-Constrained CPU-Only Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Authors:** Sahana Srinivasan, Adwaith Balakrishnan, Tom Joe James, Mohamed Fazil RM

## Abstract
This repository contains the replication code, datasets, and empirical results for our paper evaluating the deployability of small language models (SLMs) on CPU-bound edge devices. We systematically benchmark five models (TinyLlama-1.1B, Phi-3-mini-3.8B, Llama-3.2-3B, Qwen2.5-3B, Mistral-7B) across progressive quantization sweeps (Q4 to FP16) on 12th and 13th Gen Intel Core i5 architectures. We define a novel **Deployability Index** calculating the trade-offs between agentic accuracy, latency, and memory footprints.

## Zenodo Data Release
To accommodate file size limitations and ensure permanent archival, the underlying datasets and script pipelines have been bundled into Zenodo releases:
* `replication_dataset.zip` (Raw classification scores, latency telemetry, timeseries RAM usage)
* `replication_code.zip` (Python benchmarking scripts and shell environments)
* `manuscript_assets.zip` ( LaTeX source and hi-res publication figures)

## Repository Structure

```text
.
├── CITATION.cff                       # Zenodo metadata file
├── README.md                          # This file
├── manuscript/
│   ├── paper_v2.md                    # Primary manuscript markdown
│   ├── paper_v2.html                  # Self-contained readable HTML
│   ├── latex/                         # Compiled IEEE conference format
│   └── figures/                       # Publication-ready plots
├── experiments/                       # Automated SLM test harness
├── scripts/                           # Statistical analysis & graphics
├── data/                              # Evaluation prompts & MMLU anchors
└── results/                           # Empirical telemetry output
```

## Key Findings Summary
1. **The 3B Parameter Sweet Spot:** `Llama-3.2-3B` and `Qwen2.5-3B` securely occupy the optimal edge-deployment frontier, matching 7B model logic while operating within a 3GB RAM envelope at ~6 tok/s.
2. **Quantization Collapse:** At Q4 quantization, sub-2B models (e.g., TinyLlama) suffer total logic collapse, failing basic mathematical agent tasks (0% correctness), while 3B+ models retain >95% of their FP16 logic fidelity.
3. **Cross-Device Scaling:** Our Deployability Index remains highly stable across architectures; replication between 12th Gen (Alder Lake) and 13th Gen (Raptor Lake) showed a uniform 18-22% scaling increment across all models.

## Citation
Please use the following BibTeX when referencing this work or dataset:

```bibtex
@misc{srinivasan2026deployability,
  title={Deployability of LLM-Based Agent Workflows on Resource-Constrained CPU-Only Systems},
  author={Srinivasan, Sahana and Balakrishnan, Adwaith and James, Tom Joe and RM, Mohamed Fazil},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.XXXXXXX},
  url={https://doi.org/10.5281/zenodo.XXXXXXX}
}
```
