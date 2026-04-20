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
