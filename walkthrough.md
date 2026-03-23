# Walkthrough: 2-Device Experiment Pipeline

## What Was Done

Created a complete experimental pipeline for 6 experiments across 2 devices (i5-1235U and i5-1334U).

## Files Created (10 new)

| File | Purpose |
|---|---|
| [experiments/lib.py](file:///Users/tom/Documents/Research%20Paper/experiments/lib.py) | Shared library: ResourceMonitor, Ollama wrappers, CLI parser, CSV/JSON helpers |
| [experiments/e1_baseline.py](file:///Users/tom/Documents/Research%20Paper/experiments/e1_baseline.py) | E1: 4 models × 20 runs, fixed prompt, tok/s + TTFT + latency |
| [experiments/e2_quantization.py](file:///Users/tom/Documents/Research%20Paper/experiments/e2_quantization.py) | E2: 12 configs × throughput (20 runs) + classification (100 emails) |
| [experiments/e3_agent_overhead.py](file:///Users/tom/Documents/Research%20Paper/experiments/e3_agent_overhead.py) | E3: Single-step vs 3-step agent chain × 30 runs, overhead ratio |
| [experiments/e4_cross_device.py](file:///Users/tom/Documents/Research%20Paper/experiments/e4_cross_device.py) | E4: 2 models × 3 prompt lengths × 30 runs on both devices |
| [experiments/e5_memory.py](file:///Users/tom/Documents/Research%20Paper/experiments/e5_memory.py) | E5: 4-phase RAM monitoring (baseline→load→infer→recovery) + swap detection |
| [experiments/e6_coldwarm.py](file:///Users/tom/Documents/Research%20Paper/experiments/e6_coldwarm.py) | E6: 5 cold starts + 20 warm runs, TTFT comparison |
| [experiments/setup_models.sh](file:///Users/tom/Documents/Research%20Paper/experiments/setup_models.sh) | One-command setup (Ollama + models + Python deps) |
| [experiments/download_gguf.py](file:///Users/tom/Documents/Research%20Paper/experiments/download_gguf.py) | Downloads Q5/Q8 GGUF variants from HuggingFace + auto-registers in Ollama |
| [experiments/README.md](file:///Users/tom/Documents/Research%20Paper/experiments/README.md) | Quick reference, setup, running instructions, 4-week schedule |

## Files Modified (3)

| File | Changes |
|---|---|
| [README.md](file:///Users/tom/Documents/Research%20Paper/README.md) | Added 2-device table, experiments/ directory, updated structure + quick start |
| [paper_v2.md](file:///Users/tom/Documents/Research%20Paper/paper_v2.md) | Section 4.1: 2-device hardware. New Section 4.5: cross-device validation. Section 7: softened limitation #1 |
| [.gitignore](file:///Users/tom/Documents/Research%20Paper/.gitignore) | Added `models/`, `results/*.csv` |

## Verification

All 8 Python scripts pass `py_compile` — zero syntax errors.

## Execution Schedule

| Week | Device 1 (i5-1235U) | Device 2 (i5-1334U) |
|------|---------------------|---------------------|
| 1 | Setup + E1 + E6 (~3h) | Setup + E1 + E6 (~3h) |
| 2 | E2 full quantization (~6h) | E4 cross-device (~2h) |
| 3 | E3 agent overhead + E5 memory (~5h) | E3 phi3 only, validation (~1h) |
| 4 | Analysis + paper writing | — |

**Total compute:** ~20h (Device 1) + ~6h (Device 2) = **~26 hours**.
