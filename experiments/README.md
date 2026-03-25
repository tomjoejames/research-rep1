# Experiments

This directory contains the complete experiment pipeline for the research paper.

## Quick Reference

| Script | Experiment | Device | Est. Time |
|--------|-----------|--------|-----------|
| `e1_baseline.py` | Baseline inference throughput | Both | 2h × 2 |
| `e2_quantization.py` | Quantization impact (speed + accuracy) | D1 | 6h |
| `e3_agent_overhead.py` | Agent vs non-agent latency | D1 (D2 validation) | 3h |
| `e4_cross_device.py` | Cross-device reproducibility | Both | 2h × 2 |
| `e5_memory.py` | RAM pressure + swap detection | D1 | 2h |
| `e6_coldwarm.py` | Cold-start vs warm TTFT | Both | 1h × 2 |

## Setup

```bash
# One-time setup (run on each device):
chmod +x experiments/setup_models.sh
./experiments/setup_models.sh

# For Q5/Q8 variants needed by E2:
python experiments/download_gguf.py
```

## Running

All scripts accept `--device`, `--models`, `--runs`, and `--output-dir` flags.

```bash
# On Device 1 (i5-1235U):
python experiments/e1_baseline.py --device device1
python experiments/e2_quantization.py --device device1
python experiments/e3_agent_overhead.py --device device1
python experiments/e4_cross_device.py --device device1
python experiments/e5_memory.py --device device1
python experiments/e6_coldwarm.py --device device1

# On Device 2 (i5-1334U):
python experiments/e1_baseline.py --device device2
python experiments/e4_cross_device.py --device device2
python experiments/e6_coldwarm.py --device device2
```

## Shared Code

- `lib.py` — shared library (ResourceMonitor, Ollama wrappers, CLI parser)
- `setup_models.sh` — environment setup
- `download_gguf.py` — Q5/Q8 model downloader

## Output

All results go to `results/` as timestamped CSV (and JSON for time-series data).

## 4-Week Schedule

| Week | Device 1 | Device 2 |
|------|----------|----------|
| 1 | Setup + E1 + E6 | Setup + E1 + E6 |
| 2 | E2 (full quantization) | E4 |
| 3 | E3 + E5 | E3 (phi3 only, validation) |
| 4 | Analysis + paper writing | — |
