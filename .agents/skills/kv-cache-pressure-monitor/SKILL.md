---
name: kv-cache-pressure-monitor
description: Monitors RAM and swap usage to evaluate hypothesis H1 regarding hardware feasibility under increasing context pressure.
version: 1.1.0
status: research-active
---

This skill tests Hypothesis **H1 (Feasibility)** by measuring how increasing context length impacts system stability (Swap vs RAM) on 16GB systems.

## Triggers
- When the `current_context` exceeds 2048 tokens.
- For every 3 turns in a multi-step orchestration.

## Logic
1. **Monitor System Limits**: Before the next inference turn, check current system memory state.
   ```python
   import psutil
   mem = psutil.virtual_memory()
   swap = psutil.swap_memory()
   proc_mem = psutil.Process().memory_info().rss / (1024**2)
   ```
2. **Threshold Validation**:
    - If `swap.percent > 15%`: Log as "Memory Thrashing Detected".
    - If `mem.available < 500MB`: Log as "High RAM Pressure".
3. **Research Attribution**: If a timeout or crash occurs, check if it correlates with `swap.percent > 15%`. If so, attribute failure to **Hardware Feasibility (H1 Violation)**.
4. **Mandatory Resource Logging**: Stop the `ResourceMonitor` from `lib.py` and capture `peak_ram_mb`.

## Metric Logging
- Record `swap_pressure_score` (0.0 to 1.0) based on `swap.used / total_swap`.
- Append to research log: `{"context_len": N, "h1_feasible": False, "swap_pct": S}`.
