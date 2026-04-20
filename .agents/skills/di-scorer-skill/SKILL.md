---
name: di-scorer-skill
description: Aggregates experimental metrics to compute the composite Deployability Index (DI) score using research-defined weights.
version: 1.1.0
status: research-active
---

This master skill aggregates all experimental metrics to calculate the final **Deployability Index (DI)** for the paper.

## Triggers
- Completion of an experiment batch (e.g., all 50 samples for a model-quant pair).

## Logic
1. **Aggregate Metrics**:
    - **Filter Logic**: Ignore any runs marked as 'Infeasible' by the `kv-cache-pressure-monitor` to avoid skewing accuracy metrics in `paper_v2.md`.
    - Accuracy ($A$): Mean `answer_correct` from logs (filtered).
    - Latency ($L$): Mean `total_time_s`.
    - Memory ($M$): Mean `peak_ram_mb`.
    - Completion Rate ($CR$): `successful_runs / total_runs`.
2. **Normalize Values**:
    - $A_{base} = 85.0$ (Cloud Baseline).
    - $L_{norm} = L / min(L_{all\_configs})$.
    - $M_{norm} = M / 16384$ (Fraction of 16GB RAM).
3. **DI Calculation**:
   Use the exact weights from the research paper:
   $$DI = 0.35 \cdot (A/A_{base}) + 0.25 \cdot (1/L_{norm}) + 0.15 \cdot (1/M_{norm}) + 0.25 \cdot CR$$
4. **Research Analysis**: Call `ResourceMonitor.get_stats()` used by all sub-tasks to ensure $M_{norm}$ reflects peak background pressure.

## Metric Logging
- Generate a summary markdown table:
  | Model | Quant | DI Score | Feasible (H1)? |
  | :--- | :--- | :--- | :--- |
  | ... | ... | ... | ... |
- Append to `paper_v2_results.json`.
