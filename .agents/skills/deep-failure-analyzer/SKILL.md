---
name: deep-failure-analyzer
description: Correlates system performance metrics with agent failures to identify root causes like quantization noise, context saturation, or I/O bottlenecking.
version: 1.1.0
status: research-active
---

This skill provides the "Why" behind agent task failures, correlating hardware stress with logical errors.

## Triggers
- Any task where `answer_correct` is False.
- Any task where `ttft_ms` exceeds the p90 threshold of the baseline.

## Logic
1. **Collate Evidence**:
    - Get `total_time_s` and `ttft_ms` from inference logs.
    - Get `peak_ram_mb` and `avg_cpu_pct` from `ResourceMonitor.get_stats()`.
    - Inspect LLM's `response` for hallucinations or "Context Collapse" (repetition).
2. **Root Cause Mapping**:
    - **Quantization Noise**: If `error` is a small math discrepancy (e.g. 1593 vs 1603) and Quant is Q4.
    - **Context Saturation**: If LLM output is repetitive or forgets the system prompt in long contexts.
    - **I/O Bottlenecking**: If `swap_usage > 10%` and `ttft_ms` is $> 2\times$ the baseline p90, classify as a filesystem/memory swap delay rather than a model error.
    - **Thermal Throttling**: If `cpu_pct` is high but `tok/s` dropped mid-run.
3. **Verification**: If `recovery-accuracy-evaluator` fixed the error with Q8, confirm **Quantization Noise** as the cause.

## Metric Logging
- Append `failure_root_cause` to the research dataset.
- Provide a summary sentence: "Failure in Sample #X due to [Cause] under [Pressure Condition]."
