---
name: recovery-accuracy-evaluator
description: Validates agent outputs against ground truth and triggers quantization step-up on failure to measure the accuracy-precision trade-off.
version: 1.1.0
status: research-active
---

This skill ensures agentical tasks are verified against the research "Ground Truth" and provides data on how quantization noise affects multi-step reasoning.

## Triggers
- Immediately after any inference step in an `agent` task.
- When an agent generates a "Final Answer".

## Logic
1. **Identify Ground Truth**: Open `c:\Adhi\Startup\Research\research-rep1-main\data\agent_tasks.json` and find the task entry matching the current `sample_id`.
2. **Extract Numeric Result**: Parse the LLM's response to find the "Total" or "Invoice Amount".
3. **Accuracy Check**: Compare the LLM's value to `expected_answer["total"]`.
4. **Conditional Recovery**: 
    - If values match: Log "Accuracy=100%" and proceed.
    - If values mismatch: 
        1. Log the current model-quant combo.
        2. Trigger a **Quantization Step-Up**: Unload the current Q4/Q5 model and load the `mistral-q8` or `phi3-q8` variant.
        3. Re-execute the query.
        4. Calculate **Recovery Latency** = `time_q8 - time_initial`.
5. **Mandatory Resource Logging**:
   ```python
   # From lib.py
   from experiments.lib import ResourceMonitor
   rm = ResourceMonitor()
   rm.start() 
   # ... [inference code] ...
   stats = rm.get_stats()
   print(f"RESEARCH_LOG: {stats}")
   ```

## Metric Logging
- Append `{sample_id, status: "recovered", recovery_latency: X, quant_noise: Y}` to the experiment log.
