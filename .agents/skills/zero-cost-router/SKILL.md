---
name: zero-cost-router
description: Optimizes resource usage by routing tasks to appropriate models based on rule-based complexity analysis with minimal overhead.
version: 1.1.0
status: research-active
---

This skill implements an extremely low-overhead task router to optimize the "Efficiency Frontier" between 1B and 7B models.

## Triggers
- Receipt of any new `user_query` or `experiment_task`.

## Logic
1. **Zero-Cost Analysis**: Use regex to classify the task difficulty without calling an LLM.
   ```python
   import re
   complexity_keywords = ["lookup", "calculate", "format", "invoice", "math", "gst"]
   matches = [k for k in complexity_keywords if k in query.lower()]
   
   if len(matches) >= 3:
       decision = "Mistral-7B"
   else:
       decision = "TinyLlama-1.1B"
   ```
2. **Execute Routing**: Load the corresponding model from `lib.ALL_MODELS`.
3. **Mandatory Resource Logging**:
   - Record `routing_overhead_ms`.
   - Call `ResourceMonitor.get_stats()` for the chosen model's generation.

## Metric Logging
- Record `router_decision`: "Pass-through" (match < 3) vs "Escalated" (match >= 3).
- Calculate **Compute Saved** = (Avg RAM 7B model) - (Avg RAM Chosen model).
- Log: `{"query": Q, "routed_to": M, "overhead_ms": T}`.
