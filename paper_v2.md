# Deployability of LLM-Based Agent Workflows on Resource-Constrained CPU-Only Systems: An Empirical Evaluation

**Sahana, Fazil, Adwaith**

---

## Abstract

The proliferation of Large Language Model (LLM)-powered agent systems has been predominantly confined to cloud infrastructure, creating barriers related to cost, latency, privacy, and connectivity dependence. This paper presents an empirical evaluation of deploying LLM-based agent workflows on resource-constrained, CPU-only consumer hardware. We systematically benchmark four small language models—TinyLlama-1.1B, Phi-3-mini-3.8B, Qwen2.5-3B-Instruct, and Mistral-7B-Instruct-v0.3—across three GGUF quantization levels (Q4_K_M, Q5_K_M, Q8_0) on an Intel Core i5-1235U system with 16GB RAM and no discrete GPU. We evaluate these configurations across three task categories of increasing complexity: single-turn classification, structured information extraction, and multi-step agentic workflows involving tool orchestration. Our evaluation framework captures inference throughput (tokens/second), time-to-first-token (TTFT), end-to-end latency distributions (p50/p90/p95), peak memory consumption, CPU utilization, and task-specific accuracy. We propose a composite Deployability Index (DI) that quantifies the trade-off between task accuracy, resource efficiency, and operational reliability. Our findings demonstrate that [RESULTS: to be populated after experiments—e.g., quantized 3B-class models achieve X% accuracy relative to cloud baselines while reducing per-query cost by Y%, with Z tokens/second throughput]. We identify the critical thresholds at which model size, quantization aggressiveness, and task complexity render local deployment infeasible, providing practitioners with actionable deployment guidelines for edge and offline scenarios.

**Keywords:** Large Language Models, Edge AI, Quantization, Resource-Constrained Deployment, Small Language Models, LLM Agents, CPU Inference

---

## 1. Introduction

Large Language Models have fundamentally transformed the landscape of natural language processing, enabling sophisticated capabilities in text generation, reasoning, summarization, and autonomous task execution [1, 2]. The emergence of LLM-based agents—systems that combine language model inference with tool use, memory management, and multi-step reasoning—has further extended these capabilities into domains such as software engineering, scientific research, and enterprise automation [3, 4]. However, the deployment of these systems remains overwhelmingly dependent on cloud infrastructure equipped with high-end GPU accelerators, creating a paradigm where intelligence is centralized, expensive, and connectivity-dependent.

This cloud-centric deployment model introduces several critical limitations. First, API-based inference incurs per-token monetary costs that scale linearly with usage, rendering continuous agent operation economically prohibitive for resource-sensitive applications [5]. Second, network round-trip latency introduces delays that are unacceptable for real-time or interactive agent workflows. Third, transmitting potentially sensitive data to external servers raises privacy and data sovereignty concerns, particularly in regulated domains such as healthcare and finance [6]. Fourth, cloud dependence makes these systems entirely non-functional in offline or intermittently connected environments—a constraint that excludes a significant fraction of global deployment scenarios.

Concurrently, the period from 2023 to 2026 has witnessed remarkable advances in model compression and small language model (SLM) architectures. Post-training quantization techniques—particularly GGUF-based k-quant methods implemented in llama.cpp [7]—have demonstrated that model weights can be compressed from 16-bit floating point to 4-bit integers with minimal accuracy degradation [8, 9]. Simultaneously, architecturally efficient models such as Microsoft's Phi series [10, 11], Google's Gemma family [12], Meta's Llama 3.2 [13], and Alibaba's Qwen series [14] have achieved performance levels approaching frontier models at a fraction of the parameter count, specifically targeting deployment on consumer hardware.

Despite these advances, a critical gap persists in the literature: **there is no systematic empirical study evaluating the deployability of LLM-based agent workflows—as opposed to single-turn inference—on CPU-only consumer hardware under realistic resource constraints.** Existing benchmarks either focus on GPU-accelerated environments [15], evaluate single-turn generation quality without system-level metrics [8], or target extreme edge platforms (e.g., Raspberry Pi) where only sub-billion models are viable [16, 17]. The mid-range scenario—a standard consumer laptop with a modern CPU, 16GB RAM, and no discrete GPU—represents the most practically relevant deployment target for offline-capable agent systems, yet remains systematically understudied.

This paper addresses this gap through the following contributions:

1. **Systematic benchmarking** of four SLMs (1.1B–7B parameters) across three quantization levels on CPU-only consumer hardware, measuring both inference performance and task accuracy.

2. **Agent workflow evaluation** that extends beyond single-turn inference to assess multi-step tool-orchestrated workflows, capturing the compounding effects of sequential inference on latency and reliability.

3. **A composite Deployability Index (DI)** that provides a unified metric for comparing model-quantization-task configurations across the accuracy-efficiency-reliability trade-off space.

4. **Actionable deployment guidelines** identifying feasibility thresholds for model size, quantization level, and task complexity on resource-constrained hardware.

The remainder of this paper is organized as follows: Section 2 surveys related work in edge LLM deployment, quantization, and small language models. Section 3 presents the research methodology, including hypotheses, variable definitions, and experimental design. Section 4 details the experimental setup. Section 5 reports results. Section 6 discusses findings and implications. Section 7 addresses limitations. Section 8 concludes with future research directions.

---

## 2. Related Work

### 2.1 Edge AI and Local LLM Inference

The migration of AI inference from cloud to edge has been driven by latency, privacy, and cost imperatives. Saad-Falcon et al. [5] introduced the Intelligence per Watt (IPW) metric, demonstrating that local accelerators can outperform cloud systems in energy efficiency for 88.7% of real-world queries—a finding that challenges the assumption of cloud superiority for all workloads. Their longitudinal analysis (2023–2025) showed a 5.3× improvement in local IPW, with locally serviceable queries rising from 23.2% to 71.3%.

Stuhlmann et al. [15] proposed Bench360, a modular benchmarking framework that unifies task-specific quality evaluation with system-level metrics including energy consumption and cold-start latency across diverse inference backends. Their work highlighted the significant variance in performance across backends—a finding our study extends to CPU-only environments. Nguyen and Nguyen [16] evaluated 25 quantized models on single-board computers (Raspberry Pi, Orange Pi), finding that Llamafile achieved up to 4× higher throughput than Ollama on ARM platforms, attributable to optimized SIMD instruction utilization.

The cognitive edge computing survey by [6] provides a comprehensive taxonomy of edge AI architectures, identifying the convergence of on-device inference, federated learning, and neuromorphic computing as the defining trajectory of ambient intelligence. Our work contributes to this trajectory by specifically evaluating the agent workflow dimension—a dimension not addressed in existing edge inference benchmarks.

### 2.2 Quantization Methods for Local Deployment

Quantization is the foundational enabling technique for local LLM deployment. Three dominant paradigms have emerged, each with distinct trade-offs.

**GPTQ** (Generalized Post-Training Quantization) employs second-order Hessian information to minimize quantization error, achieving up to 4× speedup relative to FP16 on NVIDIA GPUs [18]. However, GPTQ requires substantial calibration data and exhibits quality degradation below 4-bit precision, limiting its utility for extreme compression.

**AWQ** (Activation-Aware Weight Quantization) preserves the top 1% of salient weights—those associated with large activation magnitudes—achieving 99% accuracy retention relative to full-precision baselines [19]. AWQ's quality preservation makes it preferred for tasks where coherence is critical, such as instruction following and multi-step reasoning.

**GGUF** (GPT-Generated Unified Format) has become the de facto standard for cross-platform CPU deployment. Its block-based k-quant system offers granular control over bits-per-weight, supporting formats from aggressive Q2_K (2-bit) to conservative Q8_0 (8-bit) [7]. Kurt [8] provided the first comprehensive evaluation of GGUF quantization levels on Llama-3.1-8B-Instruct, identifying FP8 as optimal for instruction-following integrity and Q4_K_M as the best compromise between size and quality. Our study extends this analysis to the agent workflow context, where quantization-induced accuracy degradation may compound across sequential inference steps.

Egashira et al. [20] exposed a critical security vulnerability in GGUF quantization, demonstrating that adversarial behaviors can be embedded in models that appear benign at full precision but activate malicious outputs when quantized to 4-bit k-quant. This "Mind the Gap" attack achieved an 88.7% success rate, highlighting that the local deployment community's trust in quantization as a purely mechanical compression process is unwarranted—a consideration relevant to any deployment guideline.

Advanced techniques such as SVDQuant [21] and SpecQuant [22] address the limitations of standard quantization for extreme compression. SVDQuant absorbs outlier values into a low-rank branch via Singular Value Decomposition, while SpecQuant applies Fourier-domain spectral truncation to preserve contextual cues at sub-3-bit precision. While these techniques target primarily GPU-accelerated diffusion and vision-language models, their principles inform the theoretical lower bounds of CPU-based quantized inference.

### 2.3 Small Language Models

The 2024–2026 period has produced a generation of architecturally efficient small language models specifically designed for constrained deployment.

**Microsoft Phi series.** The Phi family pioneered the "textbooks are all you need" data philosophy, prioritizing high-quality reasoning tokens over training volume. Phi-3-mini (3.8B parameters) introduced grouped-query attention and a 4K context window optimized for single-device deployment [10]. The subsequent Phi-4-Mini [11] extended this to multimodal processing via a Mixture-of-LoRAs architecture, achieving performance parity with models twice its size on complex reasoning tasks.

**Google Gemma.** The Gemma 3 family [12] introduced an interleaved local-global attention mechanism that reduces KV-cache memory by 5× while supporting 128K token contexts. This architectural innovation makes the 4B Gemma 3 model viable on 8GB RAM devices—a direct enabler for constrained deployment.

**Meta Llama 3.2.** The 1B and 3B Llama 3.2 models [13] were produced via structured pruning from Llama 3.1 8B, followed by knowledge distillation. These models specifically target tool-calling and agentic applications, with optimized grouped-query attention for efficient multi-turn dialogue.

**Alibaba Qwen.** The Qwen series includes Qwen2.5-3B-Instruct and the MoE-based Qwen3-Coder-Next (80B total / 3B active per token) [14]. The latter demonstrates that Mixture-of-Experts architectures can maintain the inference cost of a 3B model while accessing the knowledge capacity of an 80B model—illustrating the frontier of efficient model design.

**TinyLlama** [23] represents the extreme efficiency frontier at 1.1B parameters, trained on 3 trillion tokens from the SlimPajama dataset. Despite its diminutive size, TinyLlama achieves non-trivial performance on classification and extraction tasks, making it the baseline for the lowest viable agent deployment.

Liu et al. [24] (MobileLLM) established the "deep and thin" architectural principle for sub-billion models, demonstrating that depth is more important than width for small-scale language modeling. This principle underlies several of the models evaluated in our study.

### 2.4 LLM Agent Systems on Constrained Hardware

The deployment of LLM agents—as distinct from single-turn inference—on edge hardware introduces additional challenges related to multi-step latency accumulation, context management, and tool orchestration reliability.

The TinyLLM study [25] directly evaluates small language models for agentic tasks on edge devices, establishing that models below 3B parameters exhibit significant degradation in multi-step planning and tool selection accuracy. This finding informs our hypothesis regarding agent chain depth and model size interactions.

Zheng et al. [26] (HALO) addressed distributed agent inference in lossy edge networks by strategically allocating less-critical neuron groups to unreliable devices, enabling relaxed synchronization. While targeting multi-device scenarios, HALO's insights on inference partitioning inform our single-device analysis of resource allocation.

Matsutani et al. [27] demonstrated cooperative prompt caching across Raspberry Pi Zero 2W platforms, achieving a 93% reduction in time-to-first-token through distributed KV-cache state sharing. Their work highlights that memory bandwidth—not compute—is often the primary bottleneck for edge inference, a finding directly relevant to our CPU-only evaluation.

The security dimension of edge agent deployment has been characterized by Zhan et al. [28], who identified coordination-state divergence and trust erosion as critical failure modes in multi-agent IoT systems. Wang et al. [29] (OpenClaw-RL) proposed an asynchronous online reinforcement learning framework for continuous agent improvement at the edge. While these systems exceed our single-device scope, they contextualize the broader deployment ecosystem in which local agent workflows operate.

### 2.5 Research Gap

Existing literature addresses edge LLM inference (primarily on SBCs or GPUs), quantization benchmarking (primarily for single-turn tasks), and small model architecture (primarily measured by standard NLP benchmarks). However, **no existing study systematically evaluates the compound effects of quantization on multi-step agent workflows running CPU-only consumer hardware**, specifically characterizing how latency accumulation, accuracy degradation, and resource pressure interact across agent chain depths. This paper fills that gap.

---

## 3. Methodology

### 3.1 Research Hypotheses

Based on the literature review and the identified research gap, we formulate four testable hypotheses:

**H1 (Feasibility).** Quantized open-source LLMs in the 3B–7B parameter range (Q4_K_M quantization) can execute structured agent workflows on a CPU-only system with 16GB RAM, completing ≥90% of task chains without out-of-memory failure, crash, or timeout.

**H2 (Quality Retention).** Local agent deployments using Q4_K_M-quantized models achieve ≥70% of the task accuracy reported for their full-precision counterparts on standard benchmarks, with accuracy degradation increasing monotonically as quantization aggressiveness increases (Q8_0 → Q5_K_M → Q4_K_M).

**H3 (Latency-Cost Trade-off).** Local deployment eliminates per-query API costs entirely while increasing end-to-end agent task latency by a factor that varies predictably with model size: ≤2× for 1B-class models, ≤5× for 3B-class, and ≤10× for 7B-class, relative to published cloud API response times.

**H4 (Compounding Degradation).** In multi-step agent workflows, both latency and error rate compound super-linearly with chain depth, such that 3-step agent tasks exhibit disproportionately higher failure rates and latency than 3× single-step tasks.

### 3.2 Variables

#### 3.2.1 Independent Variables

| Variable | Levels | Rationale |
|---|---|---|
| Model | TinyLlama-1.1B, Phi-3-mini-3.8B, Qwen2.5-3B-Instruct, Mistral-7B-Instruct-v0.3 | Spans the 1B–7B parameter range feasible on 16GB RAM |
| Quantization Level | Q4_K_M, Q5_K_M, Q8_0 (GGUF) | Aggressive to conservative quantization |
| Task Complexity | Single-step (classification), Single-step (extraction), Multi-step (agent workflow) | Isolates agent-specific effects |
| Agent Chain Depth | 1, 2, 3 steps (for multi-step tasks only) | Tests compounding hypothesis (H4) |

#### 3.2.2 Dependent Variables

| Metric | Measurement Method | Unit |
|---|---|---|
| Task Accuracy | Correct outputs / total tasks (human-evaluated + automated) | % |
| Tokens per Second (tok/s) | Output tokens ÷ generation wall-clock time | tok/s |
| Time-to-First-Token (TTFT) | Prompt submission timestamp → first token emission | ms |
| End-to-End Latency | Total wall-clock time for complete agent task (all steps) | s |
| Latency Distribution | p50, p90, p95 across all trials | ms |
| Peak RAM Usage | Maximum Resident Set Size (maxRSS) during inference | MB |
| Steady-State RAM | Mean RSS after model load, during generation | MB |
| CPU Utilization | Mean and peak CPU% sampled at 100ms intervals | % |
| Task Completion Rate | Tasks completed without OOM/crash/timeout / total tasks | % |
| Model Load Time | Cold start → model ready for inference | s |

#### 3.2.3 Controlled Variables

To ensure internal validity, the following variables are held constant across all experimental conditions:

- **Prompt template:** Identical structured prompt per task category, with system instruction and few-shot examples fixed across models.
- **Temperature:** Set to 0.0 for deterministic output (reproducibility).
- **Maximum output tokens:** Fixed per task (128 tokens for classification, 256 for extraction, 512 for multi-step).
- **Context window:** Limited to 4096 tokens across all models.
- **Inference framework:** Ollama (wrapping llama.cpp) with identical runtime configuration.
- **System state:** Minimal background processes; documented running services.
- **Input data:** Identical test instances across all model-quantization configurations.

### 3.3 Experimental Design

We employ a **full factorial design** across model (4 levels) × quantization (3 levels) × task type (3 levels), yielding 36 experimental conditions. For the multi-step agent task, we additionally vary chain depth (3 levels), adding 36 agent-specific conditions for a total of 72 unique configurations.

Each configuration is evaluated across a minimum of 50 task instances, with 3 independent repetitions per instance, yielding ≥150 observations per condition for latency and resource metrics. This sample size provides statistical power to detect medium effect sizes (Cohen's d ≥ 0.5) at α = 0.05.

### 3.4 Agent Workflow Architecture

To evaluate agent deployability (as opposed to raw inference), we implement a structured agent workflow using a ReAct-style (Reasoning + Acting) loop [30]:

```
Agent Controller
├── Input Parser (task instruction → structured prompt)
├── LLM Inference (local, via Ollama API)
├── Output Parser (extract action or final answer)
├── Tool Router (dispatch to appropriate tool)
│   ├── Calculator (arithmetic operations)
│   ├── Lookup (key-value dictionary search)
│   └── Formatter (structured output generation)
└── Context Manager (append tool results, track state)
```

**Workflow execution:**
1. Receive task instruction
2. Construct prompt with system instruction, few-shot examples, and task input
3. LLM generates a response containing either a tool call or a final answer
4. If tool call: execute tool, append result to context, return to step 3
5. If final answer: record output, compute metrics
6. Maximum 5 iterations; exceeded iterations count as task failure

This architecture is intentionally minimal—sufficient to exercise agent capabilities (tool selection, multi-step reasoning, context accumulation) without introducing confounds from complex framework overhead. The agent is implemented as a Python script (approximately 200 lines) with no external agent framework dependencies.

### 3.5 Task Benchmark Design

#### Task 1: Email Classification (Single-Step)

- **Input:** Email text (50–200 words)
- **Output:** Category label (one of: inquiry, complaint, feedback, spam, urgent)
- **Dataset:** 100 manually curated and labeled email samples
- **Evaluation:** Accuracy, macro F1-score
- **Agent steps:** 1 (direct classification)

#### Task 2: Structured Information Extraction (Single-Step)

- **Input:** Unstructured text paragraph describing a product or event
- **Output:** JSON object with specified fields (name, date, location, price, category)
- **Dataset:** 100 curated paragraphs with gold-standard JSON annotations
- **Evaluation:** Field-level exact match, overall extraction F1-score
- **Agent steps:** 1 (direct extraction)

#### Task 3: Multi-Step Query Resolution (Agent Workflow)

- **Input:** A user query requiring information lookup, computation, and formatted response
- **Example:** "What is the total cost of 3 units of Product X at the listed price, including 18% GST? Format as an invoice."
- **Dataset:** 50 multi-step scenarios requiring 2–3 tool invocations
- **Evaluation:** Final answer correctness, tool selection accuracy, chain completion rate
- **Agent steps:** 2–3 (lookup → calculate → format)

#### Baseline Comparisons

Since our budget precludes API usage, we adopt cloud baselines from published benchmarks:
- GPT-3.5-Turbo and GPT-4o-mini accuracy scores from published evaluations on comparable task categories [31]
- Latency baselines from Bench360 [15] and published API response time measurements
- All baseline comparisons are explicitly noted as cross-study references, not controlled comparisons

### 3.6 Deployability Index

We propose a composite metric to enable single-score comparison across configurations:

$$DI = w_1 \cdot \frac{A}{A_{baseline}} + w_2 \cdot \frac{1}{L_{norm}} + w_3 \cdot \frac{1}{M_{norm}} + w_4 \cdot CR$$

Where:
- $A$ = task accuracy, $A_{baseline}$ = cloud baseline accuracy
- $L_{norm}$ = end-to-end latency normalized to fastest configuration
- $M_{norm}$ = peak RAM normalized to available system RAM (16GB)
- $CR$ = task completion rate (0–1)
- $w_1 = 0.35, w_2 = 0.25, w_3 = 0.15, w_4 = 0.25$ (weights reflecting practitioner priorities; sensitivity analysis provided)

The DI ranges from 0 (infeasible) to a theoretical maximum dependent on weight configuration. We conduct weight sensitivity analysis to demonstrate robustness.

---

## 4. Experimental Setup

### 4.1 Hardware Configuration

Experiments are conducted on two consumer-grade laptops to enable cross-device reproducibility validation:

**Device 1 (Primary):**

| Component | Specification |
|---|---|
| Processor | Intel Core i5-1235U (12th Gen Alder Lake) |
| Architecture | 2 Performance cores (up to 4.4 GHz) + 8 Efficient cores (up to 3.3 GHz), 12 threads |
| RAM | 16 GB DDR4 |
| Storage | 512 GB NVMe SSD |
| GPU | None (integrated Intel Iris Xe only, not used for inference) |
| Operating System | Ubuntu 22.04 LTS |
| Power Profile | Performance mode (plugged in) |

**Device 2 (Validation):**

| Component | Specification |
|---|---|
| Processor | Intel Core i5-1334U (13th Gen Raptor Lake) |
| Architecture | 2 Performance cores (up to 4.6 GHz) + 8 Efficient cores (up to 3.4 GHz), 12 threads |
| RAM | 16 GB DDR4 |
| GPU | None (integrated Intel Iris Xe only, not used for inference) |
| Power Profile | Performance mode (plugged in) |

Both devices represent common mid-range consumer laptops (2024–2025), making results directly applicable to a broad user base. Device 2 enables cross-generational validation (Section 4.5).

### 4.2 Software Configuration

| Component | Version |
|---|---|
| Inference Runtime | Ollama v0.x.x (to be specified) |
| Backend | llama.cpp (bundled with Ollama) |
| Agent Implementation | Custom Python 3.11 script |
| Monitoring | `psutil` (RAM, CPU), custom timing instrumentation |
| Model Format | GGUF (all models sourced from HuggingFace) |

### 4.3 Model Configurations

| Model | Parameters | GGUF Variants | Approximate Size (Q4_K_M / Q5_K_M / Q8_0) |
|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | 1.1B | Q4_K_M, Q5_K_M, Q8_0 | ~0.7 / ~0.8 / ~1.2 GB |
| Phi-3-mini-4k-instruct | 3.8B | Q4_K_M, Q5_K_M, Q8_0 | ~2.3 / ~2.7 / ~4.0 GB |
| Qwen2.5-3B-Instruct | 3.0B | Q4_K_M, Q5_K_M, Q8_0 | ~1.9 / ~2.2 / ~3.2 GB |
| Mistral-7B-Instruct-v0.3 | 7.2B | Q4_K_M, Q5_K_M, Q8_0 | ~4.1 / ~4.8 / ~7.7 GB |

All models are instruction-tuned variants selected for their chat/instruction-following capabilities, which are prerequisite for agent workflow execution.

### 4.4 Measurement Protocol

**System preparation:**
1. Reboot system; disable non-essential services and background processes
2. Verify available RAM ≥ 15.5 GB before each session
3. Set CPU governor to `performance` mode

**Per-condition execution:**
1. Cold-start Ollama; load target model; record model load time
2. Execute 5 warm-up inferences (results discarded)
3. Execute full task suite (50–100 instances per task type)
4. Record per-inference: tokens generated, generation time, TTFT, peak RSS, CPU%
5. Unload model; wait 30 seconds; proceed to next condition
6. Repeat entire sequence 3 times on separate days for variance estimation

**Data collection tooling:**
- Inference timing: Python `time.perf_counter()` at prompt submission and token callbacks
- Memory: `psutil.Process().memory_info().rss` sampled at 100ms intervals via background thread
- CPU: `psutil.cpu_percent(interval=0.1)` per-core logging
- All raw data logged to CSV per condition

### 4.5 Cross-Device Validation Design

To validate reproducibility and assess inter-generational CPU performance differences, a subset of experiments (E1: Baseline, E4: Cross-Device, E6: Cold vs Warm) are replicated identically on both devices. This two-device design enables:

1. **Reproducibility verification:** Confirming that rank ordering of model-quantization configurations is preserved across hardware variants.
2. **Generational delta quantification:** Measuring the throughput improvement from 12th → 13th Gen Intel Core processors sharing identical architecture (Alder Lake vs Raptor Lake) and RAM configuration.
3. **Statistical comparison:** Paired t-tests on identical workloads across devices, providing confidence intervals for hardware-induced performance variance.

All experiments use identical software stacks, model files, prompts, and measurement protocols across both devices.

---

## 5. Results

> [!IMPORTANT]
> **This section requires population with actual experimental data.** The following subsections define the exact tables, figures, and analyses to be completed after experiments are conducted. Placeholder annotations `[DATA]` indicate where measured values should be inserted.

### 5.1 Inference Performance

**Table 1: Throughput and Latency by Model and Quantization**

| Model | Quant | Tok/s | TTFT (ms) | Latency p50 (s) | Latency p90 (s) | Latency p95 (s) |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B | Q4_K_M | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |
| TinyLlama-1.1B | Q5_K_M | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |
| TinyLlama-1.1B | Q8_0 | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |
| Phi-3-mini-3.8B | Q4_K_M | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |
| Phi-3-mini-3.8B | Q5_K_M | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |
| Phi-3-mini-3.8B | Q8_0 | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |
| Qwen2.5-3B | Q4_K_M | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |
| Qwen2.5-3B | Q5_K_M | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |
| Qwen2.5-3B | Q8_0 | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |
| Mistral-7B | Q4_K_M | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |
| Mistral-7B | Q5_K_M | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |
| Mistral-7B | Q8_0 | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |

### 5.2 Resource Consumption

**Table 2: Memory and CPU Utilization**

| Model | Quant | Model Size (GB) | Peak RAM (MB) | Steady RAM (MB) | Avg CPU (%) | Peak CPU (%) | Load Time (s) |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | Q4_K_M | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |
| ... | ... | ... | ... | ... | ... | ... | ... |
| Mistral-7B | Q8_0 | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |

*Note: Mistral-7B Q8_0 (~7.7GB) may approach system memory limits. Record swap usage if applicable.*

### 5.3 Task Accuracy

**Table 3: Accuracy by Task and Model**

| Model | Quant | Classification Acc (%) | Classification F1 | Extraction EM (%) | Extraction F1 | Agent Accuracy (%) | Agent Completion Rate (%) |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | Q4_K_M | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |
| ... | ... | ... | ... | ... | ... | ... | ... |
| Mistral-7B | Q8_0 | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |
| *GPT-3.5-Turbo* | *Baseline* | *[published]* | *[published]* | *[published]* | *[published]* | *[published]* | *—* |

### 5.4 Agent Chain Depth Analysis

**Table 4: Multi-Step Performance by Chain Depth**

| Model | Quant | 1-Step Acc (%) | 2-Step Acc (%) | 3-Step Acc (%) | 1-Step Latency (s) | 2-Step Latency (s) | 3-Step Latency (s) |
|---|---|---|---|---|---|---|---|
| [DATA] | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |

*Analysis: Test H4 by comparing observed 3-step metrics against 3× single-step (linear expectation). Report compounding ratio = (3-step metric) / (3 × 1-step metric).*

### 5.5 Deployability Index Scores

**Table 5: DI Scores by Configuration**

| Model | Quant | DI (w1=0.35) | DI Rank | Feasible? |
|---|---|---|---|---|
| [DATA] | [DATA] | [DATA] | [DATA] | [DATA] |

*Weight sensitivity: Report DI under alternative weight vectors (accuracy-heavy: w1=0.5; latency-heavy: w2=0.4; balanced: equal weights).*

### 5.6 Planned Figures

1. **Figure 1:** System architecture diagram of the agent workflow
2. **Figure 2:** Bar chart — Tokens/second across all 12 model-quantization configurations
3. **Figure 3:** Box plot — Latency distribution per configuration (showing median, IQR, outliers)
4. **Figure 4:** Line chart — Accuracy vs. model size at each quantization level (per task)
5. **Figure 5:** Stacked bar chart — RAM consumption breakdown (model weights vs. KV-cache vs. runtime overhead)
6. **Figure 6:** Heatmap — Deployability Index across model × quantization × task type
7. **Figure 7:** Line chart — Agent accuracy and latency vs. chain depth (testing H4)

---

## 6. Discussion

> [!NOTE]
> The following discussion framework should be populated with interpretations of actual results. The structure and analytical direction are provided here.

### 6.1 Feasibility Thresholds

*Based on H1 results:* Discuss which model-quantization configurations achieved ≥90% task completion rate. Identify the specific failure modes (OOM, timeout, malformed output) and their distribution across configurations. Characterize the boundary between feasible and infeasible deployment: does the i5-1235U / 16GB RAM system support 7B Q8_0 inference, or does swap thrashing render it impractical?

### 6.2 The Quantization-Accuracy Trade-off in Agent Context

*Based on H2 results:* Analyze whether the quantization → accuracy relationship observed in single-turn benchmarks [8] holds for multi-step agent workflows. If accuracy degradation compounds across agent steps (as hypothesized in H4), this has implications for quantization selection in agent deployments that differ from single-turn guidelines.

### 6.3 Cost-Latency Equilibrium

*Based on H3 results:* Compute the effective per-query cost of local deployment (hardware amortization) versus API pricing. Identify the "break-even" query volume at which local deployment becomes cost-advantageous. Discuss latency tolerance: for which applications is the local latency overhead acceptable?

### 6.4 The Compounding Problem

*Based on H4 results:* If confirmed, super-linear latency and error compounding in agent workflows represents the most significant practical barrier to local agent deployment. Discuss mitigation strategies: prompt caching, model warm-up, reduced chain depth, hybrid architectures where the first agent step uses a larger model and subsequent steps use a faster model.

### 6.5 Practical Deployment Guidelines

Based on the aggregate results, provide a decision matrix:

| Scenario | Recommended Configuration | Rationale |
|---|---|---|
| Offline classification | [Best from results] | [Based on accuracy/latency trade-off] |
| Cost-sensitive extraction | [Best from results] | [Based on cost analysis] |
| Interactive agent | [Best from results] | [Based on latency constraints] |
| Maximum accuracy (local) | [Best from results] | [Based on DI score] |

### 6.6 Comparison with Prior Work

Compare findings with:
- Stuhlmann et al. [15] (Bench360) — extend their single-turn findings to agent context
- Nguyen and Nguyen [16] — compare SBC performance with consumer laptop performance
- Saad-Falcon et al. [5] — validate/challenge their IPW findings in our specific hardware context
- TinyLLM [25] — compare agent task thresholds

---

## 7. Limitations

1. **Limited hardware diversity.** While experiments are validated across two Intel Core i5 processors (12th and 13th Gen), results may not generalize to AMD processors, Apple Silicon, or ARM-based systems, which have fundamentally different memory architectures and SIMD capabilities.

2. **No GPU baseline.** The absence of a local GPU comparison means we cannot quantify the GPU → CPU performance gap on identical hardware. Published GPU benchmarks serve as approximate baselines but introduce cross-study confounds.

3. **Cloud baselines from literature.** API-based baselines are drawn from published studies rather than controlled experiments with identical prompts and tasks. This limits the precision of accuracy comparisons, though latency and cost comparisons remain valid.

4. **Custom task datasets.** The classification and extraction datasets are internally curated rather than established benchmarks (e.g., MMLU, HellaSwag). While this limits comparability with other studies, it enables controlled evaluation of agent-specific behaviors that standard benchmarks do not capture.

5. **Deterministic sampling.** Using temperature = 0.0 eliminates output variance but may not represent typical user-facing deployments where sampling diversity is desired.

6. **Limited agent complexity.** The 3-tool, 3-step agent workflow is intentionally minimal. Real-world agents may involve more tools, longer chains, and persistent memory, all of which would amplify the effects observed here.

7. **No energy measurement.** Direct power measurement requires hardware instrumentation (e.g., a power meter) not available in our setup. Software-estimated energy metrics (via RAPL or similar) are less reliable, and we therefore omit energy as a primary metric.

8. **Security not evaluated.** We do not empirically test the GGUF backdoor vulnerabilities described by Egashira et al. [20], though we acknowledge this as a critical concern for local deployment.

---

## 8. Conclusion

This study presents an empirical evaluation of LLM-based agent workflow deployability on resource-constrained, CPU-only consumer hardware. By systematically benchmarking four small language models across three quantization levels and three task complexity tiers, we provide the first targeted analysis of how model compression, hardware constraints, and agent workflow complexity interact in a practical deployment scenario.

Our key findings include: [TO BE COMPLETED WITH ACTUAL RESULTS]

1. **Feasibility:** [Which configurations are viable on i5-1235U / 16GB RAM]
2. **Quality retention:** [How quantization affects agent task accuracy vs. published baselines]
3. **Compounding effects:** [Whether multi-step agent workflows exhibit super-linear degradation]
4. **Optimal configurations:** [Best model-quantization pairs per use-case, as captured by the Deployability Index]

The proposed Deployability Index provides practitioners with a composite metric for evaluating local deployment configurations, enabling informed trade-off decisions between accuracy, latency, resource consumption, and reliability.

As the edge AI ecosystem matures—driven by architectural innovations in small language models [10, 12, 13], advancing quantization techniques [19, 21, 22], and distributed inference frameworks [26, 27]—the findings of this study offer empirical grounding for the transition from cloud-centric to locally-deployed intelligent agent systems. Future work should extend this evaluation to heterogeneous hardware (Apple Silicon, ARM-based devices), larger agent frameworks with persistent memory, and hybrid cloud-edge architectures that dynamically route inference based on task complexity and available resources.

---

## References

[1] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[2] J. Wei et al., "Chain-of-thought prompting elicits reasoning in large language models," in *NeurIPS*, 2022.

[3] S. Yao et al., "ReAct: Synergizing reasoning and acting in language models," in *ICLR*, 2023.

[4] T. Schick et al., "Toolformer: Language models can teach themselves to use tools," in *NeurIPS*, 2023.

[5] J. Saad-Falcon, A. Narayan, H. O. Akengin et al., "Intelligence per watt: Measuring intelligence efficiency of local AI," *arXiv preprint arXiv:2505.xxxxx*, 2025.

[6] "Cognitive edge computing: A comprehensive survey," *arXiv preprint arXiv:2501.03265*, 2025.

[7] G. Gerganov, "llama.cpp: Inference of LLaMA model in pure C/C++," GitHub Repository, 2023–2026.

[8] U. Kurt, "Which quantization should I use? A unified evaluation of llama.cpp quantization on Llama-3.1-8B-Instruct," *arXiv preprint*, 2026.

[9] "PalmBench: A comprehensive benchmark of compressed large language models on mobile platforms," in *ICLR*, 2025.

[10] Microsoft Research, "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone," *arXiv preprint arXiv:2404.14219*, 2024.

[11] A. Abouelenin, A. Ashfaq et al., "Phi-4-Mini technical report: Compact yet powerful multimodal language models via Mixture-of-LoRAs," *arXiv preprint*, 2025.

[12] Gemma Team, "Gemma 3 Technical Report," *arXiv preprint*, 2025.

[13] Meta AI, "Llama 3.2: Lightweight text models," Meta AI Blog and Technical Report, 2024.

[14] R. Cao, M. Chen et al., "Qwen3-Coder-Next Technical Report," *arXiv preprint*, 2026.

[15] L. Stuhlmann, M. Fadel Argerich, J. Fürst, "Bench360: Benchmarking local LLM inference from 360 degrees," *arXiv preprint*, 2025.

[16] T. Nguyen, T. Nguyen, "An evaluation of LLMs inference on popular single-board computers," *arXiv preprint*, 2025.

[17] H. Matsutani, N. Matsuda, N. Sugiura, "Accelerating local LLMs on resource-constrained edge devices via distributed prompt caching," *arXiv preprint*, 2026.

[18] E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh, "GPTQ: Accurate post-training quantization for generative pre-trained transformers," in *ICLR*, 2023.

[19] J. Lin, J. Tang, H. Tang, S. Yang, X. Dang, S. Han, "AWQ: Activation-aware weight quantization for LLM compression and acceleration," in *MLSys*, 2024.

[20] K. Egashira, R. Staab, M. Vero et al., "Mind the gap: A practical attack on GGUF quantization," in *ICML*, 2025.

[21] M. Li et al., "SVDQuant: Absorbing outliers by low-rank components for 4-bit diffusion models," *arXiv / ICLR*, 2024/2025.

[22] "SpecQuant: Extreme LLM compression from a Fourier frequency domain perspective," *arXiv preprint*, 2025/2026.

[23] P. Zhang et al., "TinyLlama: An open-source small language model," *arXiv preprint arXiv:2401.02385*, 2024.

[24] Z. Liu, C. Zhao et al., "MobileLLM: Optimizing sub-billion parameter language models for on-device use cases," in *ICML*, 2024.

[25] "TinyLLM: Evaluation and optimization of small language models for agentic tasks on edge devices," *arXiv preprint arXiv:2511.22138*, 2025.

[26] P. Zheng, W. Xu, H. Wang, J. Chen, X. Shen, "HALO: Semantic-aware distributed LLM inference in lossy edge network," in *IEEE INFOCOM*, 2026.

[27] H. Matsutani, N. Matsuda, N. Sugiura, "Accelerating local LLMs on resource-constrained edge devices via distributed prompt caching," *arXiv preprint*, 2026.

[28] Z. Zhan, K. Li, Y. Zhang, H. Haddadi, "Systems-level attack surface of edge agent deployments on IoT," *arXiv preprint*, 2026.

[29] Y. Wang, X. Chen, X. Jin et al., "OpenClaw-RL: Train any agent simply by talking," *arXiv preprint*, 2026.

[30] S. Yao et al., "ReAct: Synergizing reasoning and acting in language models," in *ICLR*, 2023.

[31] OpenAI, "GPT-4 Technical Report," *arXiv preprint arXiv:2303.08774*, 2023.

[32] "Sustainable LLM inference for edge AI: Evaluating quantized LLMs for energy efficiency, output accuracy, and inference latency," *ResearchGate*, 2025.
