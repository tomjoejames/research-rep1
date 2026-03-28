# Literature Review — Reference Material

## Collected Papers (2023–2026)

### 1. Bench360: Benchmarking Local LLM Inference from 360 Degrees
- **Authors:** Linus Stuhlmann, Mauricio Fadel Argerich, Jonathan Fürst
- **Year:** 2025/2026 | **Venue:** arXiv (cs.CL)
- **Summary:** Modular framework for evaluating local LLM inference across backends, quantization formats, and hardware. First unified evaluation of task-specific quality alongside system metrics (energy, cold-start latency).

### 2. HALO: Semantic-Aware Distributed LLM Inference in Lossy Edge Network
- **Authors:** Peirong Zheng, Wenchao Xu, Haozhao Wang, Jinyu Chen, Xuemin Shen
- **Year:** 2026 | **Venue:** IEEE INFOCOM
- **Summary:** Distributed inference in unreliable networks. Allocates less critical neuron groups to unstable devices with relaxed synchronization.

### 3. Accelerating Local LLMs on Resource-Constrained Edge Devices via Distributed Prompt Caching
- **Authors:** Hiroki Matsutani, Naoki Matsuda, Naoto Sugiura
- **Year:** 2026 | **Venue:** arXiv (cs.DC)
- **Summary:** Cooperative prompt caching across multiple edge devices. 93% reduction in TTFT on Raspberry Pi Zero 2W.

### 4. Intelligence per Watt: Measuring Intelligence Efficiency of Local AI
- **Authors:** Jon Saad-Falcon, Avanika Narayan, Hakki Orhun Akengin et al.
- **Year:** 2025 | **Venue:** arXiv (cs.AI)
- **Summary:** IPW metric for model capability vs. power consumption. Local accelerators outperform cloud in energy efficiency for 88.7% of real-world queries.

### 5. An Evaluation of LLMs Inference on Popular Single-board Computers
- **Authors:** Tung Nguyen, Tuyen Nguyen
- **Year:** 2025 | **Venue:** arXiv (cs.DC)
- **Summary:** Benchmarks 25 quantized models on RPi and Orange Pi. Llamafile outperforms Ollama in throughput and power efficiency for ≤1.5B models.

### 6. Gemma 3 Technical Report
- **Authors:** Google Gemma Team
- **Year:** 2025 | **Venue:** arXiv (cs.LG)
- **Summary:** Interleaved local/global attention for 5x KV-cache reduction. 128K context on 8GB RAM.

### 7. Phi-4-Mini Technical Report (Mixture-of-LoRAs)
- **Authors:** Microsoft
- **Year:** 2025 | **Venue:** arXiv (cs.CL)
- **Summary:** 3.8B multimodal model via Mixture-of-LoRAs. Performance parity with 2x larger models.

### 8. Qwen3-Coder-Next Technical Report
- **Authors:** Alibaba Qwen Team
- **Year:** 2026 | **Venue:** arXiv (cs.CL)
- **Summary:** 80B total / 3B active MoE for coding agents. Pareto frontier of coding capability and compute cost.

### 9. Mind the Gap: A Practical Attack on GGUF Quantization
- **Authors:** Kazuki Egashira, Robin Staab, Mark Vero et al.
- **Year:** 2025 | **Venue:** ICML
- **Summary:** Backdoor injection in GGUF quantized models. 88.7% attack success rate while base model appears benign.

### 10. Which Quantization Should I Use? Unified Evaluation of llama.cpp Quantization
- **Authors:** Uygar Kurt
- **Year:** 2026 | **Venue:** arXiv (cs.LG)
- **Summary:** Comprehensive guide to GGUF formats. FP8 identified as most robust for instruction-following.

### 11. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases
- **Authors:** Zechun Liu, Changsheng Zhao et al.
- **Year:** 2024 | **Venue:** ICML
- **Summary:** "Deep and thin" design principle for sub-billion models. 125M/350M outperform larger competitors.

### 12. SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models
- **Authors:** Muyan Li et al.
- **Year:** 2024/2025 | **Venue:** arXiv/ICLR
- **Summary:** Low-rank branch absorbs quantization outliers. 3.5x memory reduction for 12B FLUX.1.

### 13. SpecQuant: Extreme LLM Compression from a Fourier Frequency Domain Perspective
- **Year:** 2025/2026 | **Venue:** arXiv (cs.CL)
- **Summary:** Fourier-domain approach for ultra-low-bit quantization. Channel-wise spectral truncation preserves contextual cues.

### 14. OpenClaw-RL: Train Any Agent Simply by Talking
- **Authors:** Yinjie Wang, Xuyang Chen et al.
- **Year:** 2026 | **Venue:** arXiv (cs.LG)
- **Summary:** Online RL for edge agents using live next-state signals. Asynchronous training without interrupting inference.

### 15. Systems-Level Attack Surface of Edge Agent Deployments on IoT
- **Authors:** Zhonghao Zhan, Krinos Li et al.
- **Year:** 2026 | **Venue:** arXiv (cs.NI)
- **Summary:** Security risks of LLM agents on IoT. MQTT vulnerabilities and coordination-state divergence.

### 16. Cognitive Edge Computing Survey
- **Year:** 2025 | **Venue:** arXiv
- **Link:** https://arxiv.org/pdf/2501.03265

### 17. PalmBench: Comprehensive Benchmark
- **Year:** 2025 | **Venue:** ICLR
- **Link:** https://proceedings.iclr.cc/paper_files/paper/2025/file/a647405740b28a61311ac9cff28772e5-Paper-Conference.pdf

### 18. TinyLLM: Evaluation and Optimization of SLMs for Agentic Tasks on Edge Devices
- **Year:** 2025 | **Venue:** arXiv
- **Link:** https://arxiv.org/abs/2511.22138

### 19. Sustainable LLM Inference for Edge AI
- **Year:** 2025 | **Venue:** ResearchGate
- **Link:** https://www.researchgate.net/publication/390545072
