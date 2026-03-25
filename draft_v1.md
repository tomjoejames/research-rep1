# Deployability of LLM Agents on Resource-Constrained Systems

**Team:** Sahana, Fazil, Adwaith

## Introduction

Recent advancements in Large Language Models (LLMs) have enabled the development of intelligent agents capable of handling complex tasks such as automation, decision-making, and natural language interaction. However, most of these systems rely heavily on cloud infrastructure, which introduces challenges such as high cost, latency, and dependency on internet connectivity.

This study focuses on evaluating whether LLM-based agents can be effectively deployed on resource-constrained environments, such as local machines or lightweight Linux systems. The goal is to understand the trade-offs between performance, efficiency, and practicality when moving from cloud-based execution to local deployment.

## Problem Statement

While LLM-powered systems are powerful, their deployment typically requires:
- High computational resources
- Continuous cloud access
- Significant operational cost

This limits their usability in:
- Offline environments
- Cost-sensitive applications
- Edge devices (e.g., Raspberry Pi, local servers)

The key problem addressed in this research is:
**Can LLM-based agents perform reliably and efficiently when deployed on lightweight, resource-constrained systems?**

## Proposed Approach

To address this, we design a simple experimental setup where an LLM-based agent is deployed and tested in a local environment.

The approach includes:
- Selecting a small, efficient language model suitable for local execution
- Running the model using a local inference framework (e.g., Ollama or equivalent)
- Implementing a basic agent workflow (e.g., email response or query handling)
- Evaluating system performance under constrained conditions

This allows us to simulate real-world deployment scenarios where resources are limited.

## System Setup

The system consists of the following components:
- **Model:** A lightweight open-source LLM (e.g., Mistral / Phi)
- **Runtime:** Local execution environment (e.g., Ollama)
- **Hardware:** Standard local machine (simulating constrained setup)
- **Agent Task:** Simple workflow, such as:
  - Email response generation
  - Query classification

The agent follows a basic pipeline:
**Input > LLM Processing > Generated Output**

## Experimental Design

To evaluate deployability, we conduct a series of tests based on real-world usage scenarios.

**Metrics Measured:**
- Memory Usage (RAM): Resource consumption during execution
- Response Time: Time taken to generate output
- Output Quality: Relevance and correctness of responses

**Test Procedure:**
- Run multiple input scenarios (5–10 test cases)
- Record performance metrics for each run
- Compare results with a cloud-based model (optional baseline)

## Expected Observations (Early Stage)

At this stage, we aim to observe:
- Whether local models can handle basic automation tasks
- The trade-off between:
  - speed vs cost
  - quality vs efficiency
- Limitations of running LLMs in constrained environments

## Conclusion (Preliminary)

This study explores the feasibility of deploying LLM-based agents outside of cloud environments. By evaluating performance on local systems, we aim to understand how lightweight deployments can support real-world automation use cases while reducing cost and dependency on external infrastructure.
