# Data Directory

This directory contains the benchmark datasets used in the experiment.

## Datasets

| File | Samples | Task | Description |
|------|---------|------|-------------|
| `emails.json` | 100 | Classification | Emails labeled across 5 categories: inquiry, complaint, feedback, spam, urgent (20 each) |
| `extraction.json` | 50 | Information Extraction | Text paragraphs with gold-standard JSON containing: name, date, location, price, category |
| `agent_tasks.json` | 50 | Multi-Step Agent | Product lookup + calculation + formatting tasks with expected numerical answers |

## Format

All files are JSON arrays. Each entry contains:
- `id` — unique sample identifier
- `text` / `query` — input to the model
- `label` / `expected` / `expected_answer` — ground truth for evaluation

## Usage

These datasets are loaded automatically by `run_experiment.py`. No preprocessing required.
