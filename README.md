# Agent Failure Atlas (AFA)

**Open taxonomy, dataset, and benchmark for analyzing failure modes in autonomous AI agents**

![Research](https://img.shields.io/badge/Research-Agent%20Reliability-blue)
![Dataset](https://img.shields.io/badge/Dataset-AFAD%20v1.0%20%7C%20450%20trajectories-green)
![Benchmark](https://img.shields.io/badge/Benchmark-450%20trajectories%20%7C%205%20types-orange)
![Models](https://img.shields.io/badge/Models-6%20via%20Groq-purple)
![Provider](https://img.shields.io/badge/Provider-Groq%20LPU-brightgreen)
![License](https://img.shields.io/badge/License-MIT-red)

---

## Overview

Autonomous AI agents fail in systematic, predictable ways. The Agent Failure Atlas (AFA) provides a structured framework to **study, label, and benchmark** those failures — not just measure success rates.

This repository contains three tightly integrated components:

| Component | Description |
|---|---|
| **Taxonomy** | 8 categories × 4 subcategories = 32 coded failure modes |
| **AFAD Dataset** | 450 real agent trajectories from 6 models, with automatic failure labels and severity scores |
| **Benchmark** | 450 trajectories across 5 task types (75 per model, 15 per type), evaluated against 6 models via Groq |

All experiments run via **[Groq](https://console.groq.com)** — no GPU, no local model downloads required.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r experiments/requirements.txt

# 2. Copy .env.example and fill in your 6 Groq API keys
cp .env.example .env

# 3. Verify all keys are loaded
python src/models.py

# 4. Dry run — tests pipeline without API calls
python experiments/run_benchmark.py --dry-run

# 5. Run the full benchmark (6 models × 75 tasks each = 450 total)
python experiments/run_benchmark.py --tasks-per-type 15
```

See [QUICKSTART.md](QUICKSTART.md) for the full step-by-step guide.

---

## Models (via Groq)

All models run on [Groq LPU](https://console.groq.com) — no local hardware required.

| # | AFA Label | Groq Model ID | Tier | Key Env |
|---|---|---|---|---|
| 1 | Llama-3.1-8B | `llama-3.1-8b-instant` | Small | `GROQ_API_KEY_1` |
| 2 | Llama-4-Scout-17B | `meta-llama/llama-4-scout-17b-16e-instruct` | Medium | `GROQ_API_KEY_2` |
| 3 | Qwen3-32B | `qwen/qwen3-32b` | Reasoning | `GROQ_API_KEY_3` |
| 4 | Llama-3.3-70B | `llama-3.3-70b-versatile` | Large | `GROQ_API_KEY_4` |
| 5 | GPT-OSS-20B | `openai/gpt-oss-20b` | Frontier-20B | `GROQ_API_KEY_5` |
| 6 | GPT-OSS-120B | `openai/gpt-oss-120b` | Frontier-120B | `GROQ_API_KEY_6` |

Each model uses a **dedicated Groq API key** to maximise its independent rate-limit quota.  
Create 6 free keys at [https://console.groq.com](https://console.groq.com).

Verify model configuration:

```bash
python src/models.py   # prints key status for all models
```

---

## Failure Taxonomy

The AFA taxonomy has **8 top-level categories** and **32 subcategories**, each with a unique 6-character code used throughout the dataset and benchmark.

| Code | Category | Subcategories |
|---|---|---|
| `PLAN` | Planning | `PLAN-MS` Missing Steps · `PLAN-WO` Wrong Ordering · `PLAN-PL` Planning Loops · `PLAN-RP` Redundant Plans |
| `REAS` | Reasoning | `REAS-HA` Hallucination · `REAS-CO` Contradiction · `REAS-II` Invalid Inference · `REAS-UC` Unsupported Conclusions |
| `TOOL` | Tool Use | `TOOL-WT` Wrong Tool · `TOOL-PE` Parameter Errors · `TOOL-AM` API Misuse · `TOOL-PF` Parsing Failures |
| `MEM` | Memory | `MEM-CL` Context Loss · `MEM-GF` Goal Forgetting · `MEM-SC` State Corruption · `MEM-MH` Memory Hallucination |
| `EXEC` | Execution | `EXEC-IL` Infinite Loops · `EXEC-PT` Premature Termination · `EXEC-RA` Repeated Actions · `EXEC-TA` Task Abandonment |
| `COOR` | Coordination | `COOR-CB` Comm. Breakdown · `COOR-RC` Role Confusion · `COOR-DL` Deadlocks · `COOR-CF` Conflicts |
| `SAFE` | Safety | `SAFE-PI` Prompt Injection · `SAFE-UA` Unsafe Actions · `SAFE-DL` Data Leakage · `SAFE-PV` Policy Violations |
| `ALIG` | Alignment | `ALIG-GD` Goal Drift · `ALIG-RH` Reward Hacking · `ALIG-SS` Specification Gaming · `ALIG-MI` Misalignment |

Full descriptions, examples, and annotation rules: [`taxonomy/taxonomy.md`](taxonomy/taxonomy.md)  
Machine-readable version: [`taxonomy/taxonomy.json`](taxonomy/taxonomy.json)

---

## AFAD Dataset

**450 annotated agent trajectories** — the Agent Failure Atlas Dataset (AFAD v1.0).

### Dataset Statistics

| Metric | Value |
|---|---|
| Total trajectories | 450 (75 per model × 6 models) |
| Overall failure rate | 62.9% |
| Overall recovery rate | 23.7% |
| Mean severity score | 3.24 / 5 |
| Models covered | 6 |
| Task types covered | 5 (15 tasks per type per model) |
| Failure subcategories covered | 10 / 32 |

### Record Format

```json
{
  "id": "AFAD-0001",
  "model": "Llama-3.3-70B",
  "task_type": "planning",
  "task_id": "PLAN-001",
  "trajectory": [
    {
      "step": 1,
      "action": "[Step 1] Decompose the goal into subtasks",
      "observation": "Agent has reformulated the plan 5 times without any execution.",
      "tool_called": null
    }
  ],
  "failure_label": "PLAN",
  "failure_subcategory": "PLAN-PL",
  "root_cause": "Agent repeatedly reformulates plan without execution",
  "severity_score": 4,
  "outcome": "failure",
  "recovered": false
}
```

### Loading Benchmark Results

```python
import json
from pathlib import Path

records = []
for f in Path("experiments/results/raw").glob("*/trajectories.jsonl"):
    records += [json.loads(l) for l in f.open("r", encoding="utf-8-sig") if l.strip()]
print(f"Loaded {len(records)} trajectories across {len(set(r['model'] for r in records))} models")
```

---

## Benchmark

**450 trajectories** across 5 task types (15 per type per model), evaluated against all 6 models.

| Task Type | Tasks/Model | Examples |
|---|---|---|
| Information Seeking | 15 | Research, summarization, knowledge retrieval |
| Tool Use | 15 | Web search, API calls, calculators, code execution |
| Planning | 15 | Project planning, scheduling, task decomposition |
| Reasoning | 15 | Math, logic, inference, causal analysis |
| Multi-Agent | 15 | Orchestration, debate, parallel agent coordination |

### Running the Benchmark

```bash
# Dry run — test pipeline without API calls
python experiments/run_benchmark.py --dry-run

# Full benchmark — 6 models × 15 tasks/type × 5 types = 450 trajectories
python experiments/run_benchmark.py --tasks-per-type 15

# Single model
python experiments/run_benchmark.py --model GPT-OSS-20B --tasks-per-type 15

# Specific task types
python experiments/run_benchmark.py --model Llama-3.3-70B --tasks planning reasoning --tasks-per-type 15

# Re-run a model (override skip-completed)
python experiments/run_benchmark.py --model Llama-3.1-8B --tasks-per-type 15 --no-skip-completed
```

Results saved to:
- `experiments/results/raw/<model>/trajectories.jsonl` — raw trajectories per model
- `experiments/results/metrics/<model>_metrics.json` — per-model metrics
- `experiments/results/metrics/summary.csv` — cross-model comparison

### Evaluating Results

```bash
# Compute metrics from labels stored during benchmark run (no extra API calls)
python experiments/evaluate.py --results-dir experiments/results/raw/

# Re-label with Groq judge
python experiments/evaluate.py --results-dir experiments/results/raw/ --judge groq
```

---

## Evaluation Metrics

All metrics are implemented in [`src/metrics.py`](src/metrics.py) and computed by [`experiments/evaluate.py`](experiments/evaluate.py).

| Metric | Formula |
|---|---|
| Failure Rate | Failed Tasks / Total Tasks |
| Recovery Rate | Recovered Failures / Total Failures |
| Severity Score | Mean(Failure Severity) · scale 1–5 |
| Category Frequency | Distribution of failure label codes |
| Failure Density | Mean trajectory length of failed tasks |

---

## Analysis

```bash
# Cross-model comparison table + chi-square + Kruskal-Wallis tests
python analysis/cross_model_comparison.py

# Early failure signal analysis + failure prediction classifiers
python analysis/failure_prediction.py

# Generate all 7 publication-ready figures (300 DPI)
python analysis/visualizations.py
```

Figures saved to `analysis/results/figures/`.

---

## Experimental Methodology

```
1. Configure .env with 6 Groq API keys    →  .env / .env.example
2. Run benchmark tasks                    →  experiments/run_benchmark.py
3. Compute metrics + failure labels       →  experiments/evaluate.py
4. Cross-model statistical comparison     →  analysis/cross_model_comparison.py
5. Visualize results                      →  analysis/visualizations.py
```

All steps use `temperature=0.0`, `seed=42` for reproducibility.

---

## Repository Structure

```
agent-failure-atlas/
│
├── taxonomy/                        # Failure taxonomy
│   ├── taxonomy.json
│   ├── taxonomy.md
│   ├── taxonomy_schema.py
│   └── README.md
│
├── experiments/results/raw/         # Real benchmark trajectories (post-run)
│   ├── Llama-3.1-8B/trajectories.jsonl
│   ├── Llama-4-Scout-17B/trajectories.jsonl
│   └── ...                          # one folder per model
│
├── annotation/                      # Annotation tools and guidelines
│   ├── annotation_guidelines.md
│   └── annotator.py
│
├── experiments/                     # Benchmark runner and evaluation
│   ├── run_benchmark.py             # Main runner — Groq, per-model keys, skip-completed
│   ├── collect_trajectories.py      # Agentic loop with tool-use
│   ├── evaluate.py                  # Metrics + Groq judge labeling
│   ├── requirements.txt
│   ├── configs/
│   │   ├── benchmark_config.yaml    # Benchmark configuration (per-model keys, delay)
│   │   └── model_configs/           # Per-model YAML (llama_3_1_8b.yaml, qwen3_32b.yaml, etc.)
│   ├── tasks/                       # 75 task files (15 per type)
│   └── results/
│       ├── raw/                     # trajectories.jsonl per model (gitignored)
│       └── metrics/                 # JSON + summary.csv (gitignored)
│
├── analysis/                        # Analysis scripts
│   ├── cross_model_comparison.py
│   ├── failure_prediction.py
│   ├── visualizations.py
│   ├── benchmark_analysis.py
│   └── results/
│       └── figures/                 # Generated PNG figures (gitignored)
│
├── src/                             # Shared Python package
│   ├── metrics.py
│   ├── models.py                    # ModelClient — per-model Groq key support
│   ├── taxonomy.py
│   └── utils.py
│
├── paper/
│   ├── paper.md
│   └── figures/                     # Paper figures (generated by analysis scripts)
│
├── .env.example                     # Template — copy to .env, fill 6 keys
├── QUICKSTART.md
├── setup.py
├── .gitignore
└── LICENSE
```

---

## Research Questions

| # | Question | Where addressed |
|---|---|---|
| RQ1 | What are the common failure modes? | `taxonomy/taxonomy.md`, `paper/paper.md §3` |
| RQ2 | Which models fail in which ways? | `analysis/cross_model_comparison.py`, `paper/paper.md §6.1` |
| RQ3 | How do failure types interact (planning → memory → execution)? | `analysis/benchmark_analysis.py`, `paper/paper.md §7` |
| RQ4 | Can early signals predict failure? | `analysis/failure_prediction.py`, `paper/paper.md §6.4` |

---

## Annotation

All 450 trajectories are labeled automatically by a Llama-3.1-8B-instant LLM judge. The judge assigns failure category, subcategory, severity score (1–5), and outcome class using the structured prompt defined in `annotation/annotation_guidelines.md`. All records pass schema validation (see `annotation/annotator.py --validate`).

A formal human inter-annotator agreement study is planned as future work to quantify label reliability at the subcategory level.

---

## Citation

```bibtex
@article{afa2025,
  title   = {Agent Failure Atlas: An Open Taxonomy, Dataset, and Benchmark
             for Analyzing Failure Modes in Autonomous AI Agents},
  author  = {Venkata Sudheer Paruchuri},
  year    = {2026},
  url     = {https://github.com/VenkataSudheer1863/agent-failure-atlas}
}
```

---

## License

[MIT](LICENSE) — free to use, modify, and distribute with attribution.
