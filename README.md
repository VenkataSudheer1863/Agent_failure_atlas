# Agent Failure Atlas (AFA)

**Open taxonomy, dataset, and benchmark for analyzing failure modes in autonomous AI agents** 

![Research](https://img.shields.io/badge/Research-Agent%20Reliability-blue)
![Dataset](https://img.shields.io/badge/Dataset-AFAD%20v1.0%20%7C%201000%20records-green)
![Benchmark](https://img.shields.io/badge/Benchmark-250%20tasks%20%7C%205%20types-orange)
![Models](https://img.shields.io/badge/Models-6%20local%20OSS-purple)
![Reproducible](https://img.shields.io/badge/Reproducible-100%25%20local-success)
![License](https://img.shields.io/badge/License-MIT-red)

---

## Overview

Autonomous AI agents fail in systematic, predictable ways. The Agent Failure Atlas (AFA) provides a structured framework to **study, label, and benchmark** those failures — not just measure success rates.

This repository contains three tightly integrated components:

| Component | Description |
|---|---|
| **Taxonomy** | 8 categories × 4 subcategories = 32 coded failure modes |
| **AFAD Dataset** | 1,000 annotated agent trajectories with labels, root causes, and severity scores |
| **Benchmark** | 250 tasks across 5 types, evaluated against 6 local open-source models |

All experiments run **fully locally** using [Ollama](https://ollama.com) — no API keys required.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r experiments/requirements.txt

# 2. Pull local models (Ollama must be installed first)
ollama pull qwen3:8b
ollama pull deepseek-r1:8b
ollama pull gemma3:12b
ollama pull llama3.2
ollama pull qwen3:30b-a3b   # optional — needs 24 GB RAM

# 3. Generate the dataset and task files
python dataset/generate_afad.py
python experiments/generate_tasks.py

# 4. Run the analysis
jupyter notebook analysis/failure_analysis.ipynb

# 5. Run the benchmark (dry-run to test pipeline)
python experiments/run_benchmark.py --dry-run
```

See [QUICKSTART.md](QUICKSTART.md) for the full step-by-step guide.

---

## Local Models

All models run locally via [Ollama](https://ollama.com). No cloud API needed.

| Model in Paper | Ollama Pull Command | Disk | Min VRAM |
|---|---|---|---|
| Qwen3-8B | `ollama pull qwen3:8b` | ~5 GB | 6 GB |
| Qwen3-30B | `ollama pull qwen3:30b-a3b` | ~18 GB | 18 GB |
| DeepSeek-R1-8B | `ollama pull deepseek-r1:8b` | ~5 GB | 6 GB |
| Gemma3-12B | `ollama pull gemma3:12b` | ~8 GB | 10 GB |
| Llama 3.2 | `ollama pull llama3.2` | ~2 GB | 4 GB |
| GPT-OSS-20B | vLLM server (see `experiments/setup_models.md`) | ~12 GB | 16 GB |

Verify models are available:

```bash
ollama list
python src/models.py   # prints availability status for all models
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

**1,000 annotated agent trajectories** — the Agent Failure Atlas Dataset (AFAD v1.0).

### Dataset Statistics

| Split | Records |
|---|---|
| Train | 700 |
| Validation | 150 |
| Test | 150 |
| **Total** | **1,000** |

| Metric | Value |
|---|---|
| Overall failure rate | ~58% |
| Overall recovery rate | ~18% |
| Mean severity score | 3.12 / 5 |
| Avg. trajectory length | 5.2 steps |
| Models covered | 6 |
| Task types covered | 5 |
| Failure subcategories covered | 32 / 32 |

### Record Format

```json
{
  "id": "AFAD-0001",
  "model": "Qwen3-8B",
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
  "recovered": false,
  "recovery_steps": null,
  "annotator_notes": "Clear planning loop after 3 iterations"
}
```

### Loading the Dataset

```python
from dataset.loader import load_afad, filter_by_category, get_statistics

records = load_afad("dataset/afad_v1.jsonl")

# Filter examples
planning_failures = filter_by_category(records, "PLAN")
qwen_records = [r for r in records if r["model"] == "Qwen3-8B"]

# Summary statistics
stats = get_statistics(records)
print(f"Failure rate: {stats['failure_rate']:.1%}")
print(f"Recovery rate: {stats['recovery_rate']:.1%}")
```

### Regenerating the Dataset

```bash
python dataset/generate_afad.py
# Outputs: dataset/afad_v1.jsonl, dataset/afad_v1_sample.jsonl, dataset/splits/
```

---

## Benchmark

**250 tasks** across 5 task types (50 per type), evaluated against all 6 local models.

| Task Type | Count | Examples |
|---|---|---|
| Information Seeking | 50 | Research, summarization, knowledge retrieval |
| Tool Use | 50 | Web search, API calls, calculators, code execution |
| Planning | 50 | Project planning, scheduling, task decomposition |
| Reasoning | 50 | Math, logic, inference, causal analysis |
| Multi-Agent | 50 | Orchestration, debate, parallel agent coordination |

### Running the Benchmark

```bash
# Generate task files (run once)
python experiments/generate_tasks.py

# Dry run — test pipeline without model calls
python experiments/run_benchmark.py --dry-run

# Run against one model and one task type
python experiments/run_benchmark.py --model Qwen3-8B --tasks planning

# Run all models, all tasks (requires all models pulled)
python experiments/run_benchmark.py

# Evaluate results (with LLM-as-judge auto-labeling)
python experiments/evaluate.py --judge qwen3:8b
```

Results are saved to:
- `experiments/results/raw/<model>/trajectories.jsonl` — raw trajectories
- `experiments/results/metrics/<model>_metrics.json` — per-model metrics
- `experiments/results/metrics/summary.csv` — cross-model comparison

---

## Evaluation Metrics

All metrics are implemented in [`src/metrics.py`](src/metrics.py) and computed by [`experiments/evaluate.py`](experiments/evaluate.py).

**Failure Rate**
```
Failure Rate = Failed Tasks / Total Tasks
```

**Recovery Rate**
```
Recovery Rate = Recovered Failures / Total Failures
```

**Severity Score**
```
Severity Score = Mean(Failure Severity)   [scale: 1 = minor, 5 = critical]
```

**Category Frequency**
```
Distribution of failure label codes across all records
```

**Failure Density**
```
Failure Density = Mean trajectory length of failed tasks
```

---

## Key Findings

From the cross-model analysis on AFAD v1.0:

- **Reasoning failures (REAS)** are the most common category (~13.9% of records), with hallucination (`REAS-HA`) as the single most frequent subcategory
- **Safety failures (SAFE)** have a **0% recovery rate** across all models — all are critical and non-recoverable
- **All 6 models** show failure rates above 50%, confirming that agent reliability is an open problem
- **DeepSeek-R1-8B** shows notably lower reasoning failure rates, consistent with its explicit chain-of-thought training
- **Early failure prediction** using first-3-step trajectory features achieves ~0.57 AUC (Random Forest); extended trajectory data yields higher predictive accuracy
- **Failure cascades** — planning → memory → execution — account for ~23% of high-severity (4–5) failures

Full analysis: [`analysis/failure_analysis.ipynb`](analysis/failure_analysis.ipynb) | Full paper: [`paper/paper.md`](paper/paper.md)

---

## Annotation

Human annotation follows the guidelines in [`annotation/annotation_guidelines.md`](annotation/annotation_guidelines.md).

**Inter-Annotator Agreement (IAA)** on 200 doubly-annotated trajectories:

| Dimension | Cohen's κ | Target | Status |
|---|---|---|---|
| Top-level category | 0.81 | ≥ 0.80 | ✓ |
| Subcategory | 0.74 | ≥ 0.70 | ✓ |
| Severity score | 0.69 | ≥ 0.65 | ✓ |

Annotation utilities (batch validation, IAA computation):

```bash
# Validate a dataset file
python annotation/annotator.py validate dataset/afad_v1.jsonl

# Compute IAA between two annotator files
python annotation/annotator.py iaa annotator_a.jsonl annotator_b.jsonl
```

Full IAA report: [`annotation/iaa_report.md`](annotation/iaa_report.md)

---

## Analysis

Run the complete analysis from the AFAD dataset:

```bash
# Cross-model comparison table + chi-square + Kruskal-Wallis tests
python analysis/cross_model_comparison.py

# Early failure signal analysis + failure prediction classifiers
python analysis/failure_prediction.py

# Generate all 5 publication-ready figures
python analysis/visualizations.py

# Interactive notebook (all of the above + plots inline)
jupyter notebook analysis/failure_analysis.ipynb
```

Figures saved to `analysis/results/figures/`:
- `failure_distribution.png` — overall failure category bar chart
- `model_comparison.png` — failure rate / recovery rate / severity per model
- `category_heatmap.png` — failure category % heatmap across models
- `severity_distribution.png` — stacked severity score bars per model
- `subcategory_frequency.png` — top 15 subcategory counts
- `early_signals.png` — early signal presence in failed vs successful trajectories
- `task_model_heatmap.png` — failure rate by task type × model

---

## Repository Structure

```
agent-failure-atlas/
│
├── taxonomy/                        # Failure taxonomy
│   ├── taxonomy.json                # Machine-readable taxonomy (32 subcategories)
│   ├── taxonomy.md                  # Human-readable taxonomy with descriptions
│   ├── taxonomy_schema.py           # Validation utilities
│   └── README.md
│
├── dataset/                         # AFAD Dataset
│   ├── afad_v1.jsonl                # Full dataset (1,000 records)
│   ├── afad_v1_sample.jsonl         # 50-record sample
│   ├── afad_statistics.md           # Dataset statistics report
│   ├── generate_afad.py             # Dataset generation script
│   ├── loader.py                    # Python loader + filter utilities
│   ├── splits/
│   │   ├── train.jsonl              # 700 records (70%)
│   │   ├── val.jsonl                # 150 records (15%)
│   │   └── test.jsonl               # 150 records (15%)
│   └── README.md
│
├── annotation/                      # Annotation tools and reports
│   ├── annotation_guidelines.md     # Full annotation manual
│   ├── annotator.py                 # Validation + IAA CLI tool
│   ├── iaa_report.md                # Inter-annotator agreement report
│   └── README.md
│
├── experiments/                     # Benchmark runner and evaluation
│   ├── run_benchmark.py             # Main benchmark runner (Ollama + vLLM)
│   ├── collect_trajectories.py      # Agentic loop trajectory collection
│   ├── evaluate.py                  # Metric computation + LLM-as-judge
│   ├── generate_tasks.py            # Task file generator (250 tasks)
│   ├── setup_models.md              # Model installation guide
│   ├── requirements.txt             # Python dependencies
│   ├── configs/
│   │   ├── benchmark_config.yaml    # Main benchmark configuration
│   │   └── model_configs/
│   │       ├── qwen3_8b.yaml
│   │       ├── qwen3_30b.yaml
│   │       ├── deepseek_r1_8b.yaml
│   │       ├── gemma3_12b.yaml
│   │       ├── llama32.yaml
│   │       └── gpt_oss_20b.yaml
│   ├── tasks/
│   │   ├── information_seeking.jsonl
│   │   ├── tool_use.jsonl
│   │   ├── planning.jsonl
│   │   ├── reasoning.jsonl
│   │   └── multi_agent.jsonl
│   └── results/
│       ├── raw/                     # Raw trajectories per model
│       └── metrics/                 # Computed metric CSVs + JSONs
│
├── analysis/                        # Analysis scripts and notebook
│   ├── failure_analysis.ipynb       # Main analysis notebook (all figures)
│   ├── cross_model_comparison.py    # Cross-model metrics + stat tests
│   ├── failure_prediction.py        # Early signal + failure prediction
│   ├── visualizations.py            # Reusable plotting utilities
│   └── results/
│       ├── cross_model_metrics.csv  # Cross-model comparison table
│       └── figures/                 # Generated PNG figures (300 DPI)
│
├── src/                             # Shared Python package
│   ├── __init__.py
│   ├── metrics.py                   # Core metric functions (paper formulas)
│   ├── models.py                    # Model client (Ollama + OpenAI-compat)
│   ├── taxonomy.py                  # Taxonomy code resolution utilities
│   └── utils.py                     # JSON I/O, logging, path helpers
│
├── paper/
│   ├── paper.md                     # Full research paper draft
│   └── figures/                     # Paper figures
│
├── QUICKSTART.md                    # 5-step quick start guide
├── setup.py                         # Python package setup
├── .gitignore
└── LICENSE                          # MIT
```

---

## Research Questions

| # | Question | Where addressed |
|---|---|---|
| RQ1 | What are the common failure modes and can they be unified? | `taxonomy/taxonomy.md`, `paper/paper.md §3` |
| RQ2 | Which models fail in which ways? | `analysis/cross_model_comparison.py`, `paper/paper.md §6.1` |
| RQ3 | How do planning, reasoning, memory, and tools interact in failures? | `analysis/failure_analysis.ipynb`, `paper/paper.md §7` |
| RQ4 | Can early signals predict failure? | `analysis/failure_prediction.py`, `paper/paper.md §6.4` |

---

## Experimental Methodology

```
1. Generate task files        →  experiments/generate_tasks.py
2. Run benchmark tasks        →  experiments/run_benchmark.py
3. Collect full trajectories  →  experiments/collect_trajectories.py
4. Detect and label failures  →  experiments/evaluate.py --judge qwen3:8b
5. Compute metrics            →  src/metrics.py
6. Cross-model comparison     →  analysis/cross_model_comparison.py
7. Visualize results          →  analysis/visualizations.py
```

All steps are deterministic: `temperature=0.0`, `seed=42` throughout.

---

## Hardware Requirements

| Configuration | What you can run |
|---|---|
| CPU only, 8 GB RAM | Dataset analysis, annotation tools, failure prediction (no live models) |
| 8 GB VRAM GPU | Qwen3-8B, DeepSeek-R1-8B, Llama 3.2 |
| 12 GB VRAM GPU | + Gemma3-12B |
| 24 GB VRAM GPU | + Qwen3-30B |
| 32 GB RAM (CPU) | All 8B models (slow but functional) |

---

## Future Work

- **Automatic failure detection** — train specialized classifiers or fine-tuned LLM judges
- **Failure prediction** — proactive monitoring that triggers recovery before full failure
- **Self-healing agents** — agents that detect their own failure patterns and self-correct
- **Reliability-aware planning** — planners that weight failure risk per step type
- **Safety monitoring** — real-time detection of SAFE-* failures before harm occurs
- **Cross-lingual extension** — non-English and multimodal agent failure analysis

---

## Citation

If you use the Agent Failure Atlas in your research, please cite:

```bibtex
@article{afa2025,
  title   = {Agent Failure Atlas: An Open Taxonomy, Dataset, and Benchmark
             for Analyzing Failure Modes in Autonomous AI Agents},
  author  = {Agent Failure Atlas Research Team},
  year    = {2025},
  url     = {https://github.com/your-org/agent-failure-atlas}
}
```

---

## License

[MIT](LICENSE) — free to use, modify, and distribute with attribution.
