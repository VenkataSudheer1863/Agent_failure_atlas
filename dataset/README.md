# AFAD — Agent Failure Atlas Dataset

This directory contains the **Agent Failure Atlas Dataset (AFAD)** — an annotated collection of agent trajectories, each labeled with a failure mode from the AFA taxonomy.

## Dataset Structure

```
dataset/
  afad_v1.jsonl          # Full dataset (JSONL, one record per line)
  afad_v1_sample.jsonl   # 50-record sample for quick testing
  afad_statistics.md     # Dataset statistics and distribution report
  loader.py              # Python loader utility
  splits/
    train.jsonl          # 70% training split
    val.jsonl            # 15% validation split
    test.jsonl           # 15% test split
```

## Dataset Statistics (v1.0)

| Split | Records |
|---|---|
| Train | 700 |
| Validation | 150 |
| Test | 150 |
| **Total** | **1000** |

## Failure Category Distribution

| Category | Count | % |
|---|---|---|
| Planning | 148 | 14.8% |
| Reasoning | 187 | 18.7% |
| Tool Use | 162 | 16.2% |
| Memory | 121 | 12.1% |
| Execution | 134 | 13.4% |
| Coordination | 78 | 7.8% |
| Safety | 89 | 8.9% |
| Alignment | 81 | 8.1% |

## Model Distribution

| Model | Records |
|---|---|
| GPT-OSS-20B | 167 |
| Qwen3-8B | 167 |
| Qwen3-30B | 167 |
| DeepSeek-R1-8B | 167 |
| Gemma3-12B | 166 |
| Llama-3.2 | 166 |

## Record Format

Each JSONL record has the following fields:

```json
{
  "id": "AFAD-0001",
  "model": "Qwen3-8B",
  "task_type": "planning",
  "task_id": "PLAN-001",
  "trajectory": [
    {
      "step": 1,
      "action": "Think about how to decompose the task",
      "observation": "Agent repeatedly replans without executing",
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

## Loading the Dataset

```python
from dataset.loader import load_afad

# Load full dataset
records = load_afad("dataset/afad_v1.jsonl")

# Load only failures of a specific type
planning_failures = [r for r in records if r["failure_label"] == "PLAN"]

# Load by model
qwen_records = [r for r in records if r["model"] == "Qwen3-8B"]
```

## Notes on Local Models

The following models in this dataset are run **locally** using [Ollama](https://ollama.com) or [HuggingFace Transformers](https://huggingface.co/docs/transformers):

| Model | Source | How to run locally |
|---|---|---|
| Qwen3-8B | HuggingFace `Qwen/Qwen3-8B` | `ollama pull qwen3:8b` |
| Qwen3-30B | HuggingFace `Qwen/Qwen3-30B-A3B` | `ollama pull qwen3:30b-a3b` |
| DeepSeek-R1-8B | HuggingFace `deepseek-ai/DeepSeek-R1-Distill-Qwen-8B` | `ollama pull deepseek-r1:8b` |
| Gemma3-12B | HuggingFace `google/gemma-3-12b-it` | `ollama pull gemma3:12b` |
| Llama-3.2 | Meta via HuggingFace `meta-llama/Llama-3.2-3B-Instruct` | `ollama pull llama3.2` |
| GPT-OSS-20B | OpenAI-compatible OSS model | Served via vLLM or Ollama |

See `experiments/README.md` for full setup instructions.
