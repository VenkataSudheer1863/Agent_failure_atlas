# Experiments

This directory contains all code to **run the benchmark**, collect agent trajectories, and evaluate failure rates across models.

All models run via **[Groq](https://console.groq.com)** — no local GPU or model downloads required.

## Directory Structure

```
experiments/
  README.md                      # This file
  requirements.txt               # Python dependencies
  run_benchmark.py               # Main benchmark runner
  collect_trajectories.py        # Trajectory collection from live model runs
  evaluate.py                    # Compute per-model metrics from collected trajectories
  configs/
    benchmark_config.yaml        # Benchmark task list and settings
    model_configs/
      llama_3_1_8b.yaml          # Llama-3.1-8B config
      llama_4_scout_17b.yaml     # Llama-4-Scout-17B config
      qwen3_32b.yaml             # Qwen3-32B config
      llama_3_3_70b.yaml         # Llama-3.3-70B config
      gpt_oss_20b.yaml           # GPT-OSS-20B config
      gpt_oss_120b.yaml          # GPT-OSS-120B config
  tasks/
    information_seeking.jsonl    # 50 tasks (15 used per benchmark run)
    tool_use.jsonl               # 50 tasks (15 used per benchmark run)
    planning.jsonl               # 50 tasks (15 used per benchmark run)
    reasoning.jsonl              # 50 tasks (15 used per benchmark run)
    multi_agent.jsonl            # 50 tasks (15 used per benchmark run)
  results/
    raw/                         # Raw trajectory JSONL output per model
    metrics/                     # Per-model metric JSON and summary CSV
```

## Setup

### 1. Install Dependencies

```bash
pip install -r experiments/requirements.txt
pip install truststore   # Windows SSL fix for corporate networks
```

### 2. Configure Groq API Keys

Each model uses a dedicated API key for independent rate-limit quota:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY_1=gsk_...   # Llama-3.1-8B
GROQ_API_KEY_2=gsk_...   # Llama-4-Scout-17B
GROQ_API_KEY_3=gsk_...   # Qwen3-32B
GROQ_API_KEY_4=gsk_...   # Llama-3.3-70B
GROQ_API_KEY_5=gsk_...   # GPT-OSS-20B
GROQ_API_KEY_6=gsk_...   # GPT-OSS-120B
```

Create 6 free keys at [https://console.groq.com](https://console.groq.com).

### 3. Verify Setup

```bash
python src/models.py   # shows key status for all 6 models
```

## Running the Benchmark

```bash
# Dry run — no API calls, test pipeline only
python experiments/run_benchmark.py --dry-run

# Full benchmark — 6 models × 15 tasks/type × 5 types = 450 trajectories
python experiments/run_benchmark.py --tasks-per-type 15

# Single model
python experiments/run_benchmark.py --model GPT-OSS-20B --tasks-per-type 15

# Specific task types only
python experiments/run_benchmark.py --model Llama-3.3-70B --tasks planning reasoning --tasks-per-type 15

# Re-run a model (override skip-completed)
python experiments/run_benchmark.py --model Llama-3.1-8B --tasks-per-type 15 --no-skip-completed

# Evaluate results
python experiments/evaluate.py --results-dir experiments/results/raw/
```

## Output

Results are saved to:
- `experiments/results/raw/<ModelName>/trajectories.jsonl` — raw trajectories per model
- `experiments/results/metrics/<ModelName>_metrics.json` — per-model metrics
- `experiments/results/metrics/summary.csv` — cross-model comparison

## Notes on Reproducibility

- All experiments use `temperature=0.0` for deterministic outputs.
- Random seed is fixed at 42 for all components.
- Already-completed models are skipped automatically on resume.
