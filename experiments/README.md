# Experiments

This directory contains all code to **run the benchmark**, collect agent trajectories, and evaluate failure rates across models.

## Directory Structure

```
experiments/
  README.md                      # This file
  requirements.txt               # Python dependencies
  setup_models.md                # How to set up each local model
  run_benchmark.py               # Main benchmark runner
  collect_trajectories.py        # Trajectory collection from live model runs
  evaluate.py                    # Compute per-model metrics from collected trajectories
  configs/
    benchmark_config.yaml        # Benchmark task list and settings
    model_configs/
      qwen3_8b.yaml              # Qwen3-8B config
      qwen3_30b.yaml             # Qwen3-30B config
      deepseek_r1_8b.yaml        # DeepSeek-R1-8B config
      gemma3_12b.yaml            # Gemma3-12B config
      llama32.yaml               # Llama 3.2 config
      gpt_oss_20b.yaml           # GPT-OSS-20B config
  tasks/
    information_seeking.jsonl    # 50 information seeking tasks
    tool_use.jsonl               # 50 tool use tasks
    planning.jsonl               # 50 planning tasks
    reasoning.jsonl              # 50 reasoning tasks
    multi_agent.jsonl            # 50 multi-agent tasks
  results/
    raw/                         # Raw trajectory JSONL output per model
    metrics/                     # Per-model metric CSVs
```

## Local Model Setup

All models in this project run **locally** using [Ollama](https://ollama.com).

### Prerequisites

```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download installer from https://ollama.com/download
```

### Pull Models

```bash
ollama pull qwen3:8b           # Qwen3-8B  (~5GB)
ollama pull qwen3:30b-a3b      # Qwen3-30B-A3B (~18GB)
ollama pull deepseek-r1:8b     # DeepSeek-R1-8B (~5GB)
ollama pull gemma3:12b         # Gemma3-12B (~8GB)
ollama pull llama3.2           # Llama 3.2 3B (~2GB)
```

For GPT-OSS-20B, serve a compatible model via [vLLM](https://github.com/vllm-project/vllm) or use the `gpt4all` package.

### Verify Ollama is Running

```bash
ollama list
ollama run qwen3:8b "Hello, test"
```

## Running the Benchmark

```bash
# Install Python dependencies
pip install -r experiments/requirements.txt

# Run full benchmark (all models, all tasks)
python experiments/run_benchmark.py --config experiments/configs/benchmark_config.yaml

# Run for a single model
python experiments/run_benchmark.py --model qwen3:8b --tasks planning

# Evaluate results
python experiments/evaluate.py --results-dir experiments/results/raw/
```

## Output

Results are saved to:
- `experiments/results/raw/<model_name>/trajectories.jsonl` — raw trajectories
- `experiments/results/metrics/<model_name>_metrics.csv` — per-model metrics
- `experiments/results/summary.csv` — cross-model comparison

## Notes on Reproducibility

- All experiments use `temperature=0.0` for deterministic outputs.
- Each task is run once per model; increase `--runs` for variance estimation.
- Random seed is fixed at 42 for all components.
