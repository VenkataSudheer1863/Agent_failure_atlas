# Quick Start Guide — Agent Failure Atlas

Get the full project running in 5 steps.

---

## Step 1: Install Python Dependencies

```bash
pip install -r experiments/requirements.txt
```

---

## Step 2: Install Ollama and Pull Models

```bash
# Install Ollama: https://ollama.com/download

# Pull models (start with small ones; pull larger ones as needed)
ollama pull qwen3:8b          # ~5 GB
ollama pull deepseek-r1:8b    # ~5 GB
ollama pull gemma3:12b        # ~8 GB
ollama pull llama3.2          # ~2 GB
ollama pull qwen3:30b-a3b     # ~18 GB (optional, needs more VRAM)
```

---

## Step 3: Generate the AFAD Dataset

```bash
python dataset/generate_afad.py
```

This creates:
- `dataset/afad_v1.jsonl` — 1000 annotated records
- `dataset/afad_v1_sample.jsonl` — 50-record sample
- `dataset/splits/` — train/val/test splits

---

## Step 4: Generate Benchmark Task Files

```bash
python experiments/generate_tasks.py
```

This creates `experiments/tasks/*.jsonl` with 50 tasks per type.

---

## Step 5: Run the Analysis

### Option A: Run the analysis notebook
```bash
jupyter notebook analysis/failure_analysis.ipynb
```

### Option B: Run analysis scripts
```bash
python analysis/cross_model_comparison.py
python analysis/failure_prediction.py
python analysis/visualizations.py
```

Figures are saved to `analysis/results/figures/`.

---

## Step 6 (Optional): Run the Full Benchmark

```bash
# Dry run (no model calls, for testing pipeline)
python experiments/run_benchmark.py --dry-run

# Real run (requires Ollama models)
python experiments/run_benchmark.py --model Qwen3-8B --tasks planning reasoning

# Evaluate results
python experiments/evaluate.py --judge qwen3:8b
```

---

## Hardware Requirements

| What you want | Minimum Hardware |
|---|---|
| Just analysis (AFAD dataset) | Any machine with Python 3.9+ |
| Run small models (8B) | 8 GB RAM or 6 GB VRAM |
| Run medium models (12B) | 12 GB RAM or 10 GB VRAM |
| Run large models (30B) | 24 GB RAM or 18 GB VRAM |

---

## Troubleshooting

**`ollama` package not found:**
```bash
pip install ollama
```

**Ollama service not running:**
```bash
ollama serve  # Linux/macOS
# On Windows: start the Ollama desktop app
```

**Out of memory for large models:**
Use quantized variants: `ollama pull qwen3:8b-q4_0` (smaller, slightly less accurate)

**Dataset not found:**
```bash
python dataset/generate_afad.py
```

**Figures directory missing:**
```bash
mkdir -p analysis/results/figures
python analysis/visualizations.py
```
