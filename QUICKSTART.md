# Quick Start Guide — Agent Failure Atlas

Get the full project running in 5 steps. All inference runs via **Groq** — no local GPU required.

---

## Step 1: Install Python Dependencies

```bash
pip install -r experiments/requirements.txt
pip install truststore   # Windows only — fixes SSL on corporate networks
```

---

## Step 2: Configure Groq API Keys

Create 6 free API keys at [https://console.groq.com](https://console.groq.com) — one per model.

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

Verify all keys are loaded:
```bash
python src/models.py
```

---

## Step 3: Run the Benchmark

```bash
# Dry run — tests the full pipeline without making any API calls
python experiments/run_benchmark.py --dry-run

# Full benchmark — 6 models × 15 tasks/type × 5 types = 450 trajectories
python experiments/run_benchmark.py --tasks-per-type 15
```

Progress is logged to the console and saved incrementally per model to:
```
experiments/results/raw/<ModelName>/trajectories.jsonl
```

---

## Step 4: Compute Metrics

```bash
python experiments/evaluate.py --results-dir experiments/results/raw/
```

Outputs:
- `experiments/results/metrics/<Model>_metrics.json`
- `experiments/results/metrics/summary.csv`

---

## Step 5: Run the Analysis

```bash
# Cross-model comparison + statistical tests
python analysis/cross_model_comparison.py

# Generate all publication-ready figures
python analysis/visualizations.py

```

Figures saved to `analysis/results/figures/`.

---

## Hardware Requirements

| What you want | Minimum |
|---|---|
| Dataset analysis only (no models) | Any Python 3.9+ machine |
| Run the full benchmark | Internet connection + 6 Groq API keys (free) |

No GPU, no local model downloads required.

---

## Troubleshooting

**SSL error on Windows:**
```bash
pip install truststore
```

**Key missing / MISSING in model check:**
```bash
python src/models.py   # shows which keys are configured
```

**Rate limit (429) errors:**
Each model uses its own key so limits are independent. The runner retries automatically with exponential backoff. Add `--delay 5.0` for extra headroom.

**Resume a partially completed run:**
```bash
# Already-completed models are skipped automatically.
# Force re-run a specific model:
python experiments/run_benchmark.py --model GPT-OSS-120B --tasks-per-type 15 --no-skip-completed
```

For full details see [`experiments/README.md`](experiments/README.md).
