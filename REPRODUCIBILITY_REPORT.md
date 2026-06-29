# Reproducibility Report

**Project:** Agent Failure Atlas (AFA)
**Updated:** 2026-06-24
**Run by:** Venkata Sudheer Paruchuri (sudheer.pv@prodapt.com)
**Benchmark status:** 6/6 models complete — 450 trajectories

---

## Environment

| Component | Version / Detail |
|---|---|
| OS | Windows 11 Enterprise 10.0.26200 |
| Python | 3.14.3 |
| openai | 2.29.0 |
| pandas | 2.3.3 |
| numpy | 2.4.3 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| python-dotenv | 1.2.2 |
| truststore | 0.10.4 |

**SSL note:** `truststore.inject_into_ssl()` is required on Windows corporate networks. This is called automatically at the top of `run_benchmark.py` and `evaluate.py`. Without it, HTTPS connections to the Groq API will fail with certificate verification errors.

**Install command:**
```bash
pip install -r experiments/requirements.txt
```

---

## Provider & API Configuration

**Provider:** Groq LPU Inference — `https://api.groq.com/openai/v1`

Six dedicated API keys are used, one per model, to avoid cross-model rate-limit interference:

| Key | Model (AFA label) | Groq Model ID |
|---|---|---|
| `GROQ_API_KEY_1` | Llama-3.1-8B | `llama-3.1-8b-instant` |
| `GROQ_API_KEY_2` | Llama-4-Scout-17B | `meta-llama/llama-4-scout-17b-16e-instruct` |
| `GROQ_API_KEY_3` | Qwen3-32B | `qwen/qwen3-32b` |
| `GROQ_API_KEY_4` | Llama-3.3-70B | `llama-3.3-70b-versatile` |
| `GROQ_API_KEY_5` | GPT-OSS-20B | `openai/gpt-oss-20b` |
| `GROQ_API_KEY_6` | GPT-OSS-120B | `openai/gpt-oss-120b` |

Keys are stored in `.env` (gitignored). Copy `.env.example` and populate:
```bash
cp .env.example .env
# Edit .env with your Groq API keys
```

**Fixed benchmark parameters:**

| Parameter | Value |
|---|---|
| `temperature` | 0.0 (deterministic) |
| `seed` | 42 |
| `max_tokens` | 2048 |
| `max_steps` | 6 (per trajectory) |

---

## Dataset

- **Structure:** 5 task types × 15 tasks = 75 tasks per model
- **Total trajectories:** 450 (6 models × 75)
- **Task types:** Information Seeking, Tool Use, Planning, Reasoning, Multi-Agent
- **Judge model:** `llama-3.1-8b-instant` (Groq, via `GROQ_API_KEY_1`)

---

## Execution Order

Run scripts in the following order. Each step depends on the outputs of the previous step.

### Step 1 — Run benchmark (one model at a time)
```bash
python experiments/run_benchmark.py --model Llama-3.1-8B --tasks-per-type 15
python experiments/run_benchmark.py --model Llama-4-Scout-17B --tasks-per-type 15
python experiments/run_benchmark.py --model Qwen3-32B --tasks-per-type 15
python experiments/run_benchmark.py --model Llama-3.3-70B --tasks-per-type 15
python experiments/run_benchmark.py --model GPT-OSS-20B --tasks-per-type 15
python experiments/run_benchmark.py --model GPT-OSS-120B --tasks-per-type 15
```
Output: `experiments/results/raw/{MODEL}/trajectories.jsonl`

### Step 2 — Evaluate
```bash
python experiments/evaluate.py
```
Output: `experiments/results/metrics/` (per-model JSON + `summary.csv`)

### Step 3 — Cross-model comparison
```bash
python analysis/cross_model_comparison.py
```
Output: `analysis/results/cross_model_metrics.csv`

### Step 4 — Failure prediction
```bash
python analysis/failure_prediction.py
```
Output: printed to console (AUC scores, feature importances)

### Step 5 — Benchmark analysis
```bash
python analysis/benchmark_analysis.py
```
Output: `analysis/results/` (CSV + JSON + 5 PNG figures)

### Step 6 — Paper figures
```bash
python paper/generate_paper_figures.py
```
Output: `paper/figures/` (9 PNG files, figs 1–9)

---

## Confirmed Results

| Model | Trajectories | Failure Rate | Date |
|---|---|---|---|
| Llama-3.1-8B | 75 | 78.7% | 2026-06-23 |
| Llama-4-Scout-17B | 75 | 72.0% | 2026-06-23 |
| Qwen3-32B | 75 | 52.0% | 2026-06-23 |
| Llama-3.3-70B | 75 | 60.0% | 2026-06-24 |
| GPT-OSS-20B | 75 | 61.3% | 2026-06-24 |
| GPT-OSS-120B | 75 | 53.3% | 2026-06-24 |

**Cross-model statistical test:** Chi² = 17.76, p = 0.003 (significant at α = 0.05)

---

## Known Reproducibility Notes

1. **Rate limiting / key rotation:** Each Groq key has a daily TPD limit (100k–200k tokens/day on free tier). For long runs, key rotation across the six dedicated keys is required. Per-task saving in `run_benchmark.py` means interrupted runs can be safely resumed — completed tasks are skipped on re-run.

2. **Qwen3-32B UTF-8 BOM (Windows):** Trajectories from Qwen3-32B may be written with a UTF-8 BOM on Windows. When reading `trajectories.jsonl` for this model, open with `encoding='utf-8-sig'` rather than `encoding='utf-8'` to avoid JSON parse errors on the first line.

3. **Logistic Regression convergence:** `failure_prediction.py` runs LogisticRegression with `max_iter=500`. On the full 450-trajectory dataset this may still not converge. Either increase `max_iter` (e.g., to 2000) or rely on the Random Forest results, which converge unconditionally.

4. **Determinism:** `temperature=0.0` is used for all model calls. However, the judge model (`llama-3.1-8b-instant`) performs LLM-based labeling, so minor stochastic variation in judge outputs is possible across runs even with `temperature=0.0`, depending on Groq's serving infrastructure. Failure rates should be stable to ±1–2 percentage points.

5. **Outcome reconciliation:** `run_benchmark.py` applies bidirectional judge reconciliation — if the judge assigns a non-null `failure_label`, the trajectory `outcome` is set to `failure` regardless of the model's self-reported outcome. This fix was applied retroactively to Llama-3.1-8B (36 of 75 trajectories corrected) and is active natively for all subsequent models.
