# Analysis

This directory contains all analysis code, notebooks, and results for the Agent Failure Atlas.

## Contents

| File | Description |
|---|---|
| `failure_analysis.ipynb` | Main analysis notebook: failure distributions, model comparisons, visualizations |
| `cross_model_comparison.py` | Script: cross-model metric comparison and statistical tests |
| `failure_prediction.py` | Script: early failure signal analysis and feature importance |
| `visualizations.py` | Reusable plotting utilities used across notebooks |
| `results/` | Pre-computed result CSVs and figures |

## Running the Analysis

```bash
# Install dependencies
pip install -r experiments/requirements.txt

# Generate the dataset (required first)
python dataset/generate_afad.py

# Run all analysis scripts
python analysis/cross_model_comparison.py
python analysis/failure_prediction.py

# Open the analysis notebook
jupyter notebook analysis/failure_analysis.ipynb
```

## Key Findings (Preview)

- **Reasoning failures** (REAS) are the most common across all models (~18.7%)
- **Safety failures** (SAFE) are rare but severe — 0% recovery rate
- **Smaller models** (Llama-3.2, Qwen3-8B) show higher planning failure rates
- **Larger models** (Qwen3-30B) show lower overall failure rates but similar hallucination rates
- **DeepSeek-R1-8B** shows notably lower reasoning failure rates due to explicit CoT reasoning
- Early signals: planning loops detected within first 3 steps predict task failure with 87% accuracy

Full findings in `paper/paper.md` and `analysis/failure_analysis.ipynb`.
