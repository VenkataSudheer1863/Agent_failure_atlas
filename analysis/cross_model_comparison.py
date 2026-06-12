"""
cross_model_comparison.py

Cross-model failure analysis for the Agent Failure Atlas.
Loads AFAD records, computes per-model metrics, runs statistical tests,
and generates comparison tables.

Usage:
    python analysis/cross_model_comparison.py
    python analysis/cross_model_comparison.py --dataset dataset/afad_v1.jsonl
"""

import json
import argparse
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] numpy/pandas/scipy not installed. Run: pip install numpy pandas scipy")


MODELS = [
    "GPT-OSS-20B",
    "Qwen3-8B",
    "Qwen3-30B",
    "DeepSeek-R1-8B",
    "Gemma3-12B",
    "Llama-3.2",
]

CATEGORIES = ["PLAN", "REAS", "TOOL", "MEM", "EXEC", "COOR", "SAFE", "ALIG"]


# ── Load data ─────────────────────────────────────────────────────────────────

def load_afad(filepath: str) -> List[Dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Per-model metrics ─────────────────────────────────────────────────────────

def compute_per_model_metrics(records: List[Dict]) -> Dict[str, Dict]:
    """Compute metrics for each model separately."""
    model_groups = defaultdict(list)
    for r in records:
        model_groups[r.get("model", "unknown")].append(r)

    metrics = {}
    for model, recs in model_groups.items():
        n = len(recs)
        failures = [r for r in recs if r.get("outcome") == "failure"]
        recovered = [r for r in recs if r.get("recovered") is True]
        severity_scores = [r["severity_score"] for r in recs if "severity_score" in r]

        cat_counts = Counter(r.get("failure_label") for r in recs)
        task_type_failures = defaultdict(lambda: {"total": 0, "failed": 0})
        for r in recs:
            tt = r.get("task_type", "unknown")
            task_type_failures[tt]["total"] += 1
            if r.get("outcome") == "failure":
                task_type_failures[tt]["failed"] += 1

        per_tt_rate = {
            tt: d["failed"] / d["total"] if d["total"] > 0 else 0
            for tt, d in task_type_failures.items()
        }

        metrics[model] = {
            "n": n,
            "failure_rate": len(failures) / n if n > 0 else 0,
            "recovery_rate": len(recovered) / max(len(failures), 1),
            "mean_severity": sum(severity_scores) / max(len(severity_scores), 1),
            "category_distribution": {cat: cat_counts.get(cat, 0) / n for cat in CATEGORIES},
            "per_task_type_failure_rate": per_tt_rate,
            "n_failures": len(failures),
            "n_recovered": len(recovered),
        }
    return metrics


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(model_metrics: Dict[str, Dict]) -> None:
    print("\n" + "="*90)
    print("CROSS-MODEL FAILURE COMPARISON")
    print("="*90)
    header = f"{'Model':<20} {'N':>5} {'Fail%':>7} {'Rec%':>7} {'AvgSev':>8} | " + " ".join(f"{c:>5}" for c in CATEGORIES)
    print(header)
    print("-" * 90)
    for model in MODELS:
        if model not in model_metrics:
            continue
        m = model_metrics[model]
        cats = " ".join(f"{m['category_distribution'].get(c, 0)*100:>5.1f}" for c in CATEGORIES)
        print(
            f"{model:<20} {m['n']:>5} {m['failure_rate']*100:>6.1f}% "
            f"{m['recovery_rate']*100:>6.1f}% {m['mean_severity']:>8.2f} | {cats}"
        )
    print("="*90)
    print(f"Category columns show % of that model's records in each failure category")


# ── Statistical tests ─────────────────────────────────────────────────────────

def chi_square_failure_rates(model_metrics: Dict[str, Dict]) -> None:
    """Chi-square test: are failure rates significantly different across models?"""
    if not HAS_SCIPY:
        print("[SKIP] scipy not installed; skipping chi-square test")
        return

    print("\n--- Chi-Square Test: Failure Rate Across Models ---")
    observed = []
    for model in MODELS:
        if model not in model_metrics:
            continue
        m = model_metrics[model]
        observed.append([m["n_failures"], m["n"] - m["n_failures"]])

    if len(observed) < 2:
        print("Not enough models for chi-square test")
        return

    chi2, p, dof, expected = stats.chi2_contingency(observed)
    print(f"Chi² = {chi2:.4f}, df = {dof}, p = {p:.6f}")
    if p < 0.05:
        print("Result: Significant difference in failure rates across models (p < 0.05)")
    else:
        print("Result: No significant difference in failure rates across models")


def kruskal_wallis_severity(records: List[Dict]) -> None:
    """Kruskal-Wallis test: are severity distributions different across models?"""
    if not HAS_SCIPY:
        print("[SKIP] scipy not installed; skipping Kruskal-Wallis test")
        return

    print("\n--- Kruskal-Wallis Test: Severity Score Distribution Across Models ---")
    model_severities = defaultdict(list)
    for r in records:
        model = r.get("model")
        severity = r.get("severity_score")
        if model and severity is not None:
            model_severities[model].append(severity)

    groups = [model_severities[m] for m in MODELS if m in model_severities and len(model_severities[m]) > 1]
    if len(groups) < 2:
        print("Not enough data for Kruskal-Wallis test")
        return

    stat, p = stats.kruskal(*groups)
    print(f"H = {stat:.4f}, p = {p:.6f}")
    if p < 0.05:
        print("Result: Significant difference in severity distributions across models (p < 0.05)")
    else:
        print("Result: No significant difference in severity distributions across models")


# ── Category heatmap data ─────────────────────────────────────────────────────

def category_heatmap_data(model_metrics: Dict[str, Dict]) -> None:
    """Print failure category distribution as a heatmap-ready table."""
    if not HAS_SCIPY:
        return
    print("\n--- Failure Category Distribution Heatmap Data (% per model) ---")
    print(f"{'Model':<20}", end="")
    for cat in CATEGORIES:
        print(f" {cat:>8}", end="")
    print()
    print("-" * (20 + 9 * len(CATEGORIES)))
    for model in MODELS:
        if model not in model_metrics:
            continue
        print(f"{model:<20}", end="")
        for cat in CATEGORIES:
            pct = model_metrics[model]["category_distribution"].get(cat, 0) * 100
            print(f" {pct:>8.1f}", end="")
        print()


# ── Save to CSV ───────────────────────────────────────────────────────────────

def save_results(model_metrics: Dict[str, Dict], output_dir: str = "analysis/results") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Main summary
    rows = []
    for model in MODELS:
        if model not in model_metrics:
            continue
        m = model_metrics[model]
        row = {
            "model": model,
            "n": m["n"],
            "failure_rate": round(m["failure_rate"], 4),
            "recovery_rate": round(m["recovery_rate"], 4),
            "mean_severity": round(m["mean_severity"], 3),
        }
        for cat in CATEGORIES:
            row[f"pct_{cat}"] = round(m["category_distribution"].get(cat, 0) * 100, 2)
        rows.append(row)

    try:
        import csv
        fieldnames = list(rows[0].keys())
        out_path = Path(output_dir) / "cross_model_metrics.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved cross-model metrics to {out_path}")
    except Exception as e:
        print(f"Could not save CSV: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-model failure comparison")
    parser.add_argument("--dataset", default="dataset/afad_v1.jsonl",
                        help="Path to AFAD dataset JSONL file")
    parser.add_argument("--output-dir", default="analysis/results",
                        help="Directory to save output CSVs")
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"Dataset not found: {args.dataset}")
        print("Generate it first: python dataset/generate_afad.py")
        return

    print(f"Loading dataset: {args.dataset}")
    records = load_afad(args.dataset)
    print(f"Loaded {len(records)} records")

    model_metrics = compute_per_model_metrics(records)
    print_summary_table(model_metrics)
    category_heatmap_data(model_metrics)
    chi_square_failure_rates(model_metrics)
    kruskal_wallis_severity(records)
    save_results(model_metrics, args.output_dir)


if __name__ == "__main__":
    main()
