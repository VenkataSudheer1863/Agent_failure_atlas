"""
benchmark_analysis.py

Analyze actual benchmark results from Groq execution.
Reads trajectory JSONL files from experiments/results/raw/
and produces failure taxonomy analysis, model comparison tables, and figures.

Usage:
    python analysis/benchmark_analysis.py
    python analysis/benchmark_analysis.py --results-dir experiments/results/raw/
    python analysis/benchmark_analysis.py --figures-dir analysis/results/figures/
"""

import json
import csv
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARNING] pandas not installed: pip install pandas numpy")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
    })
except ImportError:
    HAS_PLOTTING = False
    print("[WARNING] matplotlib/seaborn not installed: pip install matplotlib seaborn")

CATEGORIES = ["PLAN", "REAS", "TOOL", "MEM", "EXEC", "COOR", "SAFE", "ALIG"]
CATEGORY_LABELS = {
    "PLAN": "Planning", "REAS": "Reasoning", "TOOL": "Tool Use",
    "MEM": "Memory", "EXEC": "Execution", "COOR": "Coordination",
    "SAFE": "Safety", "ALIG": "Alignment",
}
CATEGORY_COLORS = {
    "PLAN": "#4C72B0", "REAS": "#DD8452", "TOOL": "#55A868",
    "MEM": "#C44E52", "EXEC": "#8172B2", "COOR": "#937860",
    "SAFE": "#DA8BC3", "ALIG": "#8C8C8C",
}
TIER_ORDER = ["small", "medium", "reasoning", "large", "frontier-20B", "frontier-120B"]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_benchmark_results(results_dir: str) -> Dict[str, List[Dict]]:
    """Load all trajectory JSONL files from results_dir."""
    results_path = Path(results_dir)
    model_data = {}
    for model_dir in sorted(results_path.iterdir()):
        if model_dir.is_dir():
            traj_file = model_dir / "trajectories.jsonl"
            if traj_file.exists():
                records = []
                with open(traj_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
                if records:
                    model_data[model_dir.name] = records
                    print(f"  Loaded {len(records)} trajectories: {model_dir.name}")
    return model_data


def load_all_records(model_data: Dict[str, List[Dict]]) -> List[Dict]:
    """Flatten model_data into a single record list."""
    all_records = []
    for model_name, records in model_data.items():
        for r in records:
            r.setdefault("model", model_name)
            all_records.append(r)
    return all_records


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_model_metrics(records: List[Dict]) -> Dict:
    total = len(records)
    if total == 0:
        return {}
    failures = [r for r in records if r.get("outcome") == "failure"]
    partials = [r for r in records if r.get("outcome") == "partial"]
    successes = [r for r in records if r.get("outcome") == "success"]
    recovered = [r for r in records if r.get("recovered") is True]
    severities = [r["severity_score"] for r in records if r.get("severity_score") is not None]
    latencies = [r.get("elapsed_seconds", 0) for r in records if r.get("elapsed_seconds")]
    cat_counts = Counter(r.get("failure_label") for r in records if r.get("failure_label"))
    sub_counts = Counter(r.get("failure_subcategory") for r in records if r.get("failure_subcategory"))
    task_type_groups = defaultdict(list)
    for r in records:
        task_type_groups[r.get("task_type", "unknown")].append(r)
    per_task_type = {
        tt: round(sum(1 for r in recs if r.get("outcome") == "failure") / len(recs), 4)
        for tt, recs in task_type_groups.items()
    }
    traj_lens = [len(r.get("trajectory", [])) for r in failures]
    tier = records[0].get("tier", "unknown") if records else "unknown"
    return {
        "model": records[0].get("model", "unknown") if records else "unknown",
        "tier": tier,
        "total": total,
        "n_failures": len(failures),
        "n_partials": len(partials),
        "n_successes": len(successes),
        "failure_rate": round(len(failures) / total, 4),
        "partial_rate": round(len(partials) / total, 4),
        "success_rate": round(len(successes) / total, 4),
        "recovery_rate": round(len(recovered) / max(len(failures) + len(partials), 1), 4),
        "mean_severity": round(sum(severities) / max(len(severities), 1), 3),
        "mean_latency": round(sum(latencies) / max(len(latencies), 1), 2),
        "failure_density": round(sum(traj_lens) / max(len(traj_lens), 1), 2),
        "category_frequency": dict(cat_counts),
        "subcategory_frequency": dict(sub_counts.most_common(10)),
        "per_task_type_failure_rate": per_task_type,
    }


# ── Tables ────────────────────────────────────────────────────────────────────

def print_model_comparison_table(model_metrics: Dict[str, Dict]) -> None:
    print("\n" + "="*95)
    print("AGENT FAILURE ATLAS — MODEL COMPARISON TABLE (Real Groq Benchmark)")
    print("="*95)
    header = (
        f"{'Model':<22} {'Tier':<14} {'N':>5} {'Fail%':>7} {'Part%':>7} "
        f"{'Succ%':>7} {'Rec%':>7} {'Sev':>6} {'Lat(s)':>8}"
    )
    print(header)
    print("-"*95)

    # Sort by tier order
    def sort_key(item):
        tier = item[1].get("tier", "")
        try:
            return TIER_ORDER.index(tier)
        except ValueError:
            return 99

    for model, m in sorted(model_metrics.items(), key=sort_key):
        print(
            f"{model:<22} {m.get('tier', '?'):<14} "
            f"{m.get('total', 0):>5} "
            f"{m.get('failure_rate', 0)*100:>6.1f}% "
            f"{m.get('partial_rate', 0)*100:>6.1f}% "
            f"{m.get('success_rate', 0)*100:>6.1f}% "
            f"{m.get('recovery_rate', 0)*100:>6.1f}% "
            f"{m.get('mean_severity', 0):>6.2f} "
            f"{m.get('mean_latency', 0):>8.1f}"
        )
    print("="*95)


def print_failure_category_table(model_metrics: Dict[str, Dict]) -> None:
    print("\n" + "="*100)
    print("FAILURE CATEGORY DISTRIBUTION PER MODEL (% of total records)")
    print("="*100)
    header = f"{'Model':<22}" + "".join(f" {c:>7}" for c in CATEGORIES)
    print(header)
    print("-"*100)
    for model, m in sorted(model_metrics.items()):
        n = m.get("total", 1)
        cats = "".join(
            f" {m['category_frequency'].get(c, 0)/n*100:>6.1f}%"
            for c in CATEGORIES
        )
        print(f"{model:<22}{cats}")
    print("="*100)


def save_summary_csv(model_metrics: Dict[str, Dict], output_path: str) -> None:
    fields = [
        "model", "tier", "total", "n_failures", "n_partials", "n_successes",
        "failure_rate", "partial_rate", "success_rate", "recovery_rate",
        "mean_severity", "mean_latency", "failure_density",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model, m in sorted(model_metrics.items()):
            row = {"model": model}
            row.update({k: m.get(k, "") for k in fields[1:]})
            writer.writerow(row)
    print(f"Saved: {output_path}")


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_failure_rates(model_metrics: Dict[str, Dict], output_path: str) -> None:
    if not HAS_PLOTTING:
        return
    models = sorted(model_metrics.keys(), key=lambda m: model_metrics[m].get("failure_rate", 0))
    rates = [model_metrics[m].get("failure_rate", 0) * 100 for m in models]
    tiers = [model_metrics[m].get("tier", "") for m in models]
    colors = {
        "small": "#C44E52", "medium": "#DD8452", "reasoning": "#4C72B0",
        "large": "#55A868", "frontier-20B": "#8172B2", "frontier-120B": "#2d7d46",
    }
    bar_colors = [colors.get(t, "#aaa") for t in tiers]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, rates, color=bar_colors, edgecolor="white", height=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.set_xlabel("Failure Rate (%)")
    ax.set_title("Agent Failure Rate by Model — AFA Benchmark (Groq)")
    ax.set_xlim(0, max(rates) * 1.2 if rates else 100)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_category_heatmap(model_metrics: Dict[str, Dict], output_path: str) -> None:
    if not HAS_PLOTTING or not HAS_PANDAS:
        return
    models = sorted(model_metrics.keys())
    data = {}
    for model in models:
        m = model_metrics[model]
        n = max(m.get("total", 1), 1)
        data[model] = {
            CATEGORY_LABELS.get(c, c): m["category_frequency"].get(c, 0) / n * 100
            for c in CATEGORIES
        }
    df = pd.DataFrame(data).T
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, max(4, len(models) * 0.7)))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "% of records"})
    ax.set_title("Failure Category Distribution by Model (% of total records)")
    ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_task_type_failure_rates(model_metrics: Dict[str, Dict], output_path: str) -> None:
    if not HAS_PLOTTING or not HAS_PANDAS:
        return
    task_types = ["information_seeking", "tool_use", "planning", "reasoning", "multi_agent"]
    task_labels = {
        "information_seeking": "Info", "tool_use": "Tool", "planning": "Plan",
        "reasoning": "Reason", "multi_agent": "Multi-Ag",
    }
    models = sorted(model_metrics.keys())
    data = []
    for model in models:
        m = model_metrics[model]
        row = {task_labels[tt]: m["per_task_type_failure_rate"].get(tt, 0) * 100
               for tt in task_types if tt in task_labels}
        row["Model"] = model
        data.append(row)

    df = pd.DataFrame(data).set_index("Model")
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, max(4, len(models) * 0.7)))
    sns.heatmap(df, annot=True, fmt=".0f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=0, vmax=100,
                cbar_kws={"label": "Failure rate %"})
    ax.set_title("Per-Task-Type Failure Rate by Model (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_severity_distribution(model_metrics: Dict[str, Dict], output_path: str) -> None:
    if not HAS_PLOTTING:
        return
    models = sorted(model_metrics.keys(), key=lambda m: model_metrics[m].get("mean_severity", 0))
    sevs = [model_metrics[m].get("mean_severity", 0) for m in models]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(models, sevs, color="#C44E52", edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=10)
    ax.set_ylabel("Mean Severity Score (1-5)")
    ax.set_title("Mean Failure Severity by Model")
    ax.set_ylim(0, 5.5)
    ax.axhline(y=3.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_comparison(model_metrics: Dict[str, Dict], output_path: str) -> None:
    if not HAS_PLOTTING:
        return
    models = sorted(model_metrics.keys(), key=lambda m: model_metrics[m].get("mean_latency", 0))
    lats = [model_metrics[m].get("mean_latency", 0) for m in models]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(models, lats, color="#4C72B0", edgecolor="white", height=0.6)
    ax.bar_label(bars, fmt="%.1fs", padding=3, fontsize=10)
    ax.set_xlabel("Mean Response Latency (seconds)")
    ax.set_title("Mean Response Latency by Model — Groq LPU")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── Overall Taxonomy Summary ──────────────────────────────────────────────────

def taxonomy_analysis(all_records: List[Dict]) -> Dict:
    """Compute overall failure taxonomy statistics across all models."""
    total = len(all_records)
    failures = [r for r in all_records if r.get("outcome") == "failure"]
    n_failures = len(failures)

    cat_counts = Counter(r.get("failure_label") for r in failures if r.get("failure_label"))
    sub_counts = Counter(r.get("failure_subcategory") for r in failures if r.get("failure_subcategory"))
    task_type_counts = Counter(r.get("task_type") for r in failures if r.get("task_type"))

    return {
        "total_records": total,
        "total_failures": n_failures,
        "overall_failure_rate": round(n_failures / max(total, 1), 4),
        "category_counts": dict(cat_counts),
        "category_rates": {
            c: round(cat_counts.get(c, 0) / max(n_failures, 1), 4)
            for c in CATEGORIES
        },
        "top_subcategories": dict(sub_counts.most_common(10)),
        "failures_by_task_type": dict(task_type_counts),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AFA Benchmark Results Analysis")
    parser.add_argument("--results-dir", default="experiments/results/raw/")
    parser.add_argument("--output-dir", default="analysis/results/")
    parser.add_argument("--figures-dir", default="analysis/results/figures/")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    print(f"\nLoading benchmark results from: {args.results_dir}")
    model_data = load_benchmark_results(args.results_dir)
    if not model_data:
        print(f"No results found in {args.results_dir}")
        print("Run benchmark first: python experiments/run_benchmark.py --pilot")
        return

    all_records = load_all_records(model_data)
    print(f"Total records: {len(all_records)} across {len(model_data)} models")

    # Compute per-model metrics
    model_metrics = {}
    for model_name, records in model_data.items():
        m = compute_model_metrics(records)
        if m:
            model_metrics[model_name] = m

    # Print tables
    print_model_comparison_table(model_metrics)
    print_failure_category_table(model_metrics)

    # Taxonomy analysis
    tax = taxonomy_analysis(all_records)
    print(f"\nOverall failure rate: {tax['overall_failure_rate']:.1%}")
    print(f"Top failure categories: {dict(list(tax['category_rates'].items())[:4])}")
    print(f"Most common subcategories: {dict(list(tax['top_subcategories'].items())[:5])}")

    # Save CSV
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_summary_csv(model_metrics, str(out / "benchmark_summary.csv"))

    # Save taxonomy JSON
    tax_path = out / "taxonomy_analysis.json"
    with open(tax_path, "w", encoding="utf-8") as f:
        json.dump(tax, f, indent=2)
    print(f"Saved: {tax_path}")

    # Save per-model metrics JSON
    metrics_path = out / "model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(model_metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    # Figures
    if not args.no_figures and HAS_PLOTTING:
        figs = Path(args.figures_dir)
        figs.mkdir(parents=True, exist_ok=True)
        plot_failure_rates(model_metrics, str(figs / "benchmark_failure_rates.png"))
        plot_category_heatmap(model_metrics, str(figs / "benchmark_category_heatmap.png"))
        plot_task_type_failure_rates(model_metrics, str(figs / "benchmark_task_type_heatmap.png"))
        plot_severity_distribution(model_metrics, str(figs / "benchmark_severity.png"))
        plot_latency_comparison(model_metrics, str(figs / "benchmark_latency.png"))
        print(f"\nFigures saved to: {figs}")

    print("\nAnalysis complete.")
    return model_metrics, tax


if __name__ == "__main__":
    main()
