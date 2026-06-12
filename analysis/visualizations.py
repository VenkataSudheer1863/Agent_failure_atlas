"""
visualizations.py

Reusable plotting utilities for the Agent Failure Atlas analysis.
All plots are publication-ready (300 DPI, tight layout).

Usage:
    from analysis.visualizations import (
        plot_failure_distribution,
        plot_model_comparison,
        plot_category_heatmap,
        plot_severity_distribution,
    )
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Optional

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    HAS_PLOTTING = True
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "figure.dpi": 150,
    })
except ImportError:
    HAS_PLOTTING = False
    print("[WARNING] matplotlib/seaborn not installed. Run: pip install matplotlib seaborn")


MODELS = ["GPT-OSS-20B", "Qwen3-8B", "Qwen3-30B", "DeepSeek-R1-8B", "Gemma3-12B", "Llama-3.2"]
CATEGORIES = ["PLAN", "REAS", "TOOL", "MEM", "EXEC", "COOR", "SAFE", "ALIG"]
CATEGORY_COLORS = {
    "PLAN": "#4C72B0", "REAS": "#DD8452", "TOOL": "#55A868",
    "MEM": "#C44E52", "EXEC": "#8172B2", "COOR": "#937860",
    "SAFE": "#DA8BC3", "ALIG": "#8C8C8C",
}


def load_afad(filepath: str) -> List[Dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ensure_output_dir(path: str = "analysis/results/figures") -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Plot 1: Overall failure category distribution ────────────────────────────

def plot_failure_distribution(records: List[Dict], output_dir: str = "analysis/results/figures") -> None:
    """Bar chart: overall failure category distribution."""
    if not HAS_PLOTTING:
        return
    out = ensure_output_dir(output_dir)
    cat_counts = Counter(r.get("failure_label") for r in records if r.get("failure_label"))
    cats = [c for c in CATEGORIES if c in cat_counts]
    counts = [cat_counts[c] for c in cats]
    colors = [CATEGORY_COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cats, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, padding=3, fmt="%d")
    ax.set_title("AFAD: Failure Category Distribution (n=1000)", fontweight="bold")
    ax.set_xlabel("Failure Category")
    ax.set_ylabel("Number of Records")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    path = out / "failure_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 2: Per-model failure rate comparison ─────────────────────────────────

def plot_model_comparison(records: List[Dict], output_dir: str = "analysis/results/figures") -> None:
    """Grouped bar chart comparing failure rates, recovery rates, and mean severity per model."""
    if not HAS_PLOTTING:
        return
    out = ensure_output_dir(output_dir)

    model_data = defaultdict(lambda: {"total": 0, "failures": 0, "severity": [], "recovered": 0})
    for r in records:
        m = r.get("model", "unknown")
        model_data[m]["total"] += 1
        if r.get("outcome") == "failure":
            model_data[m]["failures"] += 1
        if r.get("severity_score"):
            model_data[m]["severity"].append(r["severity_score"])
        if r.get("recovered"):
            model_data[m]["recovered"] += 1

    model_names = [m for m in MODELS if m in model_data]
    failure_rates = [model_data[m]["failures"] / max(model_data[m]["total"], 1) * 100 for m in model_names]
    recovery_rates = [model_data[m]["recovered"] / max(model_data[m]["failures"], 1) * 100 for m in model_names]
    mean_severities = [
        sum(model_data[m]["severity"]) / max(len(model_data[m]["severity"]), 1)
        for m in model_names
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Failure Rate
    axes[0].barh(model_names, failure_rates, color="#C44E52", edgecolor="white")
    axes[0].set_title("Failure Rate (%)", fontweight="bold")
    axes[0].set_xlabel("Failure Rate (%)")
    for i, v in enumerate(failure_rates):
        axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center")

    # Recovery Rate
    axes[1].barh(model_names, recovery_rates, color="#55A868", edgecolor="white")
    axes[1].set_title("Recovery Rate (%)", fontweight="bold")
    axes[1].set_xlabel("Recovery Rate (%)")
    for i, v in enumerate(recovery_rates):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center")

    # Mean Severity
    axes[2].barh(model_names, mean_severities, color="#4C72B0", edgecolor="white")
    axes[2].set_title("Mean Severity Score", fontweight="bold")
    axes[2].set_xlabel("Mean Severity (1-5)")
    axes[2].set_xlim(0, 5)
    for i, v in enumerate(mean_severities):
        axes[2].text(v + 0.05, i, f"{v:.2f}", va="center")

    plt.suptitle("Cross-Model Failure Analysis — AFA Benchmark", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = out / "model_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 3: Failure category heatmap ─────────────────────────────────────────

def plot_category_heatmap(records: List[Dict], output_dir: str = "analysis/results/figures") -> None:
    """Heatmap: failure category distribution per model."""
    if not HAS_PLOTTING:
        return
    out = ensure_output_dir(output_dir)

    data = defaultdict(Counter)
    model_totals = Counter()
    for r in records:
        m = r.get("model", "unknown")
        cat = r.get("failure_label")
        if cat:
            data[m][cat] += 1
            model_totals[m] += 1

    model_names = [m for m in MODELS if m in data]
    matrix = []
    for m in model_names:
        total = model_totals[m]
        row = [data[m].get(c, 0) / total * 100 for c in CATEGORIES]
        matrix.append(row)

    if not HAS_PLOTTING:
        return

    import numpy as np
    df = pd.DataFrame(matrix, index=model_names, columns=CATEGORIES)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        df, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "% of model's records"}
    )
    ax.set_title("Failure Category Distribution per Model (%)", fontweight="bold", pad=12)
    ax.set_xlabel("Failure Category")
    ax.set_ylabel("Model")
    plt.tight_layout()
    path = out / "category_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 4: Severity distribution ────────────────────────────────────────────

def plot_severity_distribution(records: List[Dict], output_dir: str = "analysis/results/figures") -> None:
    """Stacked bar chart of severity score distribution per model."""
    if not HAS_PLOTTING:
        return
    out = ensure_output_dir(output_dir)

    model_severity = defaultdict(list)
    for r in records:
        m = r.get("model", "unknown")
        s = r.get("severity_score")
        if s:
            model_severity[m].append(s)

    model_names = [m for m in MODELS if m in model_severity]
    severity_colors = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c", 5: "#8e44ad"}

    fig, ax = plt.subplots(figsize=(12, 6))
    bottoms = [0] * len(model_names)
    for severity in [1, 2, 3, 4, 5]:
        counts = []
        for m in model_names:
            total = len(model_severity[m])
            count = sum(1 for s in model_severity[m] if s == severity) / total * 100
            counts.append(count)
        bars = ax.bar(model_names, counts, bottom=bottoms,
                      color=severity_colors[severity], label=f"Severity {severity}",
                      edgecolor="white", linewidth=0.5)
        bottoms = [b + c for b, c in zip(bottoms, counts)]

    ax.set_title("Severity Score Distribution per Model (%)", fontweight="bold")
    ax.set_ylabel("Percentage of Records")
    ax.set_xlabel("Model")
    ax.legend(title="Severity", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 105)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = out / "severity_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 5: Subcategory frequency ─────────────────────────────────────────────

def plot_subcategory_frequency(records: List[Dict], top_n: int = 15, output_dir: str = "analysis/results/figures") -> None:
    """Horizontal bar chart of top subcategory frequencies."""
    if not HAS_PLOTTING:
        return
    out = ensure_output_dir(output_dir)

    subcat_counts = Counter(r.get("failure_subcategory") for r in records if r.get("failure_subcategory"))
    top = subcat_counts.most_common(top_n)
    labels = [t[0] for t in top]
    values = [t[1] for t in top]
    colors = [CATEGORY_COLORS.get(l.split("-")[0], "#8C8C8C") for l in labels]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax.bar_label(bars, padding=3)
    ax.set_title(f"Top {top_n} Failure Subcategories", fontweight="bold")
    ax.set_xlabel("Count")
    plt.tight_layout()
    path = out / "subcategory_frequency.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 6: Early signals comparison (failed vs successful) ──────────────────

FAILURE_SIGNAL_KEYWORDS = {
    "loop_signal": ["again", "retry", "same", "repeated", "loop", "replan"],
    "uncertainty_signal": ["unclear", "ambiguous", "maybe", "might", "could be"],
    "error_signal": ["error", "failed", "exception", "cannot", "unable"],
    "tool_failure_signal": ["bad request", "400", "429", "401", "tool error"],
    "hallucination_signal": ["as i mentioned", "as you know", "i recall"],
    "abandon_signal": ["give up", "abandon", "impossible", "cannot complete"],
}


def plot_early_signals(records: list, output_dir: str = "analysis/results/figures") -> None:
    """Grouped bar chart: early signal presence in failed vs successful trajectories."""
    if not HAS_PLOTTING:
        return
    out = ensure_output_dir(output_dir)

    failures = [r for r in records if r.get("outcome") == "failure"]
    successes = [r for r in records if r.get("outcome") == "success"]

    def pct_with_signal(recs, keywords):
        count = 0
        for r in recs:
            traj = r.get("trajectory", [])[:3]
            text = " ".join(
                (s.get("action", "") + " " + s.get("observation", "")).lower()
                for s in traj
            )
            if any(kw in text for kw in keywords):
                count += 1
        return count / max(len(recs), 1) * 100

    signal_names = list(FAILURE_SIGNAL_KEYWORDS.keys())
    fail_pcts = [pct_with_signal(failures, FAILURE_SIGNAL_KEYWORDS[s]) for s in signal_names]
    succ_pcts = [pct_with_signal(successes, FAILURE_SIGNAL_KEYWORDS[s]) for s in signal_names]

    x = range(len(signal_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar([i - width / 2 for i in x], fail_pcts, width, label=f"Failed (n={len(failures)})", color="#C44E52")
    bars2 = ax.bar([i + width / 2 for i in x], succ_pcts, width, label=f"Success (n={len(successes)})", color="#55A868")
    ax.bar_label(bars1, fmt="%.1f%%", padding=2, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f%%", padding=2, fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels([s.replace("_signal", "").replace("_", " ") for s in signal_names], rotation=20, ha="right")
    ax.set_ylabel("% of trajectories with signal in first 3 steps")
    ax.set_title("Early Failure Signals: Failed vs Successful Trajectories", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, max(max(fail_pcts), max(succ_pcts)) * 1.3 + 5)
    plt.tight_layout()
    path = out / "early_signals.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 7: Failure rate by task type × model heatmap ────────────────────────

def plot_task_model_heatmap(records: list, output_dir: str = "analysis/results/figures") -> None:
    """Heatmap: failure rate by task type x model."""
    if not HAS_PLOTTING:
        return
    out = ensure_output_dir(output_dir)

    task_types = ["information_seeking", "tool_use", "planning", "reasoning", "multi_agent"]
    task_labels = ["Info Seeking", "Tool Use", "Planning", "Reasoning", "Multi-Agent"]

    counts: dict = {m: {tt: {"total": 0, "failed": 0} for tt in task_types} for m in MODELS}
    for r in records:
        m = r.get("model")
        tt = r.get("task_type")
        if m in counts and tt in task_types:
            counts[m][tt]["total"] += 1
            if r.get("outcome") == "failure":
                counts[m][tt]["failed"] += 1

    matrix = []
    for m in MODELS:
        row = []
        for tt in task_types:
            d = counts[m][tt]
            rate = d["failed"] / d["total"] * 100 if d["total"] > 0 else float("nan")
            row.append(rate)
        matrix.append(row)

    import numpy as np
    df = pd.DataFrame(matrix, index=MODELS, columns=task_labels)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        df, annot=True, fmt=".1f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax, vmin=0, vmax=100,
        cbar_kws={"label": "Failure Rate (%)"}
    )
    ax.set_title("Failure Rate (%) by Task Type x Model", fontweight="bold", pad=12)
    ax.set_xlabel("Task Type")
    ax.set_ylabel("Model")
    plt.tight_layout()
    path = out / "task_model_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Generate all plots ────────────────────────────────────────────────────────

def generate_all_plots(dataset_path: str = "dataset/afad_v1.jsonl",
                       output_dir: str = "analysis/results/figures") -> None:
    """Generate all standard AFA plots from the AFAD dataset."""
    if not HAS_PLOTTING:
        print("[ERROR] Plotting libraries not installed. Run: pip install matplotlib seaborn pandas")
        return

    print(f"Loading {dataset_path}")
    records = load_afad(dataset_path)
    print(f"Loaded {len(records)} records\n")

    plot_failure_distribution(records, output_dir)
    plot_model_comparison(records, output_dir)
    plot_category_heatmap(records, output_dir)
    plot_severity_distribution(records, output_dir)
    plot_subcategory_frequency(records, output_dir=output_dir)
    plot_early_signals(records, output_dir)
    plot_task_model_heatmap(records, output_dir)

    print(f"\nAll 7 plots saved to: {output_dir}")


if __name__ == "__main__":
    generate_all_plots()
