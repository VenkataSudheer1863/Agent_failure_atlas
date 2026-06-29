"""
generate_paper_figures.py

Generates all publication-ready figures for the AFA paper.
Outputs to paper/figures/ at 300 DPI.
"""

import json
import shutil
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODELS = ["Llama-3.1-8B", "Llama-4-Scout-17B", "Qwen3-32B", "Llama-3.3-70B", "GPT-OSS-20B", "GPT-OSS-120B"]
CATEGORIES = ["PLAN", "REAS", "TOOL", "MEM", "EXEC", "COOR", "SAFE", "ALIG"]
CATEGORY_FULL = {
    "PLAN": "Planning", "REAS": "Reasoning", "TOOL": "Tool Use",
    "MEM": "Memory", "EXEC": "Execution", "COOR": "Coordination",
    "SAFE": "Safety", "ALIG": "Alignment",
}
PALETTE = {
    "PLAN": "#4C72B0", "REAS": "#DD8452", "TOOL": "#55A868",
    "MEM": "#C44E52", "EXEC": "#8172B2", "COOR": "#937860",
    "SAFE": "#DA8BC3", "ALIG": "#8C8C8C",
}

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


def load_trajectories(results_dir="experiments/results/raw"):
    records = []
    for traj_file in sorted(Path(results_dir).glob("*/trajectories.jsonl")):
        for line in open(traj_file, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                r.setdefault("model", traj_file.parent.name)
                records.append(r)
    return records


# ── Figure 1: Taxonomy overview ───────────────────────────────────────────────

def fig_taxonomy_overview():
    """Visual overview of the 8-category taxonomy with subcategory counts."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 8.5)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")

    subcats = {
        "PLAN": ["PLAN-MS\nMissing Steps", "PLAN-WO\nWrong Order", "PLAN-PL\nPlanning Loops", "PLAN-RP\nRedundant Plans"],
        "REAS": ["REAS-HA\nHallucination", "REAS-CO\nContradiction", "REAS-II\nInvalid Inference", "REAS-UC\nUnsupported Conc."],
        "TOOL": ["TOOL-WT\nWrong Tool", "TOOL-PE\nParam Errors", "TOOL-AM\nAPI Misuse", "TOOL-PF\nParsing Failures"],
        "MEM":  ["MEM-CL\nContext Loss", "MEM-GF\nGoal Forgetting", "MEM-SC\nState Corruption", "MEM-MH\nMem. Hallucination"],
        "EXEC": ["EXEC-IL\nInfinite Loops", "EXEC-PT\nPremature Term.", "EXEC-RA\nRepeated Actions", "EXEC-TA\nTask Abandonment"],
        "COOR": ["COOR-CB\nComm. Breakdown", "COOR-RC\nRole Confusion", "COOR-DL\nDeadlocks", "COOR-CF\nConflicts"],
        "SAFE": ["SAFE-PI\nPrompt Injection", "SAFE-UA\nUnsafe Actions", "SAFE-DL\nData Leakage", "SAFE-PV\nPolicy Violations"],
        "ALIG": ["ALIG-GD\nGoal Drift", "ALIG-RH\nReward Hacking", "ALIG-SS\nSpec. Gaming", "ALIG-MI\nMisalignment"],
    }

    for row_idx, (cat, subs) in enumerate(subcats.items()):
        y = 7.5 - row_idx
        color = PALETTE[cat]
        # Category label box
        rect = mpatches.FancyBboxPatch((0.05, y - 0.35), 1.7, 0.7,
            boxstyle="round,pad=0.05", linewidth=1.5,
            edgecolor=color, facecolor=color, alpha=0.85)
        ax.add_patch(rect)
        ax.text(0.95, y, f"{cat}\n{CATEGORY_FULL[cat]}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

        # Subcategory boxes
        for j, sub in enumerate(subs):
            x = 2.2 + j * 2.95
            rect2 = mpatches.FancyBboxPatch((x, y - 0.33), 2.7, 0.66,
                boxstyle="round,pad=0.04", linewidth=1,
                edgecolor=color, facecolor=color, alpha=0.15)
            ax.add_patch(rect2)
            ax.text(x + 1.35, y, sub, ha="center", va="center",
                    fontsize=7.5, color="#333333")
            # Connector line
            ax.annotate("", xy=(x, y), xytext=(1.75, y),
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

    ax.text(7, 8.35, "AFA Taxonomy: 8 Categories × 4 Subcategories = 32 Failure Codes",
            ha="center", fontsize=13, fontweight="bold", color="#222222")

    plt.tight_layout(pad=0.5)
    path = OUT / "fig1_taxonomy_overview.pdf"
    plt.savefig(OUT / "fig1_taxonomy_overview.png", dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {OUT}/fig1_taxonomy_overview.png")


# ── Figure 2: Failure category distribution ───────────────────────────────────

def fig_category_distribution(records):
    failure_records = [r for r in records if r.get("outcome") == "failure"]
    cat_counts = Counter(r["failure_label"] for r in failure_records if r.get("failure_label"))
    cats = [c for c in CATEGORIES if c in cat_counts]
    counts = [cat_counts[c] for c in cats]
    labels = [f"{c}\n({CATEGORY_FULL[c]})" for c in cats]
    colors = [PALETTE[c] for c in cats]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8, width=0.65)
    ax.bar_label(bars, padding=3, fontsize=10, fontweight="bold")
    ax.set_title("AFAD v1.0: Failure Category Distribution (N = 283, strict failures)", fontweight="bold", pad=10)
    ax.set_xlabel("Failure Category")
    ax.set_ylabel("Number of Annotated Trajectories")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUT / "fig2_category_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}/fig2_category_distribution.png")


# ── Figure 3: Cross-model comparison ─────────────────────────────────────────

def fig_model_comparison(records):
    model_data = {m: {"total": 0, "failures": 0, "non_success": 0, "severity": [], "recovered": 0}
                  for m in MODELS}
    for r in records:
        m = r.get("model")
        if m not in model_data:
            continue
        model_data[m]["total"] += 1
        if r.get("outcome") == "failure":
            model_data[m]["failures"] += 1
        if r.get("outcome") in ("failure", "partial"):
            model_data[m]["non_success"] += 1
        if r.get("recovered") is True:
            model_data[m]["recovered"] += 1
        if r.get("severity_score"):
            model_data[m]["severity"].append(r["severity_score"])

    names = [m for m in MODELS if model_data[m]["total"] > 0]
    fail_rates  = [model_data[m]["failures"] / model_data[m]["total"] * 100 for m in names]
    rec_rates   = [model_data[m]["recovered"] / max(model_data[m]["non_success"], 1) * 100 for m in names]
    severities  = [sum(model_data[m]["severity"]) / max(len(model_data[m]["severity"]), 1) for m in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = sns.color_palette("muted", len(names))

    for ax, vals, title, xlabel, fmt in [
        (axes[0], fail_rates,  "Failure Rate (%)",      "Failure Rate (%)",    "{:.1f}%"),
        (axes[1], rec_rates,   "Recovery Rate (%)",     "Recovery Rate (%)",   "{:.1f}%"),
        (axes[2], severities,  "Mean Severity Score",   "Mean Severity (1–5)", "{:.2f}"),
    ]:
        bars = ax.barh(names, vals, color=palette, edgecolor="white", height=0.55)
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, max(vals) * 1.2 + 2 if vals else 100)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                    fmt.format(v), va="center", fontsize=9)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, alpha=0.35)

    fig.suptitle("Cross-Model Failure Analysis — AFA Benchmark (N = 450)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "fig3_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}/fig3_model_comparison.png")


# ── Figure 4: Category heatmap ────────────────────────────────────────────────

def fig_category_heatmap(records):
    data = defaultdict(Counter)
    model_totals = Counter()
    for r in records:
        m = r.get("model")
        model_totals[m] += 1  # denominator = all 75 trajectories per model
        if r.get("outcome") == "failure":
            cat = r.get("failure_label")
            if cat:
                data[m][cat] += 1

    names = [m for m in MODELS if m in model_totals]
    matrix = [[data[m].get(c, 0) / model_totals[m] * 100 for c in CATEGORIES] for m in names]
    df = pd.DataFrame(matrix, index=names, columns=[f"{c}\n{CATEGORY_FULL[c]}" for c in CATEGORIES])

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.4,
                ax=ax, cbar_kws={"label": "% of model trajectories", "shrink": 0.8},
                vmin=0, vmax=25)
    ax.set_title("Failure Category Distribution per Model (% of Records)", fontweight="bold", pad=12)
    ax.set_xlabel("Failure Category")
    ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(OUT / "fig4_category_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}/fig4_category_heatmap.png")


# ── Figure 5: Severity distribution ──────────────────────────────────────────

def fig_severity_distribution(records):
    model_sev = defaultdict(list)
    for r in records:
        m = r.get("model")
        s = r.get("severity_score")
        if m in MODELS and s:
            model_sev[m].append(s)

    names = [m for m in MODELS if m in model_sev]
    sev_colors = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c", 5: "#8e44ad"}

    fig, ax = plt.subplots(figsize=(12, 6))
    bottoms = np.zeros(len(names))
    for sev in [1, 2, 3, 4, 5]:
        vals = np.array([
            sum(1 for x in model_sev[m] if x == sev) / len(model_sev[m]) * 100
            for m in names
        ])
        ax.bar(names, vals, bottom=bottoms, color=sev_colors[sev],
               label=f"Severity {sev}", edgecolor="white", linewidth=0.4, width=0.6)
        bottoms += vals

    ax.set_title("Severity Score Distribution per Model (%)", fontweight="bold")
    ax.set_ylabel("Percentage of Trajectories (%)")
    ax.set_xlabel("Model")
    ax.legend(title="Severity Level", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.set_ylim(0, 108)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "fig5_severity_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}/fig5_severity_distribution.png")


# ── Figure 6: Top-15 subcategory frequency ────────────────────────────────────

def fig_subcategory_frequency(records):
    subcat_counts = Counter(r.get("failure_subcategory") for r in records if r.get("failure_subcategory"))
    top = subcat_counts.most_common(15)
    labels = [t[0] for t in top]
    values = [t[1] for t in top]
    colors = [PALETTE.get(l.split("-")[0], "#8C8C8C") for l in labels]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white", height=0.65)
    ax.bar_label(bars, padding=3, fontsize=9)
    n = len(top)
    ax.set_title(f"Failure Subcategory Distribution ({n} observed, of 32 defined codes)", fontweight="bold")
    ax.set_xlabel("Number of Annotated Trajectories")
    ax.xaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
    legend_patches = [mpatches.Patch(color=PALETTE[c], label=f"{c} — {CATEGORY_FULL[c]}") for c in CATEGORIES]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right", ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(OUT / "fig6_subcategory_frequency.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}/fig6_subcategory_frequency.png")


# ── Figure 7: Early failure signals ──────────────────────────────────────────

SIGNAL_KEYWORDS = {
    "Loop / Replan": ["again", "retry", "same", "repeated", "loop", "replan"],
    "Uncertainty": ["unclear", "ambiguous", "maybe", "might", "could be"],
    "Error / Failure": ["error", "failed", "exception", "cannot", "unable"],
    "Tool Failure": ["bad request", "400", "429", "401", "tool error"],
    "Hallucination": ["as i mentioned", "as you know", "i recall"],
    "Abandonment": ["give up", "abandon", "impossible", "cannot complete"],
}


def fig_early_signals(records):
    failures = [r for r in records if r.get("outcome") == "failure"]
    successes = [r for r in records if r.get("outcome") == "success"]

    def pct(recs, kws):
        c = 0
        for r in recs:
            traj = r.get("trajectory", [])[:3]
            text = " ".join((s.get("action","") + " " + s.get("observation","")).lower() for s in traj)
            if any(k in text for k in kws):
                c += 1
        return c / max(len(recs), 1) * 100

    names = list(SIGNAL_KEYWORDS.keys())
    fail_pcts = [pct(failures, SIGNAL_KEYWORDS[s]) for s in names]
    succ_pcts = [pct(successes, SIGNAL_KEYWORDS[s]) for s in names]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w/2, fail_pcts, w, label=f"Failed (n={len(failures)})",
                color="#C44E52", edgecolor="white")
    b2 = ax.bar(x + w/2, succ_pcts, w, label=f"Successful (n={len(successes)})",
                color="#55A868", edgecolor="white")
    ax.bar_label(b1, fmt="%.1f%%", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f%%", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("% of Trajectories Containing Signal\n(in First 3 Steps)")
    ax.set_title("Early Trajectory Signals: Failed vs. Successful Trajectories", fontweight="bold")
    ax.legend(frameon=True)
    ax.set_ylim(0, max(max(fail_pcts), max(succ_pcts)) * 1.35 + 3)
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUT / "fig7_early_signals.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}/fig7_early_signals.png")


# ── Figure 8: Task-type × model failure heatmap ──────────────────────────────

def fig_task_model_heatmap(records):
    TASK_LABELS = {
        "information_seeking": "Info Seeking",
        "tool_use": "Tool Use",
        "planning": "Planning",
        "reasoning": "Reasoning",
        "multi_agent": "Multi-Agent",
    }
    task_types = list(TASK_LABELS.keys())
    counts = {m: {tt: {"total": 0, "failed": 0} for tt in task_types} for m in MODELS}
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
            row.append(d["failed"] / d["total"] * 100 if d["total"] > 0 else float("nan"))
        matrix.append(row)

    df = pd.DataFrame(matrix, index=MODELS, columns=[TASK_LABELS[tt] for tt in task_types])
    flat = [v for row in matrix for v in row if not (isinstance(v, float) and v != v)]
    vmin_val = max(0, min(flat) - 5) if flat else 0
    vmax_val = min(100, max(flat) + 5) if flat else 100
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn_r", linewidths=0.4, ax=ax,
                vmin=vmin_val, vmax=vmax_val, cbar_kws={"label": "Strict Failure Rate (%)", "shrink": 0.8})
    ax.set_title("Strict Failure Rate (%) by Task Type and Model", fontweight="bold", pad=12)
    ax.set_xlabel("Task Type")
    ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(OUT / "fig8_task_model_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}/fig8_task_model_heatmap.png")


# ── Figure 9: Failure prediction AUC curves (bar chart) ──────────────────────

def fig_failure_prediction_auc():
    steps = [1, 2, 3, 5]
    lr_auc  = [0.647, 0.648, 0.650, 0.658]
    rf_auc  = [0.663, 0.672, 0.683, 0.658]
    lr_std  = [0.043, 0.041, 0.036, 0.041]
    rf_std  = [0.030, 0.025, 0.017, 0.022]

    x = np.arange(len(steps))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, lr_auc, w, yerr=lr_std, capsize=4, label="Logistic Regression",
           color="#4C72B0", edgecolor="white", error_kw={"ecolor": "#222"})
    ax.bar(x + w/2, rf_auc, w, yerr=rf_std, capsize=4, label="Random Forest",
           color="#DD8452", edgecolor="white", error_kw={"ecolor": "#222"})
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random baseline (AUC = 0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"First {s} Step{'s' if s > 1 else ''}" for s in steps])
    ax.set_ylabel("ROC-AUC (5-fold CV, mean ± std)")
    ax.set_title("Early Failure Prediction AUC vs. Number of Trajectory Steps Used",
                 fontweight="bold")
    ax.set_ylim(0.40, 0.80)
    ax.legend(frameon=True)
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
    for i, (lr, rf) in enumerate(zip(lr_auc, rf_auc)):
        ax.text(i - w/2, lr + 0.012, f"{lr:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, rf + 0.012, f"{rf:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig9_failure_prediction_auc.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}/fig9_failure_prediction_auc.png")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results_dir = "experiments/results/raw"
    if not Path(results_dir).exists() or not any(Path(results_dir).glob("*/trajectories.jsonl")):
        print(f"No benchmark results found in {results_dir}")
        print("Run the benchmark first: python experiments/run_benchmark.py --tasks-per-type 15")
        raise SystemExit(1)
    print(f"Loading trajectories from {results_dir}...")
    records = load_trajectories(results_dir)
    print(f"Loaded {len(records)} records\n")

    print("Generating paper figures...")
    fig_taxonomy_overview()
    fig_category_distribution(records)
    fig_model_comparison(records)
    fig_category_heatmap(records)
    fig_severity_distribution(records)
    fig_subcategory_frequency(records)
    fig_early_signals(records)
    fig_task_model_heatmap(records)
    fig_failure_prediction_auc()

    print(f"\nAll 9 figures saved to: {OUT}")
