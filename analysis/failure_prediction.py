"""
failure_prediction.py

Early failure signal analysis for the Agent Failure Atlas.
Analyzes whether early trajectory steps contain signals predictive of final outcome.
Trains simple classifiers to predict failure from early trajectory features.

Usage:
    python analysis/failure_prediction.py
    python analysis/failure_prediction.py --dataset dataset/afad_v1.jsonl
"""

import json
import sys
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARNING] scikit-learn not installed. Run: pip install scikit-learn pandas numpy")


FAILURE_SIGNAL_KEYWORDS = {
    "loop_signal": ["again", "retry", "same", "repeated", "loop", "replan", "re-plan"],
    "uncertainty_signal": ["i'm not sure", "unclear", "ambiguous", "maybe", "might", "could be"],
    "error_signal": ["error", "failed", "exception", "cannot", "unable", "refused"],
    "abandon_signal": ["give up", "abandon", "impossible", "can't", "cannot complete"],
    "tool_failure_signal": ["bad request", "400", "429", "401", "tool error", "api error"],
    "hallucination_signal": ["as i mentioned", "as you know", "i recall", "i remember"],
}


def load_afad(filepath: str) -> List[Dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_early_features(record: Dict, n_early_steps: int = 3) -> Dict:
    """
    Extract features from the first n_early_steps of a trajectory.
    These are used to predict the final failure outcome.
    """
    trajectory = record.get("trajectory", [])
    early_steps = trajectory[:n_early_steps]
    all_text = " ".join(
        (s.get("action", "") + " " + s.get("observation", "")).lower()
        for s in early_steps
    )

    features = {
        "n_steps_early": len(early_steps),
        "total_text_len": len(all_text),
    }

    # Signal keyword counts
    for signal_name, keywords in FAILURE_SIGNAL_KEYWORDS.items():
        features[signal_name] = sum(1 for kw in keywords if kw in all_text)

    # Tool use features
    tool_calls = [s for s in early_steps if s.get("tool_called")]
    features["n_tool_calls_early"] = len(tool_calls)
    features["has_tool_failure"] = int(
        any("error" in (s.get("tool_output") or "").lower() for s in early_steps)
    )

    # Model as categorical feature
    model_map = {
        "GPT-OSS-20B": 0, "Qwen3-8B": 1, "Qwen3-30B": 2,
        "DeepSeek-R1-8B": 3, "Gemma3-12B": 4, "Llama-3.2": 5
    }
    features["model_id"] = model_map.get(record.get("model", ""), -1)

    # Task type as categorical feature
    task_map = {
        "information_seeking": 0, "tool_use": 1, "planning": 2,
        "reasoning": 3, "multi_agent": 4
    }
    features["task_type_id"] = task_map.get(record.get("task_type", ""), -1)

    return features


def build_feature_matrix(records: List[Dict], n_early_steps: int = 3) -> Tuple:
    """Build feature matrix X and label vector y."""
    X, y = [], []
    for r in records:
        feats = extract_early_features(r, n_early_steps)
        X.append(list(feats.values()))
        y.append(1 if r.get("outcome") == "failure" else 0)
    return X, y, list(feats.keys())


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_early_signals(records: List[Dict]) -> None:
    """Analyze which early signals correlate with failure."""
    print("\n--- Early Signal Analysis ---")
    print(f"Total records: {len(records)}")

    outcomes = Counter(r.get("outcome") for r in records)
    print(f"Outcomes: {dict(outcomes)}")

    failures = [r for r in records if r.get("outcome") == "failure"]
    successes = [r for r in records if r.get("outcome") != "failure"]

    print(f"\nSignal presence in first 3 steps:")
    print(f"{'Signal':<30} {'Failure%':>10} {'Success%':>10}")
    print("-" * 55)

    for signal_name, keywords in FAILURE_SIGNAL_KEYWORDS.items():
        def has_signal(record):
            traj = record.get("trajectory", [])[:3]
            text = " ".join((s.get("action", "") + " " + s.get("observation", "")).lower()
                           for s in traj)
            return any(kw in text for kw in keywords)

        fail_pct = sum(1 for r in failures if has_signal(r)) / max(len(failures), 1) * 100
        succ_pct = sum(1 for r in successes if has_signal(r)) / max(len(successes), 1) * 100
        print(f"{signal_name:<30} {fail_pct:>9.1f}% {succ_pct:>9.1f}%")


def train_failure_predictor(records: List[Dict]) -> None:
    """Train and evaluate failure prediction classifiers."""
    if not HAS_SKLEARN:
        print("[SKIP] scikit-learn not installed; skipping classifier training")
        return

    print("\n--- Failure Prediction (Early Trajectory Features) ---")

    for n_steps in [1, 2, 3, 5]:
        X, y, feature_names = build_feature_matrix(records, n_early_steps=n_steps)
        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y)

        if sum(y_arr) < 10 or (len(y_arr) - sum(y_arr)) < 10:
            print(f"  Skipping n_steps={n_steps}: insufficient class balance")
            continue

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Logistic Regression
        lr = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
        lr_scores = cross_val_score(lr, X_arr, y_arr, cv=cv, scoring="roc_auc")

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        rf_scores = cross_val_score(rf, X_arr, y_arr, cv=cv, scoring="roc_auc")

        print(f"\n  Using first {n_steps} step(s) as features:")
        print(f"    Logistic Regression AUC: {lr_scores.mean():.3f} +/- {lr_scores.std():.3f}")
        print(f"    Random Forest AUC      : {rf_scores.mean():.3f} +/- {rf_scores.std():.3f}")

    # Feature importance with full-data Random Forest (3 steps)
    print("\n--- Feature Importances (Random Forest, 3 early steps) ---")
    X, y, feature_names = build_feature_matrix(records, n_early_steps=3)
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf.fit(X_arr, y_arr)
    importances = sorted(
        zip(feature_names, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for feat, imp in importances:
        bar = "#" * int(imp * 50)
        print(f"  {feat:<35} {imp:.4f} {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Failure prediction analysis")
    parser.add_argument("--dataset", default="dataset/afad_v1.jsonl")
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"Dataset not found: {args.dataset}")
        print("Generate it first: python dataset/generate_afad.py")
        return

    print(f"Loading {args.dataset}")
    records = load_afad(args.dataset)

    analyze_early_signals(records)
    train_failure_predictor(records)


if __name__ == "__main__":
    main()
