"""
metrics.py

Core evaluation metrics for the Agent Failure Atlas.
All metrics are defined exactly as in the paper.
"""

from typing import List, Dict, Optional
from collections import Counter


def compute_failure_rate(records: List[Dict]) -> float:
    """
    Failure Rate = Failed Tasks / Total Tasks
    
    Args:
        records: List of AFAD records
        
    Returns:
        Failure rate as a float in [0, 1]
    """
    if not records:
        return 0.0
    failed = sum(1 for r in records if r.get("outcome") == "failure")
    return failed / len(records)


def compute_recovery_rate(records: List[Dict]) -> float:
    """
    Recovery Rate = Recovered Failures / Total Failures
    
    Args:
        records: List of AFAD records
        
    Returns:
        Recovery rate as a float in [0, 1]
    """
    failures = [r for r in records if r.get("outcome") in ("failure", "partial")]
    if not failures:
        return 0.0
    recovered = sum(1 for r in failures if r.get("recovered") is True)
    return recovered / len(failures)


def compute_severity_score(records: List[Dict]) -> float:
    """
    Severity Score = Mean(Failure Severity)
    
    Args:
        records: List of AFAD records
        
    Returns:
        Mean severity score in [1, 5]
    """
    scores = [r["severity_score"] for r in records if "severity_score" in r and r["severity_score"] is not None]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_category_frequency(records: List[Dict]) -> Dict[str, float]:
    """
    Category Frequency = Distribution of failure types
    
    Args:
        records: List of AFAD records
        
    Returns:
        Dict mapping category code to proportion [0, 1]
    """
    if not records:
        return {}
    counts = Counter(r.get("failure_label") for r in records if r.get("failure_label"))
    total = sum(counts.values())
    return {cat: count / total for cat, count in counts.items()}


def compute_failure_density(records: List[Dict]) -> float:
    """
    Failure Density = Failures per Trajectory (average steps in failed trajectories)
    
    Args:
        records: List of AFAD records
        
    Returns:
        Mean number of steps in failed trajectories
    """
    failed = [r for r in records if r.get("outcome") == "failure"]
    if not failed:
        return 0.0
    step_counts = [len(r.get("trajectory", [])) for r in failed]
    return sum(step_counts) / len(step_counts)


def compute_all_metrics(records: List[Dict]) -> Dict:
    """
    Compute all AFA metrics for a list of records.
    
    Returns:
        Dict with all metric values
    """
    return {
        "total": len(records),
        "failure_rate": round(compute_failure_rate(records), 4),
        "recovery_rate": round(compute_recovery_rate(records), 4),
        "severity_score": round(compute_severity_score(records), 3),
        "category_frequency": {k: round(v, 4) for k, v in compute_category_frequency(records).items()},
        "failure_density": round(compute_failure_density(records), 2),
    }


def compute_per_model_metrics(records: List[Dict]) -> Dict[str, Dict]:
    """
    Compute all metrics broken down by model.
    
    Returns:
        Dict mapping model name to metrics dict
    """
    model_groups: Dict[str, List[Dict]] = {}
    for r in records:
        model = r.get("model", "unknown")
        model_groups.setdefault(model, []).append(r)
    return {model: compute_all_metrics(recs) for model, recs in model_groups.items()}


def compute_per_task_type_metrics(records: List[Dict]) -> Dict[str, Dict]:
    """
    Compute all metrics broken down by task type.
    
    Returns:
        Dict mapping task type to metrics dict
    """
    task_groups: Dict[str, List[Dict]] = {}
    for r in records:
        tt = r.get("task_type", "unknown")
        task_groups.setdefault(tt, []).append(r)
    return {tt: compute_all_metrics(recs) for tt, recs in task_groups.items()}


if __name__ == "__main__":
    # Self-test with minimal example records
    test_records = [
        {"outcome": "failure", "recovered": False, "severity_score": 4,
         "failure_label": "PLAN", "trajectory": [{"step": 1}, {"step": 2}], "model": "Llama-3.1-8B"},
        {"outcome": "failure", "recovered": True, "severity_score": 3,
         "failure_label": "REAS", "trajectory": [{"step": 1}], "model": "Llama-3.1-8B"},
        {"outcome": "success", "recovered": False, "severity_score": 1,
         "failure_label": "TOOL", "trajectory": [{"step": 1}, {"step": 2}, {"step": 3}], "model": "Llama-3.3-70B"},
    ]
    metrics = compute_all_metrics(test_records)
    print("Self-test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
