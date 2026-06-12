"""
loader.py
Utilities for loading, filtering, and splitting the AFAD dataset.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional


def load_afad(filepath: str) -> List[Dict]:
    """
    Load an AFAD JSONL file and return a list of record dicts.

    Args:
        filepath: Path to the .jsonl file

    Returns:
        List of record dicts
    """
    records = []
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def filter_by_model(records: List[Dict], model: str) -> List[Dict]:
    """Filter records by model name."""
    return [r for r in records if r.get("model") == model]


def filter_by_category(records: List[Dict], category: str) -> List[Dict]:
    """Filter records by top-level failure category (e.g., 'PLAN', 'REAS')."""
    return [r for r in records if r.get("failure_label") == category]


def filter_by_subcategory(records: List[Dict], subcategory: str) -> List[Dict]:
    """Filter records by failure subcategory code (e.g., 'PLAN-PL')."""
    return [r for r in records if r.get("failure_subcategory") == subcategory]


def filter_by_outcome(records: List[Dict], outcome: str) -> List[Dict]:
    """Filter by outcome: 'success', 'failure', or 'partial'."""
    return [r for r in records if r.get("outcome") == outcome]


def filter_by_severity(records: List[Dict], min_severity: int = 1, max_severity: int = 5) -> List[Dict]:
    """Filter records by severity score range."""
    return [
        r for r in records
        if min_severity <= r.get("severity_score", 0) <= max_severity
    ]


def get_statistics(records: List[Dict]) -> Dict:
    """
    Compute summary statistics for a list of AFAD records.

    Returns:
        Dict with counts by model, category, subcategory, outcome, and severity distribution
    """
    from collections import Counter

    stats = {
        "total": len(records),
        "by_model": Counter(r.get("model") for r in records),
        "by_category": Counter(r.get("failure_label") for r in records),
        "by_subcategory": Counter(r.get("failure_subcategory") for r in records),
        "by_outcome": Counter(r.get("outcome") for r in records),
        "severity_distribution": Counter(r.get("severity_score") for r in records),
        "recovery_rate": sum(1 for r in records if r.get("recovered")) / max(len(records), 1),
        "failure_rate": sum(1 for r in records if r.get("outcome") == "failure") / max(len(records), 1),
    }
    return stats


def make_splits(
    records: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Split records into train/val/test sets.

    Args:
        records: Full list of records
        train_ratio: Fraction for training (default 0.70)
        val_ratio: Fraction for validation (default 0.15)
        seed: Random seed for reproducibility

    Returns:
        Dict with 'train', 'val', 'test' keys
    """
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train: n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def save_splits(splits: Dict[str, List[Dict]], output_dir: str) -> None:
    """Save train/val/test splits to JSONL files in output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in splits.items():
        out_path = out / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for record in split_records:
                f.write(json.dumps(record) + "\n")
        print(f"Saved {len(split_records)} records to {out_path}")


def load_splits(splits_dir: str) -> Dict[str, List[Dict]]:
    """Load train/val/test splits from a directory of JSONL files."""
    splits_path = Path(splits_dir)
    splits = {}
    for split_name in ["train", "val", "test"]:
        path = splits_path / f"{split_name}.jsonl"
        if path.exists():
            splits[split_name] = load_afad(str(path))
    return splits


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python loader.py <path_to_afad.jsonl>")
        sys.exit(1)

    filepath = sys.argv[1]
    records = load_afad(filepath)
    stats = get_statistics(records)
    print(f"\n=== AFAD Dataset Statistics ===")
    print(f"Total records : {stats['total']}")
    print(f"Failure rate  : {stats['failure_rate']:.1%}")
    print(f"Recovery rate : {stats['recovery_rate']:.1%}")
    print(f"\nBy Model:")
    for model, count in sorted(stats["by_model"].items()):
        print(f"  {model:20s}: {count}")
    print(f"\nBy Category:")
    for cat, count in sorted(stats["by_category"].items()):
        print(f"  {cat:10s}: {count}")
    print(f"\nBy Outcome:")
    for outcome, count in sorted(stats["by_outcome"].items()):
        print(f"  {outcome:10s}: {count}")
