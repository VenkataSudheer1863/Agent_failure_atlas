"""
annotator.py
Annotation utilities for the Agent Failure Atlas Dataset (AFAD).
Includes batch annotation helpers and Inter-Annotator Agreement (IAA) computation.

Usage:
    python annotation/annotator.py --iaa annotator1.jsonl annotator2.jsonl
    python annotation/annotator.py --validate experiments/results/raw/Llama-3.1-8B/trajectories.jsonl
"""

import json
import sys
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple


# ── Taxonomy codes ──────────────────────────────────────────────────────────

VALID_LABELS = [
    "PLAN", "REAS", "TOOL", "MEM", "EXEC", "COOR", "SAFE", "ALIG"
]

VALID_SUBCATEGORIES = [
    "PLAN-MS", "PLAN-WO", "PLAN-PL", "PLAN-RP",
    "REAS-HA", "REAS-CO", "REAS-II", "REAS-UC",
    "TOOL-WT", "TOOL-PE", "TOOL-AM", "TOOL-PF",
    "MEM-CL",  "MEM-GF",  "MEM-SC",  "MEM-MH",
    "EXEC-IL", "EXEC-PT", "EXEC-RA", "EXEC-TA",
    "COOR-CB", "COOR-RC", "COOR-DL", "COOR-CF",
    "SAFE-PI", "SAFE-UA", "SAFE-DL", "SAFE-PV",
    "ALIG-GD", "ALIG-RH", "ALIG-SS", "ALIG-MI",
]


# ── Loading ──────────────────────────────────────────────────────────────────

def load_jsonl(filepath: str) -> List[Dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict], filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} records to {filepath}")


# ── Validation ───────────────────────────────────────────────────────────────

def validate_record(record: Dict) -> List[str]:
    """Return list of validation errors for a single record."""
    errors = []
    # Accept task_id as synonym for id (AFAD benchmark records use task_id)
    if "id" not in record and "task_id" in record:
        record = {**record, "id": record["task_id"]}

    # Structural fields required for all records
    required_always = ["id", "model", "task_type", "trajectory", "outcome", "recovered"]
    for field in required_always:
        if field not in record:
            errors.append(f"Missing field: {field}")

    # Annotation fields only required for non-success records
    outcome = record.get("outcome", "")
    if outcome in ("failure", "partial"):
        for field in ["failure_label", "failure_subcategory", "root_cause", "severity_score"]:
            if field not in record:
                errors.append(f"Missing field: {field}")

    if "outcome" in record and record["outcome"] not in ("success", "failure", "partial"):
        errors.append(f"Invalid outcome: {record['outcome']}")

    # Only validate failure annotation fields for non-success records
    if outcome in ("failure", "partial"):
        if "failure_label" in record and record["failure_label"] not in VALID_LABELS:
            errors.append(f"Invalid failure_label: {record['failure_label']}")

        if "failure_subcategory" in record and record["failure_subcategory"] not in VALID_SUBCATEGORIES:
            errors.append(f"Invalid failure_subcategory: {record['failure_subcategory']}")

        if "severity_score" in record:
            score = record["severity_score"]
            if not isinstance(score, int) or not (1 <= score <= 5):
                errors.append(f"severity_score must be int 1-5, got: {score}")

        if "failure_label" in record and "failure_subcategory" in record:
            label = record["failure_label"]
            subcat = record["failure_subcategory"]
            if label and subcat and not subcat.startswith(label + "-"):
                errors.append(
                    f"Subcategory {subcat} does not belong to label {label}"
                )

    return errors


def validate_dataset(filepath: str) -> dict:
    """Validate all records in a JSONL file and print a report. Returns summary dict."""
    records = load_jsonl(filepath)
    total_errors = 0
    error_details = []
    for i, record in enumerate(records):
        errors = validate_record(record)
        if errors:
            rid = record.get("id", record.get("task_id", f"record_{i}"))
            print(f"[{rid}] {len(errors)} error(s):")
            for e in errors:
                print(f"  - {e}")
            total_errors += len(errors)
            error_details.append({"record_id": rid, "errors": errors})

    print(f"\nValidated {len(records)} records. Total errors: {total_errors}")
    if total_errors == 0:
        print("All records are valid.")
    return {"file": filepath, "n_records": len(records), "n_errors": total_errors, "errors": error_details}


# ── IAA: Cohen's Kappa ────────────────────────────────────────────────────────

def cohen_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    """Compute Cohen's Kappa for two sequences of categorical labels."""
    assert len(labels_a) == len(labels_b), "Both annotators must label the same set of items"
    n = len(labels_a)
    if n == 0:
        return 0.0

    # Observed agreement
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    # Expected agreement
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    all_labels = set(count_a) | set(count_b)
    expected = sum(
        (count_a.get(label, 0) / n) * (count_b.get(label, 0) / n)
        for label in all_labels
    )

    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def align_records(
    records_a: List[Dict], records_b: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """Align two annotation sets by record ID."""
    index_a = {r["id"]: r for r in records_a}
    index_b = {r["id"]: r for r in records_b}
    shared_ids = sorted(set(index_a) & set(index_b))
    if not shared_ids:
        raise ValueError("No shared record IDs found between the two annotator files")
    return [index_a[i] for i in shared_ids], [index_b[i] for i in shared_ids]


def compute_iaa(filepath_a: str, filepath_b: str) -> None:
    """
    Compute and print IAA metrics between two annotator files.

    Args:
        filepath_a: Path to annotator A's JSONL
        filepath_b: Path to annotator B's JSONL
    """
    records_a = load_jsonl(filepath_a)
    records_b = load_jsonl(filepath_b)

    aligned_a, aligned_b = align_records(records_a, records_b)
    n = len(aligned_a)
    print(f"Aligned {n} records for IAA computation\n")

    # Top-level category
    labels_category_a = [r["failure_label"] for r in aligned_a]
    labels_category_b = [r["failure_label"] for r in aligned_b]
    kappa_category = cohen_kappa(labels_category_a, labels_category_b)

    # Subcategory
    labels_subcat_a = [r["failure_subcategory"] for r in aligned_a]
    labels_subcat_b = [r["failure_subcategory"] for r in aligned_b]
    kappa_subcat = cohen_kappa(labels_subcat_a, labels_subcat_b)

    # Severity (treat as categorical)
    labels_severity_a = [str(r["severity_score"]) for r in aligned_a]
    labels_severity_b = [str(r["severity_score"]) for r in aligned_b]
    kappa_severity = cohen_kappa(labels_severity_a, labels_severity_b)

    # Outcome
    labels_outcome_a = [r["outcome"] for r in aligned_a]
    labels_outcome_b = [r["outcome"] for r in aligned_b]
    kappa_outcome = cohen_kappa(labels_outcome_a, labels_outcome_b)

    print("=== Inter-Annotator Agreement Report ===")
    print(f"{'Dimension':<30} {'κ (Cohen)':>12} {'Interpretation':>20}")
    print("-" * 65)
    for dim, kappa in [
        ("Top-level category", kappa_category),
        ("Subcategory", kappa_subcat),
        ("Severity score", kappa_severity),
        ("Outcome", kappa_outcome),
    ]:
        interp = interpret_kappa(kappa)
        print(f"{dim:<30} {kappa:>12.4f} {interp:>20}")

    print(f"\nTotal aligned pairs: {n}")
    print(
        "\nTarget thresholds: category ≥ 0.80 | subcategory ≥ 0.70 | severity ≥ 0.65"
    )


def interpret_kappa(kappa: float) -> str:
    if kappa < 0:
        return "Poor (< 0)"
    elif kappa < 0.2:
        return "Slight"
    elif kappa < 0.4:
        return "Fair"
    elif kappa < 0.6:
        return "Moderate"
    elif kappa < 0.8:
        return "Substantial"
    else:
        return "Almost Perfect"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AFAD Annotation Utilities")
    subparsers = parser.add_subparsers(dest="command")

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate a JSONL dataset file")
    val_parser.add_argument("filepath", help="Path to AFAD JSONL file")

    # IAA command
    iaa_parser = subparsers.add_parser("iaa", help="Compute IAA between two annotator files")
    iaa_parser.add_argument("file_a", help="Annotator A's JSONL file")
    iaa_parser.add_argument("file_b", help="Annotator B's JSONL file")

    args = parser.parse_args()

    if args.command == "validate":
        validate_dataset(args.filepath)
    elif args.command == "iaa":
        compute_iaa(args.file_a, args.file_b)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
