"""
taxonomy.py

Taxonomy loading and resolution utilities for the Agent Failure Atlas.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


# ── Inline taxonomy for use without loading taxonomy.json ────────────────────

CATEGORY_NAMES = {
    "PLAN": "Planning",
    "REAS": "Reasoning",
    "TOOL": "Tool Use",
    "MEM": "Memory",
    "EXEC": "Execution",
    "COOR": "Coordination",
    "SAFE": "Safety",
    "ALIG": "Alignment",
}

SUBCATEGORY_NAMES = {
    "PLAN-MS": "Missing Steps",
    "PLAN-WO": "Wrong Ordering",
    "PLAN-PL": "Planning Loops",
    "PLAN-RP": "Redundant Plans",
    "REAS-HA": "Hallucination",
    "REAS-CO": "Contradiction",
    "REAS-II": "Invalid Inference",
    "REAS-UC": "Unsupported Conclusions",
    "TOOL-WT": "Wrong Tool Selection",
    "TOOL-PE": "Parameter Errors",
    "TOOL-AM": "API Misuse",
    "TOOL-PF": "Parsing Failures",
    "MEM-CL": "Context Loss",
    "MEM-GF": "Goal Forgetting",
    "MEM-SC": "State Corruption",
    "MEM-MH": "Memory Hallucination",
    "EXEC-IL": "Infinite Loops",
    "EXEC-PT": "Premature Termination",
    "EXEC-RA": "Repeated Actions",
    "EXEC-TA": "Task Abandonment",
    "COOR-CB": "Communication Breakdown",
    "COOR-RC": "Role Confusion",
    "COOR-DL": "Deadlocks",
    "COOR-CF": "Conflicts",
    "SAFE-PI": "Prompt Injection",
    "SAFE-UA": "Unsafe Actions",
    "SAFE-DL": "Data Leakage",
    "SAFE-PV": "Policy Violations",
    "ALIG-GD": "Goal Drift",
    "ALIG-RH": "Reward Hacking",
    "ALIG-SS": "Specification Gaming",
    "ALIG-MI": "Misalignment",
}

SUBCATEGORY_RECOVERABILITY = {
    "PLAN-MS": True, "PLAN-WO": True, "PLAN-PL": False, "PLAN-RP": True,
    "REAS-HA": False, "REAS-CO": True, "REAS-II": False, "REAS-UC": True,
    "TOOL-WT": True, "TOOL-PE": True, "TOOL-AM": False, "TOOL-PF": True,
    "MEM-CL": True, "MEM-GF": False, "MEM-SC": False, "MEM-MH": False,
    "EXEC-IL": False, "EXEC-PT": False, "EXEC-RA": True, "EXEC-TA": False,
    "COOR-CB": True, "COOR-RC": True, "COOR-DL": False, "COOR-CF": False,
    "SAFE-PI": False, "SAFE-UA": False, "SAFE-DL": False, "SAFE-PV": False,
    "ALIG-GD": True, "ALIG-RH": False, "ALIG-SS": False, "ALIG-MI": False,
}


def get_category_name(code: str) -> str:
    """Return human-readable name for a top-level category code."""
    return CATEGORY_NAMES.get(code, f"Unknown ({code})")


def get_subcategory_name(code: str) -> str:
    """Return human-readable name for a subcategory code."""
    return SUBCATEGORY_NAMES.get(code, f"Unknown ({code})")


def get_category_for_subcategory(subcat_code: str) -> Optional[str]:
    """Return the top-level category code for a subcategory code."""
    if "-" in subcat_code:
        return subcat_code.split("-")[0]
    return None


def is_recoverable(subcat_code: str) -> bool:
    """Return whether a failure subcategory is typically recoverable."""
    return SUBCATEGORY_RECOVERABILITY.get(subcat_code, False)


def get_all_subcategories_for_category(category_code: str) -> List[str]:
    """Return all subcategory codes for a given top-level category."""
    return [code for code in SUBCATEGORY_NAMES if code.startswith(category_code + "-")]


def load_taxonomy_from_file(taxonomy_path: str = "taxonomy/taxonomy.json") -> Dict:
    """Load the full taxonomy JSON from file."""
    path = Path(taxonomy_path)
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_subcategory_code(code: str) -> bool:
    """Check if a string is a valid subcategory code."""
    return code in SUBCATEGORY_NAMES


def validate_category_code(code: str) -> bool:
    """Check if a string is a valid top-level category code."""
    return code in CATEGORY_NAMES


if __name__ == "__main__":
    print("AFA Taxonomy Utilities - Self Test")
    print(f"Total categories: {len(CATEGORY_NAMES)}")
    print(f"Total subcategories: {len(SUBCATEGORY_NAMES)}")
    print(f"\nRecoverable subcategories: {sum(SUBCATEGORY_RECOVERABILITY.values())}")
    print(f"Non-recoverable subcategories: {sum(not v for v in SUBCATEGORY_RECOVERABILITY.values())}")

    print("\nSubcategories for PLAN:")
    for c in get_all_subcategories_for_category("PLAN"):
        print(f"  {c} - {get_subcategory_name(c)} (recoverable: {is_recoverable(c)})")
