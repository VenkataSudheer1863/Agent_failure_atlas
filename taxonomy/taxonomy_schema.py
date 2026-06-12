"""
taxonomy_schema.py
Programmatic JSON schema and validation utilities for AFAD records.
"""

VALID_MODELS = [
    "GPT-OSS-20B",
    "Qwen3-8B",
    "Qwen3-30B",
    "DeepSeek-R1-8B",
    "Gemma3-12B",
    "Llama-3.2",
]

VALID_TASK_TYPES = [
    "information_seeking",
    "tool_use",
    "planning",
    "reasoning",
    "multi_agent",
]

VALID_FAILURE_LABELS = [
    "PLAN", "REAS", "TOOL", "MEM",
    "EXEC", "COOR", "SAFE", "ALIG",
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

AFAD_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AFAD Record Schema",
    "description": "Schema for a single record in the Agent Failure Atlas Dataset (AFAD)",
    "type": "object",
    "required": [
        "id", "model", "task_type", "trajectory",
        "failure_label", "failure_subcategory",
        "root_cause", "severity_score", "outcome", "recovered"
    ],
    "properties": {
        "id": {"type": "string"},
        "model": {"type": "string", "enum": VALID_MODELS},
        "task_type": {"type": "string", "enum": VALID_TASK_TYPES},
        "task_id": {"type": "string"},
        "trajectory": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["step", "action", "observation"],
                "properties": {
                    "step": {"type": "integer"},
                    "action": {"type": "string"},
                    "observation": {"type": "string"},
                    "tool_called": {"type": "string"},
                    "tool_params": {"type": "object"},
                    "tool_output": {"type": "string"},
                }
            }
        },
        "failure_label": {"type": "string", "enum": VALID_FAILURE_LABELS},
        "failure_subcategory": {"type": "string", "enum": VALID_SUBCATEGORIES},
        "root_cause": {"type": "string"},
        "severity_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "outcome": {"type": "string", "enum": ["success", "failure", "partial"]},
        "recovered": {"type": "boolean"},
        "recovery_steps": {"type": "integer"},
        "annotator_notes": {"type": "string"},
    }
}


def validate_record(record: dict) -> list:
    """
    Validate a single AFAD record against the schema.
    Returns a list of error messages. Empty list means valid.
    """
    errors = []
    required = [
        "id", "model", "task_type", "trajectory",
        "failure_label", "failure_subcategory",
        "root_cause", "severity_score", "outcome", "recovered"
    ]
    for field in required:
        if field not in record:
            errors.append(f"Missing required field: {field}")

    if "model" in record and record["model"] not in VALID_MODELS:
        errors.append(f"Invalid model: {record['model']}. Must be one of {VALID_MODELS}")

    if "task_type" in record and record["task_type"] not in VALID_TASK_TYPES:
        errors.append(f"Invalid task_type: {record['task_type']}")

    if "failure_label" in record and record["failure_label"] not in VALID_FAILURE_LABELS:
        errors.append(f"Invalid failure_label: {record['failure_label']}")

    if "failure_subcategory" in record and record["failure_subcategory"] not in VALID_SUBCATEGORIES:
        errors.append(f"Invalid failure_subcategory: {record['failure_subcategory']}")

    if "severity_score" in record:
        if not isinstance(record["severity_score"], int) or not (1 <= record["severity_score"] <= 5):
            errors.append("severity_score must be an integer between 1 and 5")

    if "outcome" in record and record["outcome"] not in ["success", "failure", "partial"]:
        errors.append("outcome must be 'success', 'failure', or 'partial'")

    if "trajectory" in record and not isinstance(record["trajectory"], list):
        errors.append("trajectory must be a list of steps")

    return errors


if __name__ == "__main__":
    # Quick self-test
    sample = {
        "id": "AFAD-0001",
        "model": "Qwen3-8B",
        "task_type": "planning",
        "task_id": "PLAN-001",
        "trajectory": [
            {"step": 1, "action": "Plan task", "observation": "Planning loop detected"}
        ],
        "failure_label": "PLAN",
        "failure_subcategory": "PLAN-PL",
        "root_cause": "Agent repeatedly reformulates plan without execution",
        "severity_score": 4,
        "outcome": "failure",
        "recovered": False,
    }
    errs = validate_record(sample)
    if errs:
        print("Validation errors:", errs)
    else:
        print("Record is valid.")
