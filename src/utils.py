"""
utils.py

Common utilities for the Agent Failure Atlas project.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger for a module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── JSON I/O ──────────────────────────────────────────────────────────────────

def load_jsonl(filepath: str) -> List[Dict]:
    """Load a JSONL file and return list of dicts."""
    records = []
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {filepath}: {e}")
    return records


def save_jsonl(records: List[Dict], filepath: str, indent: bool = False) -> None:
    """Save list of dicts to a JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            if indent:
                f.write(json.dumps(record, indent=2) + "\n")
            else:
                f.write(json.dumps(record) + "\n")


def load_json(filepath: str) -> Any:
    """Load a single JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save data as a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


# ── Path utilities ────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    return Path(__file__).parent.parent


# ── Timestamp ─────────────────────────────────────────────────────────────────

def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def now_short() -> str:
    """Return current local time as a short string (for filenames)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Record utilities ──────────────────────────────────────────────────────────

def group_by(records: List[Dict], key: str) -> Dict[str, List[Dict]]:
    """Group a list of records by a key field."""
    groups: Dict[str, List[Dict]] = {}
    for r in records:
        val = str(r.get(key, "unknown"))
        groups.setdefault(val, []).append(r)
    return groups


def flatten_trajectory(trajectory: List[Dict]) -> str:
    """Flatten a trajectory list into a single readable string."""
    lines = []
    for step in trajectory:
        s = step.get("step", "?")
        action = step.get("action", "")
        obs = step.get("observation", "")
        lines.append(f"[Step {s}] Action: {action} | Observation: {obs}")
        if step.get("tool_called"):
            lines.append(f"  Tool: {step['tool_called']}({step.get('tool_params', {})})")
            lines.append(f"  Output: {step.get('tool_output', '')}")
    return "\n".join(lines)


# ── Progress display ──────────────────────────────────────────────────────────

def print_progress(current: int, total: int, prefix: str = "", width: int = 40) -> None:
    """Print a simple text progress bar."""
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix} |{bar}| {current}/{total} ({pct:.1%})", end="", flush=True)
    if current == total:
        print()


if __name__ == "__main__":
    print("AFA Utils - Self Test")
    print(f"Project root: {get_project_root()}")
    print(f"Timestamp: {now_iso()}")
    test = [{"a": 1, "g": "x"}, {"a": 2, "g": "y"}, {"a": 3, "g": "x"}]
    grouped = group_by(test, "g")
    print(f"Group by 'g': x={len(grouped.get('x',[]))}, y={len(grouped.get('y',[]))}")
