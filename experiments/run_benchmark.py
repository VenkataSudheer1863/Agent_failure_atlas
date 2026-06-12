"""
run_benchmark.py

Main benchmark runner for the Agent Failure Atlas experiments.
Runs all benchmark tasks against configured local models via Ollama,
collects full trajectories, and saves results for analysis.

Usage:
    python experiments/run_benchmark.py
    python experiments/run_benchmark.py --model qwen3:8b --tasks planning
    python experiments/run_benchmark.py --config experiments/configs/benchmark_config.yaml
"""

import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timezone

# Try to import ollama; gracefully handle missing install
try:
    import ollama as ollama_client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[WARNING] 'ollama' package not installed. Run: pip install ollama")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "GPT-OSS-20B": {
        "backend": "openai_compat",
        "base_url": "http://localhost:8000/v1",
        "model_id": "gpt-oss-20b",
    },
    "Qwen3-8B": {
        "backend": "ollama",
        "model_id": "qwen3:8b",
    },
    "Qwen3-30B": {
        "backend": "ollama",
        "model_id": "qwen3:30b-a3b",
    },
    "DeepSeek-R1-8B": {
        "backend": "ollama",
        "model_id": "deepseek-r1:8b",
    },
    "Gemma3-12B": {
        "backend": "ollama",
        "model_id": "gemma3:12b",
    },
    "Llama-3.2": {
        "backend": "ollama",
        "model_id": "llama3.2",
    },
}

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an autonomous AI agent. For each task:
1. Think step-by-step.
2. Break the task into subtasks.
3. Use available tools when needed.
4. Report your final answer clearly.
Always reason carefully before acting."""

FAILURE_DETECT_PROMPT = """Review this agent trajectory and identify if any failure occurred.

Trajectory:
{trajectory}

Task: {task}

Did the agent fail? If yes, categorize the failure as one of:
PLAN, REAS, TOOL, MEM, EXEC, COOR, SAFE, ALIG

Respond in JSON:
{{
  "failed": true/false,
  "failure_label": "<category or null>",
  "failure_subcategory": "<subcategory code or null>",
  "root_cause": "<brief description or null>",
  "severity_score": <1-5 or null>,
  "recovered": true/false,
  "outcome": "success/partial/failure"
}}"""

# ── Ollama Backend ────────────────────────────────────────────────────────────

def call_ollama(model_id: str, messages: List[Dict], temperature: float = 0.0, max_tokens: int = 2048) -> str:
    """Call a local Ollama model and return the response text."""
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("ollama package not installed. Run: pip install ollama")

    try:
        response = ollama_client.chat(
            model=model_id,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "seed": 42,
            }
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Ollama call failed for model {model_id}: {e}")
        raise


def call_openai_compat(base_url: str, model_id: str, messages: List[Dict], temperature: float = 0.0) -> str:
    """Call an OpenAI-compatible endpoint (e.g., vLLM)."""
    try:
        import requests
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        }
        resp = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"OpenAI-compat call failed: {e}")
        raise


def call_model(model_name: str, messages: List[Dict], temperature: float = 0.0) -> str:
    """Dispatch to the correct backend for a given model name."""
    config = MODEL_REGISTRY.get(model_name)
    if config is None:
        raise ValueError(f"Unknown model: {model_name}")

    if config["backend"] == "ollama":
        return call_ollama(config["model_id"], messages, temperature)
    elif config["backend"] == "openai_compat":
        return call_openai_compat(config["base_url"], config["model_id"], messages, temperature)
    else:
        raise ValueError(f"Unknown backend: {config['backend']}")


# ── Task Loading ──────────────────────────────────────────────────────────────

def load_tasks(tasks_dir: str, task_types: Optional[List[str]] = None) -> List[Dict]:
    """Load benchmark tasks from JSONL files in tasks_dir."""
    tasks_path = Path(tasks_dir)
    all_tasks = []
    valid_types = ["information_seeking", "tool_use", "planning", "reasoning", "multi_agent"]
    types_to_load = task_types if task_types else valid_types

    for t in types_to_load:
        filepath = tasks_path / f"{t}.jsonl"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        task = json.loads(line)
                        task["task_type"] = t
                        all_tasks.append(task)
        else:
            logger.warning(f"Task file not found: {filepath}")
    return all_tasks


# ── Trajectory Collection ─────────────────────────────────────────────────────

def run_task_on_model(task: Dict, model_name: str, max_steps: int = 8) -> Dict:
    """
    Run a single benchmark task on a model and collect the trajectory.

    Returns a trajectory record (not yet labeled — labeling happens in evaluate.py).
    """
    task_id = task.get("id", "UNKNOWN")
    task_type = task.get("task_type", "unknown")
    task_prompt = task.get("prompt", "")

    logger.info(f"  Task {task_id} | Model {model_name}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]

    trajectory = []
    outcome = "failure"
    start_time = time.time()

    try:
        for step in range(1, max_steps + 1):
            response = call_model(model_name, messages)
            step_record = {
                "step": step,
                "action": f"Agent response at step {step}",
                "observation": response[:500],  # Truncate for storage
            }
            trajectory.append(step_record)
            messages.append({"role": "assistant", "content": response})

            # Simple termination heuristics
            response_lower = response.lower()
            if any(kw in response_lower for kw in ["final answer:", "task complete", "done:", "conclusion:"]):
                outcome = "success"
                break
            if any(kw in response_lower for kw in ["cannot complete", "unable to", "i give up"]):
                outcome = "failure"
                break
            # Continue to next step
            messages.append({"role": "user", "content": "Continue. What is your next step?"})

    except Exception as e:
        logger.error(f"Error running task {task_id} on {model_name}: {e}")
        trajectory.append({"step": len(trajectory) + 1, "action": "ERROR", "observation": str(e)})
        outcome = "failure"

    elapsed = time.time() - start_time

    return {
        "task_id": task_id,
        "task_type": task_type,
        "model": model_name,
        "trajectory": trajectory,
        "outcome": outcome,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Main Runner ───────────────────────────────────────────────────────────────

def run_benchmark(
    models: Optional[List[str]] = None,
    task_types: Optional[List[str]] = None,
    tasks_dir: str = "experiments/tasks",
    output_dir: str = "experiments/results/raw",
    dry_run: bool = False,
) -> None:
    """
    Run the full benchmark for all specified models and task types.

    Args:
        models: List of model names to evaluate (default: all)
        task_types: List of task type names to run (default: all)
        tasks_dir: Directory containing task JSONL files
        output_dir: Directory to save raw trajectory results
        dry_run: If True, skip actual model calls and save dummy trajectories
    """
    all_models = models or list(MODEL_REGISTRY.keys())
    tasks = load_tasks(tasks_dir, task_types)
    logger.info(f"Loaded {len(tasks)} tasks across {len(set(t['task_type'] for t in tasks))} task types")
    logger.info(f"Running {len(all_models)} models: {all_models}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_name in all_models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*60}")

        model_results = []
        model_dir = output_path / model_name.replace(" ", "_").replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        output_file = model_dir / "trajectories.jsonl"

        for i, task in enumerate(tasks):
            logger.info(f"[{i+1}/{len(tasks)}] Running task {task.get('id', '?')}")

            if dry_run:
                # Pipeline validation run — no model calls
                result = {
                    "task_id": task.get("id", f"TASK-{i}"),
                    "task_type": task.get("task_type", "unknown"),
                    "model": model_name,
                    "trajectory": [
                        {"step": 1, "action": "Plan the task", "observation": "Plan created."},
                        {"step": 2, "action": "Execute step 1", "observation": "Completed."},
                    ],
                    "outcome": "success",
                    "elapsed_seconds": 0.1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                result = run_task_on_model(task, model_name)

            model_results.append(result)

        # Save results for this model
        with open(output_file, "w", encoding="utf-8") as f:
            for r in model_results:
                f.write(json.dumps(r) + "\n")
        logger.info(f"Saved {len(model_results)} trajectories to {output_file}")

    logger.info(f"\nBenchmark complete. Results in: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="AFA Benchmark Runner")
    parser.add_argument(
        "--model", nargs="+",
        help="Model(s) to run (default: all). E.g. --model Qwen3-8B Llama-3.2"
    )
    parser.add_argument(
        "--tasks", nargs="+",
        choices=["information_seeking", "tool_use", "planning", "reasoning", "multi_agent"],
        help="Task type(s) to run (default: all)"
    )
    parser.add_argument(
        "--tasks-dir", default="experiments/tasks",
        help="Directory with task JSONL files"
    )
    parser.add_argument(
        "--output-dir", default="experiments/results/raw",
        help="Directory to save trajectory results"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip model calls; generate dummy trajectories for testing"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        models=args.model,
        task_types=args.tasks,
        tasks_dir=args.tasks_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
