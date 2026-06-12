"""
collect_trajectories.py

Collect agent trajectories from live local model runs using Ollama.
This script runs a configurable agentic loop and saves full step-by-step trajectories.

Supports:
- Single-model, single-task runs
- Batch runs across all models and tasks
- Tool execution (benchmark tool calls for evaluation tasks)

Usage:
    python experiments/collect_trajectories.py --model Qwen3-8B --task-type planning
    python experiments/collect_trajectories.py --all --output experiments/results/raw/
"""

import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Benchmark tool registry ───────────────────────────────────────────────────

BENCHMARK_TOOLS = {
    "web_search": lambda q: f"[SEARCH RESULT] Top result for '{q}': Retrieved relevant information about {q} from indexed web sources.",
    "calculator": lambda expr: f"[CALC RESULT] {expr} = {eval_safe(expr)}",
    "code_execute": lambda code: f"[EXEC RESULT] Code executed successfully. Output: [evaluated: {code[:50]}]",
    "api_call": lambda endpoint: f"[API RESULT] {{\"status\": 200, \"data\": \"response for {endpoint}\"}}",
    "read_file": lambda path: f"[FILE CONTENT] Contents of {path}: [file content retrieved]",
    "write_file": lambda args: f"[FILE WRITE] Successfully wrote to {args}",
    "database_query": lambda query: f"[DB RESULT] Query '{query}' returned 5 rows: [result set]",
}

TOOL_INJECTION_PROMPT = """Available tools:
- web_search(query): Search the web
- calculator(expression): Evaluate math expressions
- code_execute(code): Execute code
- api_call(endpoint): Call an API endpoint
- read_file(path): Read a file
- write_file(path, content): Write to a file
- database_query(sql): Query a database

To use a tool, write: TOOL: <tool_name>(<args>)
The result will be provided as: RESULT: <output>
"""


def eval_safe(expr: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        allowed = set("0123456789+-*/()., ")
        if all(c in allowed for c in expr):
            return str(eval(expr, {"__builtins__": {}}))
        return "Error: Invalid expression"
    except Exception:
        return "Error: Could not evaluate"


def parse_tool_call(response: str) -> Optional[Dict]:
    """Parse a TOOL: <name>(<args>) call from model output."""
    import re
    match = re.search(r"TOOL:\s*(\w+)\(([^)]*)\)", response)
    if match:
        return {"tool": match.group(1), "args": match.group(2).strip()}
    return None


def execute_tool(tool_call: Dict) -> str:
    """Execute a benchmark tool call and return the result."""
    tool_name = tool_call["tool"]
    args = tool_call["args"]
    if tool_name in BENCHMARK_TOOLS:
        try:
            result = BENCHMARK_TOOLS[tool_name](args)
        except Exception as e:
            result = f"[TOOL ERROR] {tool_name} failed: {str(e)}"
    else:
        result = f"[TOOL ERROR] Unknown tool: {tool_name}"
    return result


# ── Agentic Loop ──────────────────────────────────────────────────────────────

def run_agentic_loop(
    task: Dict,
    model_name: str,
    max_steps: int = 8,
    temperature: float = 0.0,
) -> Dict:
    """
    Run a full agentic loop for a task on a given model.
    Handles tool calls and multi-step reasoning.
    """
    try:
        import ollama as ollama_client
    except ImportError:
        raise RuntimeError("ollama package required. Run: pip install ollama")

    # Get Ollama model ID
    MODEL_IDS = {
        "GPT-OSS-20B": None,  # Not via Ollama
        "Qwen3-8B": "qwen3:8b",
        "Qwen3-30B": "qwen3:30b-a3b",
        "DeepSeek-R1-8B": "deepseek-r1:8b",
        "Gemma3-12B": "gemma3:12b",
        "Llama-3.2": "llama3.2",
    }
    ollama_model = MODEL_IDS.get(model_name)
    if not ollama_model:
        raise ValueError(f"No Ollama ID for model: {model_name}")

    task_id = task.get("id", "unknown")
    task_prompt = task.get("prompt", "")
    task_type = task.get("task_type", "unknown")

    system_msg = (
        "You are an autonomous AI agent. Solve the task step by step.\n"
        + TOOL_INJECTION_PROMPT
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": task_prompt},
    ]

    trajectory = []
    outcome = "failure"
    start_time = time.time()

    for step in range(1, max_steps + 1):
        try:
            response = ollama_client.chat(
                model=ollama_model,
                messages=messages,
                options={"temperature": temperature, "seed": 42},
            )
            text = response["message"]["content"]
        except Exception as e:
            logger.error(f"Model call failed at step {step}: {e}")
            trajectory.append({
                "step": step,
                "action": "MODEL_ERROR",
                "observation": str(e),
            })
            break

        step_record = {
            "step": step,
            "action": text[:300],
            "observation": "",
            "tool_called": None,
            "tool_params": None,
            "tool_output": None,
        }

        # Check for tool call
        tool_call = parse_tool_call(text)
        if tool_call:
            tool_result = execute_tool(tool_call)
            step_record["tool_called"] = tool_call["tool"]
            step_record["tool_params"] = {"args": tool_call["args"]}
            step_record["tool_output"] = tool_result
            step_record["observation"] = tool_result
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"RESULT: {tool_result}\nContinue with the next step."})
        else:
            step_record["observation"] = "No tool called."
            messages.append({"role": "assistant", "content": text})

        trajectory.append(step_record)

        # Termination check
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["final answer:", "task complete", "done.", "conclusion:"]):
            outcome = "success"
            break
        elif any(kw in text_lower for kw in ["cannot", "unable to", "i give up", "impossible"]):
            outcome = "failure"
            break

        if step < max_steps:
            messages.append({"role": "user", "content": "Continue. What is your next step?"})

    elapsed = time.time() - start_time

    return {
        "id": f"TRAJ-{task_id}-{model_name.replace(' ', '_')}",
        "task_id": task_id,
        "task_type": task_type,
        "model": model_name,
        "trajectory": trajectory,
        "n_steps": len(trajectory),
        "outcome": outcome,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Batch Collection ──────────────────────────────────────────────────────────

def collect_all(
    models: List[str],
    tasks_dir: str,
    output_dir: str,
    task_types: Optional[List[str]] = None,
    max_steps: int = 8,
) -> None:
    """Collect trajectories for all models and tasks."""
    from experiments.run_benchmark import load_tasks
    tasks = load_tasks(tasks_dir, task_types)
    logger.info(f"Loaded {len(tasks)} tasks")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        logger.info(f"\nCollecting for model: {model_name}")
        results = []
        model_dir = out / model_name.replace(" ", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        for i, task in enumerate(tasks):
            logger.info(f"  [{i+1}/{len(tasks)}] Task {task.get('id')}")
            try:
                result = run_agentic_loop(task, model_name, max_steps)
                results.append(result)
            except Exception as e:
                logger.error(f"  Failed: {e}")

        out_file = model_dir / "trajectories.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        logger.info(f"  Saved {len(results)} trajectories to {out_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ALL_MODELS = ["Qwen3-8B", "Qwen3-30B", "DeepSeek-R1-8B", "Gemma3-12B", "Llama-3.2"]

    parser = argparse.ArgumentParser(description="Collect agent trajectories from local models")
    parser.add_argument("--model", nargs="+", choices=ALL_MODELS, help="Model(s) to run")
    parser.add_argument("--all", action="store_true", help="Run all models")
    parser.add_argument("--task-type", nargs="+", help="Task type(s) to collect")
    parser.add_argument("--tasks-dir", default="experiments/tasks")
    parser.add_argument("--output", default="experiments/results/raw/")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    models = ALL_MODELS if args.all else (args.model or ["Qwen3-8B"])
    collect_all(
        models=models,
        tasks_dir=args.tasks_dir,
        output_dir=args.output,
        task_types=args.task_type,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
