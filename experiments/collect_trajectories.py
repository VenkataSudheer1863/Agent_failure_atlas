"""
collect_trajectories.py

Collect agent trajectories via Groq-hosted models with tool-augmented agentic loop.
All inference is via Groq (https://api.groq.com/openai/v1). No local GPU required.

Usage:
    python experiments/collect_trajectories.py --model GPT-OSS-20B --task-type planning
    python experiments/collect_trajectories.py --all --output experiments/results/raw/

Note: This script delegates to run_benchmark.py for the actual inference.
For most use cases, use run_benchmark.py directly.
"""

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import json
import re
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Benchmark tool registry (simulated) ──────────────────────────────────────

BENCHMARK_TOOLS = {
    "web_search": lambda q: f"[SEARCH RESULT] Top result for '{q}': Retrieved relevant information about {q} from indexed web sources.",
    "calculator": lambda expr: f"[CALC RESULT] {expr} = {_eval_safe(expr)}",
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


def _eval_safe(expr: str) -> str:
    try:
        allowed = set("0123456789+-*/()., ")
        if all(c in allowed for c in expr):
            return str(eval(expr, {"__builtins__": {}}))
        return "Error: Invalid expression"
    except Exception:
        return "Error: Could not evaluate"


def _parse_tool_call(response: str) -> Optional[Dict]:
    match = re.search(r"TOOL:\s*(\w+)\(([^)]*)\)", response)
    if match:
        return {"tool": match.group(1), "args": match.group(2).strip()}
    return None


def _execute_tool(tool_call: Dict) -> str:
    tool_name = tool_call["tool"]
    args = tool_call["args"]
    if tool_name in BENCHMARK_TOOLS:
        try:
            return BENCHMARK_TOOLS[tool_name](args)
        except Exception as e:
            return f"[TOOL ERROR] {tool_name} failed: {str(e)}"
    return f"[TOOL ERROR] Unknown tool: {tool_name}"


def _strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# ── Groq model registry (mirrors run_benchmark.py) ───────────────────────────

MODEL_REGISTRY = {
    "Llama-3.1-8B": {"model_id": "llama-3.1-8b-instant", "tier": "small"},
    "Llama-4-Scout-17B": {"model_id": "meta-llama/llama-4-scout-17b-16e-instruct", "tier": "medium"},
    "Qwen3-32B": {"model_id": "qwen/qwen3-32b", "tier": "reasoning"},
    "Llama-3.3-70B": {"model_id": "llama-3.3-70b-versatile", "tier": "large"},
    "GPT-OSS-20B": {"model_id": "openai/gpt-oss-20b", "tier": "frontier-20B"},
    "GPT-OSS-120B": {"model_id": "openai/gpt-oss-120b", "tier": "frontier-120B"},
}

_groq_clients: dict = {}

_MODEL_KEY_MAP = {
    "Llama-3.1-8B":      "GROQ_API_KEY_1",
    "Llama-4-Scout-17B": "GROQ_API_KEY_2",
    "Qwen3-32B":         "GROQ_API_KEY_3",
    "Llama-3.3-70B":     "GROQ_API_KEY_4",
    "GPT-OSS-20B":       "GROQ_API_KEY_5",
    "GPT-OSS-120B":      "GROQ_API_KEY_6",
}


def _get_groq_client(model_name: str = "Llama-3.1-8B"):
    """Return a cached per-model Groq client using that model's dedicated key."""
    if model_name not in _groq_clients:
        import os
        from openai import OpenAI
        key_env = _MODEL_KEY_MAP.get(model_name, "GROQ_API_KEY_1")
        api_key = os.environ.get(key_env, "") or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(f"No API key found for {model_name} (checked {key_env})")
        _groq_clients[model_name] = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=120,
        )
    return _groq_clients[model_name]


def _call_groq(model_name: str, messages: List[Dict], max_tokens: int = 2048) -> str:
    """Call Groq model and return response text."""
    config = MODEL_REGISTRY[model_name]
    model_id = config["model_id"]
    if "qwen3" in model_id.lower() and max_tokens < 4096:
        max_tokens = 4096

    client = _get_groq_client(model_name)
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    return _strip_thinking(content)


# ── Agentic Loop ──────────────────────────────────────────────────────────────

def run_agentic_loop(
    task: Dict,
    model_name: str,
    max_steps: int = 6,
    temperature: float = 0.0,
) -> Dict:
    """
    Run a full agentic loop with tool use for a task on a Groq-hosted model.

    Handles tool call parsing, simulated tool execution, and multi-step reasoning.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    task_id = task.get("id", "unknown")
    task_prompt = task.get("prompt", "")
    task_type = task.get("task_type", "unknown")
    model_cfg = MODEL_REGISTRY[model_name]

    logger.info(f"  Task {task_id} | {model_name} ({model_cfg['model_id']})")

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
            text = _call_groq(model_name, messages)
        except Exception as e:
            logger.error(f"Model call failed at step {step}: {e}")
            trajectory.append({
                "step": step, "action": "MODEL_ERROR",
                "observation": str(e), "tool_called": None,
                "tool_params": None, "tool_output": None,
            })
            break

        step_record = {
            "step": step,
            "action": text[:400],
            "observation": "",
            "tool_called": None,
            "tool_params": None,
            "tool_output": None,
        }

        # Handle tool call
        tool_call = _parse_tool_call(text)
        if tool_call:
            tool_result = _execute_tool(tool_call)
            step_record.update({
                "tool_called": tool_call["tool"],
                "tool_params": {"args": tool_call["args"]},
                "tool_output": tool_result,
                "observation": tool_result,
            })
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": f"RESULT: {tool_result}\nContinue with the next step.",
            })
        else:
            step_record["observation"] = "No tool called."
            messages.append({"role": "assistant", "content": text})

        trajectory.append(step_record)

        text_lower = text.lower()
        if any(kw in text_lower for kw in [
            "final answer:", "task complete", "done.", "conclusion:",
            "in summary", "the answer is", "therefore,"
        ]):
            outcome = "success"
            break
        if any(kw in text_lower for kw in [
            "cannot", "unable to", "i give up", "impossible", "i'm unable"
        ]):
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
        "provider": "groq",
        "model_id": model_cfg["model_id"],
        "tier": model_cfg["tier"],
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
    max_steps: int = 6,
    inter_request_delay: float = 1.0,
) -> None:
    """Collect tool-use trajectories for all models and tasks via Groq."""
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
                if inter_request_delay > 0:
                    time.sleep(inter_request_delay)
            except Exception as e:
                logger.error(f"  Failed: {e}")

        out_file = model_dir / "trajectories.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        logger.info(f"  Saved {len(results)} trajectories to {out_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    all_models = list(MODEL_REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="Collect agent trajectories via Groq-hosted models"
    )
    parser.add_argument("--model", nargs="+", choices=all_models, help="Model(s) to run")
    parser.add_argument("--all", action="store_true", help="Run all models")
    parser.add_argument("--task-type", nargs="+", help="Task type(s)")
    parser.add_argument("--tasks-dir", default="experiments/tasks")
    parser.add_argument("--output", default="experiments/results/raw/")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Inter-request delay in seconds (rate-limit buffer)")
    args = parser.parse_args()

    models = all_models if args.all else (args.model or ["GPT-OSS-20B"])
    collect_all(
        models=models,
        tasks_dir=args.tasks_dir,
        output_dir=args.output,
        task_types=args.task_type,
        max_steps=args.max_steps,
        inter_request_delay=args.delay,
    )


if __name__ == "__main__":
    main()
