"""
models.py

Model abstraction layer for the Agent Failure Atlas experiments.
All models served exclusively via Groq (https://api.groq.com/openai/v1).

Model registry (verified available 2026-06-15):
  Llama-3.1-8B     → llama-3.1-8b-instant            (small tier)       GROQ_API_KEY_1
  Llama-4-Scout-17B → meta-llama/llama-4-scout-17b-16e-instruct (medium) GROQ_API_KEY_2
  Qwen3-32B        → qwen/qwen3-32b                   (reasoning tier)   GROQ_API_KEY_3
  Llama-3.3-70B    → llama-3.3-70b-versatile          (large tier)       GROQ_API_KEY_4
  GPT-OSS-20B      → openai/gpt-oss-20b               (frontier-20B)     GROQ_API_KEY_5
  GPT-OSS-120B     → openai/gpt-oss-120b              (frontier-120B)    GROQ_API_KEY_6
  Judge            → llama-3.1-8b-instant              (uses model's key)

Required environment variables (one per model):
  GROQ_API_KEY_1 … GROQ_API_KEY_6 — see .env.example
"""

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import os
import time
import logging
from typing import List, Dict, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq provider endpoint (single provider — Groq only)
# ---------------------------------------------------------------------------

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ---------------------------------------------------------------------------
# Model registry — all Groq-hosted, verified available 2026-06-15
# Each model has its own dedicated API key env var.
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict] = {
    "Llama-3.1-8B": {
        "model_id": "llama-3.1-8b-instant",
        "api_key_env": "GROQ_API_KEY_1",
        "tier": "small",
        "parameters": "8B",
        "description": "Llama 3.1 8B — small, ultra-low-latency model on Groq LPU.",
    },
    "Llama-4-Scout-17B": {
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "api_key_env": "GROQ_API_KEY_2",
        "tier": "medium",
        "parameters": "17B (16 experts MoE)",
        "description": "Llama 4 Scout 17B MoE — medium tier, new-generation architecture.",
    },
    "Qwen3-32B": {
        "model_id": "qwen/qwen3-32b",
        "api_key_env": "GROQ_API_KEY_3",
        "tier": "reasoning",
        "parameters": "32B",
        "description": "Qwen3 32B — reasoning-capable model with chain-of-thought <think> traces.",
    },
    "Llama-3.3-70B": {
        "model_id": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY_4",
        "tier": "large",
        "parameters": "70B",
        "description": "Llama 3.3 70B — large versatile model on Groq.",
    },
    "GPT-OSS-20B": {
        "model_id": "openai/gpt-oss-20b",
        "api_key_env": "GROQ_API_KEY_5",
        "tier": "frontier-20B",
        "parameters": "~20B",
        "description": "GPT-OSS-20B — frontier-20B tier, available on Groq.",
    },
    "GPT-OSS-120B": {
        "model_id": "openai/gpt-oss-120b",
        "api_key_env": "GROQ_API_KEY_6",
        "tier": "frontier-120B",
        "parameters": "~120B",
        "description": "GPT-OSS-120B — largest available on Groq; frontier capability ceiling.",
    },
    "Judge": {
        "model_id": "llama-3.1-8b-instant",
        "api_key_env": "GROQ_API_KEY_1",
        "tier": "judge",
        "parameters": "8B",
        "description": "Llama 3.1 8B — fast, low-cost judge for automated failure labeling.",
    },
}


# ---------------------------------------------------------------------------
# Model Client
# ---------------------------------------------------------------------------

class ModelClient:
    """
    Unified client for calling Groq-hosted language models.

    Uses the OpenAI-compatible chat completions interface.
    Includes retry logic with exponential backoff.

    Usage:
        client = ModelClient("Qwen3-32B")
        response = client.chat([{"role": "user", "content": "Hello"}])
        print(response)
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 120,
        seed: int = 42,
        max_retries: int = 3,
    ):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: '{model_name}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        self.model_name = model_name
        self.config = MODEL_REGISTRY[model_name]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.seed = seed
        self.max_retries = max_retries

        key_env = self.config.get("api_key_env", "GROQ_API_KEY_1")
        api_key = os.environ.get(key_env, "") or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            logger.warning(f"No API key found for {model_name} (checked {key_env})")

        self._client = OpenAI(
            api_key=api_key,
            base_url=_GROQ_BASE_URL,
            timeout=timeout,
        )

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """
        Send a chat message and return the response text.

        For Qwen3-32B, strips <think>...</think> reasoning traces from output.

        Args:
            messages: List of dicts with 'role' and 'content' keys.
            **kwargs: Override temperature, max_tokens.

        Returns:
            Response text as a string.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        model_id = self.config["model_id"]

        # Qwen3-32B reasoning model requires extra tokens for <think> trace
        if "qwen3" in model_id.lower() and max_tokens < 4096:
            max_tokens = 4096

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""
                # Strip chain-of-thought traces from reasoning models
                content = _strip_thinking(content)
                return content
            except Exception as exc:
                msg = str(exc)
                wait = 2 ** attempt
                if "429" in msg or "rate limit" in msg.lower() or "quota" in msg.lower():
                    logger.warning(
                        f"Rate limit on {self.model_name} "
                        f"(attempt {attempt+1}). Retrying in {wait}s..."
                    )
                else:
                    logger.warning(
                        f"API call failed for {self.model_name} (attempt {attempt+1}): {exc}. "
                        f"Retrying in {wait}s..."
                    )
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(wait)

        raise RuntimeError(
            f"All {self.max_retries} attempts failed for {self.model_name}. "
            f"Last error: {last_error}"
        )

    def is_available(self) -> bool:
        """Check if this model's Groq API key is configured."""
        key_env = self.config.get("api_key_env", "GROQ_API_KEY_1")
        return bool(os.environ.get(key_env, "") or os.environ.get("GROQ_API_KEY", ""))

    def benchmark_metadata(self) -> dict:
        """Return benchmark metadata for result recording."""
        return {
            "provider": "groq",
            "model_id": self.config["model_id"],
            "tier": self.config.get("tier", ""),
        }

    def __repr__(self) -> str:
        return (
            f"ModelClient(model={self.model_name}, "
            f"groq_id={self.config['model_id']}, "
            f"tier={self.config['tier']})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> chain-of-thought traces from model output."""
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_all_models() -> None:
    """Print availability status for all models in the registry."""
    print(f"\n{'Label':<22} {'Key Env':<20} {'Status':<8} {'Groq Model ID':<50} {'Tier'}")
    print("-" * 120)
    for name, config in MODEL_REGISTRY.items():
        if name == "Judge":
            continue
        key_env = config.get("api_key_env", "?")
        key_set = bool(os.environ.get(key_env, ""))
        status = "OK" if key_set else "MISSING"
        print(
            f"{name:<22} {key_env:<20} {status:<8} {config['model_id']:<50} {config['tier']}"
        )
    judge = MODEL_REGISTRY["Judge"]
    key_env = judge.get("api_key_env", "GROQ_API_KEY_1")
    key_set = bool(os.environ.get(key_env, ""))
    status = "OK" if key_set else "MISSING"
    print(f"\n{'Judge':<22} {key_env:<20} {status:<8} {judge['model_id']}")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        from pathlib import Path as _Path
        load_dotenv(_Path(__file__).parent.parent / ".env")
    except ImportError:
        pass
    print("Agent Failure Atlas — Groq Model Registry")
    check_all_models()
