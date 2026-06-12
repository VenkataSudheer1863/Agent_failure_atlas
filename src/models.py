"""
models.py

Model abstraction layer for the Agent Failure Atlas experiments.
Provides a unified interface for calling local Ollama models and
OpenAI-compatible endpoints.

Local models (via Ollama):
- Qwen3-8B          : ollama pull qwen3:8b
- Qwen3-30B-A3B     : ollama pull qwen3:30b-a3b
- DeepSeek-R1-8B    : ollama pull deepseek-r1:8b
- Gemma3-12B        : ollama pull gemma3:12b
- Llama-3.2         : ollama pull llama3.2

Remote / custom endpoint:
- GPT-OSS-20B       : vLLM server at http://localhost:8000/v1
"""

import time
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, Dict] = {
    "GPT-OSS-20B": {
        "backend": "openai_compat",
        "base_url": "http://localhost:8000/v1",
        "model_id": "gpt-oss-20b",
        "description": "GPT-OSS 20B via vLLM. Requires: vLLM server running on port 8000.",
    },
    "Qwen3-8B": {
        "backend": "ollama",
        "model_id": "qwen3:8b",
        "description": "Qwen3 8B by Alibaba Cloud. Install: ollama pull qwen3:8b",
    },
    "Qwen3-30B": {
        "backend": "ollama",
        "model_id": "qwen3:30b-a3b",
        "description": "Qwen3 30B MoE (3B active). Install: ollama pull qwen3:30b-a3b",
    },
    "DeepSeek-R1-8B": {
        "backend": "ollama",
        "model_id": "deepseek-r1:8b",
        "description": "DeepSeek-R1 distilled 8B. Install: ollama pull deepseek-r1:8b",
    },
    "Gemma3-12B": {
        "backend": "ollama",
        "model_id": "gemma3:12b",
        "description": "Google Gemma 3 12B. Install: ollama pull gemma3:12b",
    },
    "Llama-3.2": {
        "backend": "ollama",
        "model_id": "llama3.2",
        "description": "Meta Llama 3.2 3B. Install: ollama pull llama3.2",
    },
}


# ── Model Client ──────────────────────────────────────────────────────────────

class ModelClient:
    """
    Unified client for calling local and remote language models.
    
    Usage:
        client = ModelClient("Qwen3-8B")
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
        self._validate_backend()

    def _validate_backend(self) -> None:
        """Check that the required backend library is available."""
        backend = self.config["backend"]
        if backend == "ollama":
            try:
                import ollama  # noqa
            except ImportError:
                raise ImportError(
                    "The 'ollama' Python package is required for local models.\n"
                    "Install it with: pip install ollama\n"
                    f"Then pull the model: {self.config['description']}"
                )
        elif backend == "openai_compat":
            try:
                import requests  # noqa
            except ImportError:
                raise ImportError("The 'requests' package is required: pip install requests")

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """
        Send a chat message to the model and return the response text.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Override default parameters (temperature, max_tokens, etc.)
            
        Returns:
            Response text as a string
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        backend = self.config["backend"]

        if backend == "ollama":
            return self._call_ollama(messages, temperature, max_tokens)
        elif backend == "openai_compat":
            return self._call_openai_compat(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _call_ollama(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Call an Ollama local model."""
        import ollama as ollama_client

        retries = 3
        for attempt in range(retries):
            try:
                response = ollama_client.chat(
                    model=self.config["model_id"],
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "seed": self.seed,
                    },
                )
                return response["message"]["content"]
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(f"Ollama call failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)

    def _call_openai_compat(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Call an OpenAI-compatible API endpoint (e.g., vLLM)."""
        import requests

        payload = {
            "model": self.config["model_id"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": self.seed,
        }
        resp = requests.post(
            f"{self.config['base_url']}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def is_available(self) -> bool:
        """Check if the model is reachable and available."""
        backend = self.config["backend"]
        if backend == "ollama":
            try:
                import requests
                resp = requests.get("http://localhost:11434/api/tags", timeout=5)
                if resp.ok:
                    tags = [m["name"] for m in resp.json().get("models", [])]
                    return any(self.config["model_id"] in tag for tag in tags)
                return False
            except Exception:
                return False
        elif backend == "openai_compat":
            try:
                import requests
                resp = requests.get(f"{self.config['base_url']}/models", timeout=5)
                return resp.ok
            except Exception:
                return False
        return False

    def __repr__(self) -> str:
        return f"ModelClient(model={self.model_name}, backend={self.config['backend']})"


# ── Availability check ────────────────────────────────────────────────────────

def check_all_models() -> None:
    """Print availability status for all models in the registry."""
    print(f"\n{'Model':<20} {'Backend':<15} {'Available':<12} {'Description'}")
    print("-" * 80)
    for name, config in MODEL_REGISTRY.items():
        try:
            client = ModelClient(name)
            available = "✓" if client.is_available() else "✗"
        except Exception:
            available = "✗ (pkg missing)"
        desc_short = config["description"].split(". ")[0]
        print(f"{name:<20} {config['backend']:<15} {available:<12} {desc_short}")


if __name__ == "__main__":
    print("Agent Failure Atlas — Model Registry")
    check_all_models()
