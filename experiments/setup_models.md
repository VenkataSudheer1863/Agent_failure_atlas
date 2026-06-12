# Local Model Setup Guide

All models in the Agent Failure Atlas project run **locally** using [Ollama](https://ollama.com). This guide walks through setup for each model.

---

## Step 1: Install Ollama

### Linux / macOS
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
Download and run the installer from: https://ollama.com/download/windows

After installation, start the Ollama service:
```bash
ollama serve
# Or on Windows, the desktop app starts it automatically
```

---

## Step 2: Pull Each Model

| Model in Paper | Ollama Tag | Disk Space | RAM Required |
|---|---|---|---|
| Qwen3-8B | `qwen3:8b` | ~5.2 GB | 8 GB |
| Qwen3-30B | `qwen3:30b-a3b` | ~18 GB | 24 GB |
| DeepSeek-R1-8B | `deepseek-r1:8b` | ~5.0 GB | 8 GB |
| Gemma3-12B | `gemma3:12b` | ~8.1 GB | 12 GB |
| Llama 3.2 | `llama3.2` | ~2.0 GB | 4 GB |
| GPT-OSS-20B | vLLM / custom | ~12 GB | 20 GB |

```bash
# Pull all models (run one at a time or in parallel)
ollama pull qwen3:8b
ollama pull qwen3:30b-a3b
ollama pull deepseek-r1:8b
ollama pull gemma3:12b
ollama pull llama3.2
```

---

## Step 3: Verify Models

```bash
# List all pulled models
ollama list

# Quick test
ollama run qwen3:8b "Say hello in one sentence."
ollama run deepseek-r1:8b "What is 2+2?"
ollama run gemma3:12b "What is the capital of France?"
ollama run llama3.2 "Summarize: The sky is blue."
```

---

## Step 4: GPT-OSS-20B (via vLLM)

If you have a compatible 20B OSS model (e.g., `mistralai/Mistral-7B-Instruct-v0.3` as a substitute, or a true 20B model):

```bash
pip install vllm

# Serve the model on localhost:8000
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --host 0.0.0.0 \
  --port 8000
```

The benchmark runner will use the OpenAI-compatible API at `http://localhost:8000/v1`.

---

## Step 5: Configure Model Endpoints

Edit `experiments/configs/model_configs/<model>.yaml` to point to your local instance.

Example for Qwen3-8B:
```yaml
model_name: Qwen3-8B
ollama_model: qwen3:8b
base_url: http://localhost:11434
temperature: 0.0
max_tokens: 2048
timeout: 120
```

---

## Hardware Recommendations

| Setup | Models You Can Run |
|---|---|
| 8 GB VRAM GPU | Qwen3-8B, DeepSeek-R1-8B, Llama 3.2 |
| 16 GB VRAM GPU | + Gemma3-12B |
| 24 GB VRAM GPU | + Qwen3-30B |
| CPU only (16 GB RAM) | Llama 3.2 (slow) |
| CPU only (32 GB RAM) | All 8B models (slow) |

---

## Troubleshooting

**Ollama not responding:**
```bash
# Check if running
curl http://localhost:11434/api/tags

# Restart service (Linux)
systemctl restart ollama
```

**Out of memory:**
- Use `--num-gpu 0` to run on CPU
- Try quantized variants: `qwen3:8b-q4_0` instead of `qwen3:8b`

**Model pull fails:**
```bash
# Check disk space
df -h

# Retry with explicit registry
ollama pull registry.ollama.ai/library/qwen3:8b
```
