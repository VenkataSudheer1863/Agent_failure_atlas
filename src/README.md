# src — Core Source Package

This directory contains shared utilities used across all components of the Agent Failure Atlas project.

## Modules

| Module | Description |
|---|---|
| `metrics.py` | Core metric computation functions |
| `models.py` | Model abstraction layer (Ollama, OpenAI-compat) |
| `taxonomy.py` | Taxonomy loading and code resolution utilities |
| `utils.py` | Common utilities (logging, JSON I/O, path handling) |

## Usage

```python
from src.metrics import compute_failure_rate, compute_recovery_rate
from src.models import ModelClient
from src.taxonomy import get_category_for_subcategory
```
