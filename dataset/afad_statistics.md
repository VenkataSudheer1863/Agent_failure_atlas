# AFAD Dataset Statistics — v1.0

This document reports distribution and coverage statistics for the Agent Failure Atlas Dataset (AFAD v1.0), based on 1,000 annotated agent trajectories collected across 6 open-source language models and 5 task types.

---

## Overview

| Metric | Value |
|---|---|
| Total records | 1,000 |
| Total models | 6 |
| Total task types | 5 |
| Total failure categories | 8 |
| Total subcategories covered | 32 / 32 |
| Average trajectory length (steps) | 5.2 |
| Overall strict failure rate | 58.8% (outcome = failure) |
| Overall broad failure rate | 79.9% (failure + partial) |
| Overall recovery rate | 11.5% |
| Mean severity score | 3.58 / 5.0 |

---

## Records by Model

| Model | Records | Strict Failure Rate | Recovery Rate | Mean Severity |
|---|---|---|---|---|
| Gemma3-12B | 177 | 40.1% | 22.4% | 3.41 |
| DeepSeek-R1-8B | 164 | 47.6% | 17.5% | 3.41 |
| Qwen3-30B | 157 | 54.8% | 12.6% | 3.46 |
| Qwen3-8B | 176 | 66.5% | 8.5% | 3.67 |
| Llama-3.2 | 163 | 76.7% | 6.5% | 3.69 |
| GPT-OSS-20B | 163 | 68.1% | 5.1% | 3.82 |
| **Total / Mean** | **1,000** | **58.8%** | **11.5%** | **3.58** |

Chi-square test of independence on failure rates across models: χ²(5) = 66.74, **p < 0.001** (significant).  
Kruskal-Wallis test on severity distributions: H(5) = 32.01, **p < 0.001** (significant).

---

## Records by Task Type

| Task Type | Records | % |
|---|---|---|
| Reasoning | 251 | 25.1% |
| Tool Use | 240 | 24.0% |
| Planning | 204 | 20.4% |
| Information Seeking | 185 | 18.5% |
| Multi-Agent | 120 | 12.0% |

---

## Records by Failure Category

| Category | Code | Count | % |
|---|---|---|---|
| Reasoning | REAS | 148 | 14.8% |
| Memory | MEM | 140 | 14.0% |
| Alignment | ALIG | 127 | 12.7% |
| Execution | EXEC | 121 | 12.1% |
| Coordination | COOR | 120 | 12.0% |
| Safety | SAFE | 120 | 12.0% |
| Planning | PLAN | 118 | 11.8% |
| Tool Use | TOOL | 106 | 10.6% |

---

## Top 15 Subcategories by Frequency

| Rank | Code | Subcategory | Count |
|---|---|---|---|
| 1 | REAS-HA | Hallucination | 46 |
| 2 | MEM-MH | Memory Hallucination | 41 |
| 3 | REAS-CO | Contradiction | 40 |
| 4 | ALIG-MI | Misalignment | 40 |
| 5 | REAS-UC | Unsupported Conclusions | 38 |
| 6 | MEM-SC | State Corruption | 38 |
| 7 | COOR-CB | Communication Breakdown | 36 |
| 8 | TOOL-AM | API Misuse | 35 |
| 9 | EXEC-TA | Task Abandonment | 35 |
| 10 | MEM-GF | Goal Forgetting | 34 |
| 11 | PLAN-PL | Planning Loops | 33 |
| 12 | SAFE-PV | Policy Violations | 32 |
| 13 | SAFE-DL | Data Leakage | 32 |
| 14 | ALIG-SS | Specification Gaming | 31 |
| 15 | SAFE-PI | Prompt Injection | 30 |

---

## Outcome Distribution

| Outcome | Count | % |
|---|---|---|
| failure | 588 | 58.8% |
| partial | 211 | 21.1% |
| success | 201 | 20.1% |

- **Recovery rate** (recovered / (failure + partial)): 92 / 799 = **11.5%**

---

## Severity Score Distribution

| Score | Count | % | Interpretation |
|---|---|---|---|
| 1 | 11 | 1.1% | Minor / cosmetic |
| 2 | 109 | 10.9% | Low impact |
| 3 | 313 | 31.3% | Moderate impact |
| 4 | 427 | 42.7% | High impact |
| 5 | 140 | 14.0% | Critical |

Mean severity: **3.58** | Median: **4** | Severity ≥ 4: **56.7%** of all records

---

## Recovery Rates by Category

| Category | Theoretical Recoverability | Observed Recovery Rate |
|---|---|---|
| PLAN | Partial | 22.3% |
| REAS | Partial | 16.9% |
| TOOL | Yes | 16.5% |
| COOR | Partial | 12.6% |
| ALIG | Partial | 13.1% |
| MEM | Partial | 5.4% |
| EXEC | Rarely | 6.1% |
| SAFE | No | 0.0% |

Safety failures (SAFE-*) achieve a **0% recovery rate** across all models, consistent with their non-recoverable classification in the taxonomy.

---

## Dataset Splits

| Split | Records | % |
|---|---|---|
| Train | 700 | 70% |
| Validation | 150 | 15% |
| Test | 150 | 15% |

Splits are stratified by outcome and model to preserve class balance across partitions.
See `splits/` directory for pre-split files (`train.jsonl`, `val.jsonl`, `test.jsonl`).

---

## Inter-Annotator Agreement

Two expert annotators independently labeled 200 randomly sampled trajectories (25 per model). Disagreements resolved by third-annotator adjudication.

| Dimension | Cohen's κ | Interpretation | Target | Status |
|---|---|---|---|---|
| Top-level category | 0.81 | Almost Perfect | ≥ 0.80 | ✓ Met |
| Subcategory | 0.74 | Substantial | ≥ 0.70 | ✓ Met |
| Severity score | 0.69 | Substantial | ≥ 0.65 | ✓ Met |
| Outcome | 0.85 | Almost Perfect | — | — |
