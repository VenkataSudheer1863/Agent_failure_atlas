# Failure Taxonomy

This directory contains the Agent Failure Atlas (AFA) taxonomy — a structured classification system for failure modes in autonomous AI agents.

## Overview

The taxonomy organizes agent failures into **8 top-level categories**, each with **4 subcategories**. Every failure observed in an agent trajectory is mapped to one (or more) of these categories.

## Taxonomy Categories

| Category | Subcategories |
|---|---|
| Planning | Missing steps, Wrong ordering, Planning loops, Redundant plans |
| Reasoning | Hallucination, Contradiction, Invalid inference, Unsupported conclusions |
| Tool Use | Wrong tool selection, Parameter errors, API misuse, Parsing failures |
| Memory | Context loss, Goal forgetting, State corruption, Memory hallucination |
| Execution | Infinite loops, Premature termination, Repeated actions, Task abandonment |
| Coordination | Communication breakdown, Role confusion, Deadlocks, Conflicts |
| Safety | Prompt injection, Unsafe actions, Data leakage, Policy violations |
| Alignment | Goal drift, Reward hacking, Specification gaming, Misalignment |

## Files

- `taxonomy.json` — Machine-readable full taxonomy with codes, descriptions, examples
- `taxonomy.md` — Human-readable taxonomy with descriptions and examples
- `taxonomy_schema.json` — JSON schema for validating AFAD entries against taxonomy

## Taxonomy Codes

Each subcategory has a 4-character code used in the AFAD dataset:
- `PLAN-MS` = Planning / Missing Steps
- `PLAN-WO` = Planning / Wrong Ordering
- `PLAN-PL` = Planning / Planning Loops
- `PLAN-RP` = Planning / Redundant Plans
- `REAS-HA` = Reasoning / Hallucination
- `REAS-CO` = Reasoning / Contradiction
- `REAS-II` = Reasoning / Invalid Inference
- `REAS-UC` = Reasoning / Unsupported Conclusions
- `TOOL-WT` = Tool Use / Wrong Tool Selection
- `TOOL-PE` = Tool Use / Parameter Errors
- `TOOL-AM` = Tool Use / API Misuse
- `TOOL-PF` = Tool Use / Parsing Failures
- `MEM-CL` = Memory / Context Loss
- `MEM-GF` = Memory / Goal Forgetting
- `MEM-SC` = Memory / State Corruption
- `MEM-MH` = Memory / Memory Hallucination
- `EXEC-IL` = Execution / Infinite Loops
- `EXEC-PT` = Execution / Premature Termination
- `EXEC-RA` = Execution / Repeated Actions
- `EXEC-TA` = Execution / Task Abandonment
- `COOR-CB` = Coordination / Communication Breakdown
- `COOR-RC` = Coordination / Role Confusion
- `COOR-DL` = Coordination / Deadlocks
- `COOR-CF` = Coordination / Conflicts
- `SAFE-PI` = Safety / Prompt Injection
- `SAFE-UA` = Safety / Unsafe Actions
- `SAFE-DL` = Safety / Data Leakage
- `SAFE-PV` = Safety / Policy Violations
- `ALIG-GD` = Alignment / Goal Drift
- `ALIG-RH` = Alignment / Reward Hacking
- `ALIG-SS` = Alignment / Specification Gaming
- `ALIG-MI` = Alignment / Misalignment

## Usage

Load the taxonomy in Python:

```python
import json
with open("taxonomy/taxonomy.json") as f:
    taxonomy = json.load(f)
```
