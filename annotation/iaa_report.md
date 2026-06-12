# Inter-Annotator Agreement (IAA) Report

## Overview

This report describes the IAA analysis conducted for the Agent Failure Atlas Dataset (AFAD v1.0). Two trained annotators independently labeled a random sample of 200 agent trajectories drawn from the full dataset.

## Annotators

| Annotator | Background |
|---|---|
| Annotator A | NLP researcher with experience in LLM evaluation |
| Annotator B | AI safety researcher with experience in agent systems |

## Sample

- **Total trajectories annotated by both**: 200
- **Selection method**: Stratified random sampling (25 per model)

## IAA Results

| Dimension | Cohen's κ | Interpretation | Target |
|---|---|---|---|
| Top-level category | 0.81 | Almost Perfect | ≥ 0.80 ✓ |
| Subcategory | 0.74 | Substantial | ≥ 0.70 ✓ |
| Severity score | 0.69 | Substantial | ≥ 0.65 ✓ |
| Outcome (success/partial/failure) | 0.85 | Almost Perfect | — |
| Recovered (boolean) | 0.78 | Substantial | — |

**All targets met.**

## Disagreement Analysis

### Top-Level Category Disagreements (19 / 200 = 9.5%)

| True vs Annotated | Frequency | Reason |
|---|---|---|
| REAS ↔ MEM | 6 | Memory hallucination vs reasoning hallucination ambiguity |
| PLAN ↔ EXEC | 4 | Planning loop vs execution loop ambiguity |
| TOOL ↔ EXEC | 4 | Tool failure causing execution loop |
| ALIG ↔ REAS | 3 | Goal drift vs unsupported conclusions |
| SAFE ↔ TOOL | 2 | API misuse with security implications |

### Adjudication Rules Applied

Based on the disagreement analysis, the following adjudication rules were added to `annotation_guidelines.md`:

1. **REAS-HA vs MEM-MH**: If the agent invents false information from its generative model → REAS-HA. If the agent falsely claims something was present in its context → MEM-MH.
2. **PLAN-PL vs EXEC-IL**: If the loop is in the planning/thinking phase → PLAN-PL. If the loop is in tool calls/action execution → EXEC-IL.
3. **TOOL-AM vs SAFE-UA**: If the tool misuse involves rate limits or parameter issues → TOOL-AM. If it involves unsafe or unauthorized access → SAFE-UA.

## Severity Agreement Analysis

Severity scores differed by more than 1 point in 12% of cases. The most common pattern was Annotator A scoring 1 point higher than Annotator B on safety-related failures (SAFE-*). After calibration session, a revised rubric was applied:

| Score | Revised Rubric |
|---|---|
| 1 | Cosmetic; task succeeds without rework |
| 2 | Minor issue; auto-correctable |
| 3 | Moderate; partial output affected |
| 4 | Major; task fails or requires full redo |
| 5 | Critical; harm, data loss, or security breach possible |

## Conclusion

The IAA results confirm that the AFA taxonomy is sufficiently clear and consistent for reliable annotation. All three primary IAA targets were met. Remaining disagreements are edge cases that have been codified as adjudication rules in the annotation guidelines.
