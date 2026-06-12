# Annotation

This directory contains all tools, guidelines, and inter-annotator agreement analysis for labeling agent trajectories with AFA taxonomy codes.

## Contents

| File | Description |
|---|---|
| `annotation_guidelines.md` | Full annotation manual for human labelers |
| `annotator.py` | Python annotation utilities (batch labeling, IAA computation) |
| `iaa_report.md` | Inter-Annotator Agreement (IAA) report |
| `annotation_template.jsonl` | Blank template for annotating new trajectories |

## Annotation Process

1. Read the full taxonomy (`taxonomy/taxonomy.md`)
2. Review the annotation guidelines (`annotation_guidelines.md`)
3. For each trajectory, identify:
   - The primary failure category
   - The most specific subcategory code
   - Root cause (free text)
   - Severity score (1–5)
   - Whether the agent recovered
4. Record annotations in JSONL format matching `taxonomy/taxonomy_schema.py`
5. Run IAA checks with `annotator.py --iaa`

## Inter-Annotator Agreement

We report Cohen's Kappa for:
- **Top-level category**: κ = 0.81 (strong agreement)
- **Subcategory**: κ = 0.74 (substantial agreement)
- **Severity score**: κ = 0.69 (substantial agreement)

See `iaa_report.md` for full breakdown.
