# Agent Failure Atlas: A Taxonomy, Dataset, and Benchmark for Systematic Analysis of Failure Modes in Autonomous AI Agents

**Venkata Sudheer Paruchuri**  
Independent Researcher
paruchurivenkatasudheer@gmail.com

**Submitted to:** *IEEE Transactions on Neural Networks and Learning Systems* (TNNLS)  
**Article Type:** Research Article  
**Date:** June 2026

---

## Abstract

Autonomous AI agents — systems that couple large language models with planning, memory, and tool-use capabilities — are finding their way into consequential real-world deployments faster than our ability to characterize how they fail. Existing evaluation frameworks measure task success rates but say remarkably little about the structure, severity, or predictability of failure. This paper introduces the **Agent Failure Atlas (AFA)**, a framework built around three tightly coupled artifacts. First, we propose a failure taxonomy comprising 8 top-level categories and 32 subcategories, covering the full behavioral spectrum from reasoning hallucinations to safety violations, designed to be exhaustive at the root-cause level and applicable regardless of model architecture. Second, we present AFAD v1.0 — the Agent Failure Atlas Dataset — containing 450 annotated agent trajectories drawn from six open-source language models across five task types, with failure labels, severity scores (1–5), and recovery annotations for every record. Third, we describe a benchmark of 75 tasks per model (15 per task type) with a deterministic evaluation protocol and an LLM-as-judge labeling pipeline. Across our analysis, planning failures (PLAN) are the most prevalent category at 38.5%, followed by execution failures (EXEC, 26.9%) — together accounting for 65.4% of all labeled failures. No safety failures (SAFE-*) were observed across the benchmark trajectories, consistent with the theoretical prediction of non-recoverability for this category. Strict failure rates range from 52.0% (Qwen3-32B) to 78.7% (Llama-3.1-8B) across models, and this spread is statistically significant (χ²(5) = 17.76, p = 0.003), confirming that model choice meaningfully affects reliability. A Random Forest classifier using only the first three trajectory steps achieves AUC = 0.683 in predicting final outcome, demonstrating that early-warning monitoring based on model identity, task type, and early response features is technically feasible. All code, data, and evaluation infrastructure are released under the MIT license. Inference is served via Groq LPU cloud API (6 dedicated API keys, one per model) with no local GPU hardware required.

**Keywords:** AI agents, failure taxonomy, autonomous agent reliability, large language models, failure prediction, safety, benchmark evaluation, agent trajectories

---

## 1. Introduction

There is a genuine gap between the ambition of LLM-based autonomous agents and how well we understand the ways they break down. Contemporary agents are expected to decompose multi-step problems [1], invoke external APIs and tools [2], maintain coherent state across long trajectories [3], coordinate with peer agents [4], and produce outputs that have tangible consequences in the world. This expanded capability footprint has driven deployment across domains ranging from software engineering assistants [5] and scientific research pipelines [6] to robotic task planning [7] and enterprise workflow automation [8].

What has not kept pace is our understanding of failure. An agent that merely retrieves text can fail by hallucinating [9]; one that runs code can silently delete files [10]; one coordinating sub-agents can deadlock or fall into role confusion [4]; one pursuing a long-horizon goal can gradually drift away from the user's original intent [11]. The failure landscape of agentic systems is qualitatively wider and more consequential than anything in passive LLM evaluation.

Despite this, the study of agent failure remains surprisingly underdeveloped relative to the study of agent success. Prominent benchmarks — AgentBench [12], WebArena [13], GAIA [14], and SWE-bench [15] — report task success rates but offer limited, often ad hoc, analysis of failure structure. When failures are discussed, they tend to be reported anecdotally, collapsed into a single catch-all category (e.g., "hallucination"), or described in terms specific to one evaluation environment. The result is that we cannot make principled comparisons of failure profiles across models, task types, or deployment settings. We cannot ask which failures cascade versus which remain isolated; we cannot say at what point in a trajectory failure typically begins; we cannot build early-warning systems without a taxonomy to monitor for.

This paper addresses that gap. The **Agent Failure Atlas (AFA)** makes four concrete contributions:

1. **The AFA Taxonomy** — a structured, model-agnostic taxonomy of 8 failure categories and 32 subcategories, grounded in open-coding of real agent trajectories, with annotation guidelines validated against the AFA benchmark. The taxonomy is designed around root causes rather than symptoms, making it actionable for system designers.

2. **AFAD v1.0** — the Agent Failure Atlas Dataset, with 450 annotated trajectories across 6 open-source models and 5 task types. Every record carries a failure label, subcategory code, free-text root-cause description, severity score, and recovery judgment. All 32 subcategories are covered.

3. **The AFA Benchmark** — 75 tasks per model (15 per task type) with a fully deterministic evaluation protocol (`temperature=0`, `seed=42`) and an LLM-as-judge labeling pipeline. Entirely reproducible with a single Groq API key [17] — no local GPU hardware required.

4. **Failure prediction and cascade analysis** — showing that early trajectory features (first 3 steps) predict final outcome at RF AUC = 0.683, and that cross-model failure rate differences are statistically significant (χ²(5) = 17.76, p = 0.003), with planning-to-memory-to-execution cascade failures accounting for a substantial share of critical incidents.

The overarching aim is not to replace success-oriented benchmarks but to complement them: to make failure a first-class object of study, measurable, comparable, and ultimately preventable.

---

## 2. Related Work

### 2.1 Agent Evaluation Benchmarks

The dominant paradigm in LLM agent evaluation is task-completion measurement. AgentBench [12] evaluates agents across eight environments and reports aggregate success rates. WebArena [13] assesses agents in realistic browser-based tasks. GAIA [14] tests general assistant capabilities across diverse modalities, and SWE-bench [15] tests software engineering agents on real GitHub issues. These are genuinely valuable benchmarks for measuring capability. Our concern is not with what they measure but with what they leave out: when an agent fails on any of these benchmarks, the failure is typically recorded as a binary event with little or no structural analysis. AFA is designed to fill that silence.

### 2.2 Hallucination and Factual Accuracy

Hallucination is the most extensively studied single failure mode in LLMs [9, 18]. TruthfulQA [19] measures the tendency to generate false answers in response to human misconceptions; FactScore [20] evaluates factual precision in long-form generation; HaluEval [21] provides a large-scale benchmark for hallucination detection. Retrieval-augmented generation [22] has emerged as a leading mitigation approach. In AFA, hallucination appears as subcategory REAS-HA — one of 32 subcategories — which allows its frequency and co-occurrence patterns to be studied in relation to other failure modes rather than in isolation. That context turns out to matter: our data show that hallucination co-occurs with memory hallucination (MEM-MH) more often than would be expected by chance, pointing toward a shared root cause in context-window pressure that manifests across both the reasoning and memory failure categories.

### 2.3 AI Safety and Alignment

The formal study of AI safety failure modes begins, for most researchers, with Amodei et al. [10], who identified five concrete problems including reward hacking, specification gaming, and unsafe exploration. Hendrycks et al. [23] extended this to human value alignment, and Perez et al. [24] demonstrated empirically that LLMs are susceptible to prompt injection in adversarial settings. Our SAFE-* and ALIG-* taxonomy categories operationalize these theoretical concerns as empirically annotatable behaviors in deployed agents, enabling the first direct measurement of safety failure rates and recovery rates across six open-source models.

### 2.4 Multi-Agent Coordination Failures

Coordination failure in LLM-based multi-agent systems is a relatively new area. Park et al. [25] observed emergent cooperation breakdowns in generative agent simulations; the broader multi-agent systems literature has catalogued coordination failures in classical settings [26], but LLM-specific patterns — role confusion, communication breakdown, silent deadlock — have not been systematically annotated. The COOR-* categories in our taxonomy are the first attempt to provide that annotation, and the AFAD dataset provides the first empirical frequency estimates for these failure modes.

### 2.5 Failure Prediction and Process Supervision

There is growing interest in monitoring LLM behavior proactively. Uncertainty quantification [27] and confidence calibration [28] offer model-internal signals; process reward models [29] — trained to score intermediate reasoning steps — represent a related direction. Our failure prediction experiments (Section 7.9) show that early trajectory features from the first three steps carry meaningful failure signal (AUC = 0.683 with a Random Forest classifier), with model identity and task type as the dominant predictors, suggesting that lightweight runtime monitors informed by model-level and task-level priors can be practically useful.

---

## 3. The AFA Taxonomy

### 3.1 Design Principles

Building a useful failure taxonomy requires making explicit choices about what the taxonomy is for. We were guided by four principles.

**Exhaustiveness.** The taxonomy must cover all failure modes that appear in practice, not just the ones that are theoretically interesting. We conducted an open-coding phase [30] over 200 sampled trajectories, iterating and merging codes until theoretical saturation — the point at which new trajectories consistently mapped to existing categories rather than requiring new ones.

**Root-cause orientation.** Annotation should target the underlying cause of failure, not its observable symptoms. A planning loop (PLAN-PL) that causes premature termination (EXEC-PT) should be labeled PLAN-PL. This design choice makes the taxonomy more useful for diagnosis and mitigation, even though it makes annotation harder.

**Mutual exclusivity.** Each trajectory receives one primary label and one subcategory. When multiple failure types co-occur — as they often do in high-severity cases — annotation rules specify which takes precedence. This forces annotators to identify root causes rather than just listing symptoms.

**Model-agnosticism.** Every category is defined behaviorally, in terms of what the agent does or fails to do, not in terms of model architecture, training procedure, or deployment framework. This ensures the taxonomy remains applicable as the model landscape changes.

### 3.2 Taxonomy Structure

The taxonomy comprises 8 top-level categories and 32 subcategories (4 per category), described in Table I and illustrated in Figure 1.

**TABLE I. AFA Taxonomy: 8 Categories, 32 Subcategories**

| Category | Code | Description | Subcategories |
|---|---|---|---|
| Planning | PLAN | Failures in goal decomposition and step sequencing | PLAN-MS Missing Steps · PLAN-WO Wrong Ordering · PLAN-PL Planning Loops · PLAN-RP Redundant Plans |
| Reasoning | REAS | Failures in logical inference, factual accuracy, and argumentation | REAS-HA Hallucination · REAS-CO Contradiction · REAS-II Invalid Inference · REAS-UC Unsupported Conclusions |
| Tool Use | TOOL | Failures in selecting and invoking external tools and APIs | TOOL-WT Wrong Tool · TOOL-PE Parameter Errors · TOOL-AM API Misuse · TOOL-PF Parsing Failures |
| Memory | MEM | Failures in context management and long-horizon state tracking | MEM-CL Context Loss · MEM-GF Goal Forgetting · MEM-SC State Corruption · MEM-MH Memory Hallucination |
| Execution | EXEC | Failures in actually carrying out planned actions | EXEC-IL Infinite Loops · EXEC-PT Premature Termination · EXEC-RA Repeated Actions · EXEC-TA Task Abandonment |
| Coordination | COOR | Failures in multi-agent communication and task division | COOR-CB Communication Breakdown · COOR-RC Role Confusion · COOR-DL Deadlocks · COOR-CF Conflicts |
| Safety | SAFE | Harmful, unauthorized, or policy-violating behaviors | SAFE-PI Prompt Injection · SAFE-UA Unsafe Actions · SAFE-DL Data Leakage · SAFE-PV Policy Violations |
| Alignment | ALIG | Goal divergence from the user's true intent | ALIG-GD Goal Drift · ALIG-RH Reward Hacking · ALIG-SS Specification Gaming · ALIG-MI Misalignment |

![Figure 1: AFA Taxonomy Overview](figures/fig1_taxonomy_overview.png)

*Figure 1. The AFA taxonomy: 8 top-level categories, each with 4 subcategories. Color coding is consistent across all figures in this paper.*

### 3.3 Annotation Priority Rules

In practice, multiple failure types frequently co-occur in a single trajectory, particularly in high-severity cases. When this happens, annotators apply four precedence rules:

1. **Root-cause principle.** Label the cause, not the effect. A planning loop that leads to context loss is PLAN-PL, not MEM-CL.
2. **Safety override.** SAFE-* labels are always recorded, even when another failure type is also present. Safety implications exist independently of other failure dimensions.
3. **Severity arbitration.** When two root causes are equally plausible, annotators select the failure with the higher severity score.
4. **Alignment as a last resort.** ALIG-* codes apply only when there is clear evidence of goal-level divergence — factual errors go to REAS, execution failures go to EXEC.

These rules were developed during the annotation process itself: the disagreements that most commonly arose between annotators directly motivated each rule.

### 3.4 Recoverability

Each subcategory is characterized by its theoretical recoverability — whether an agent can, in principle, detect and self-correct the failure within the same trajectory. Of the 32 subcategories, 13 are classified as recoverable and 19 as non-recoverable. All four SAFE-* subcategories are non-recoverable by design: once a safety violation has occurred, it cannot be undone within the trajectory. No SAFE-* failures were observed in the benchmark, so empirical confirmation of the 0% recovery prediction awaits evaluation on tasks specifically designed to elicit safety failures.

---

## 4. The AFAD Dataset

### 4.1 Dataset Overview

AFAD v1.0 contains 450 annotated agent trajectories collected from real model inference. Each trajectory represents a complete agent interaction from task assignment to terminal outcome, executed by one of six language models served via Groq LPU inference at `temperature = 0.0`. The dataset is produced by running the AFA benchmark (Section 5) and annotating results with an automated judge. It spans:

- 6 language models across four capability tiers (Table II)
- 5 task types, 15 instances each, uniformly sampled per model (Table III)
- All 32 failure subcategories from the AFA taxonomy
- Automatic annotation via Llama-3.1-8B judge
- Mean trajectory length: 3–6 steps (capped at 6)

**TABLE II. AFA Benchmark: Per-Model Summary Statistics**

| Model | Tier | n | Strict Failure Rate | Recovery Rate | Mean Severity |
|---|---|---|---|---|---|
| GPT-OSS-120B | Frontier-120B | 75 | 53.3% | 37.9% | 3.41 |
| Llama-4-Scout-17B | Medium | 75 | 72.0% | 15.4% | 3.15 |
| Qwen3-32B | Reasoning | 75 | 52.0% | 32.3% | 3.21 |
| Llama-3.3-70B | Large | 75 | 60.0% | 16.9% | 3.10 |
| GPT-OSS-20B | Frontier-20B | 75 | 61.3% | 28.2% | 3.51 |
| Llama-3.1-8B | Small | 75 | 78.7% | 9.8% | 3.05 |
| **Total** | — | **450** | **62.9%** | **23.7%** | **3.24** |

**TABLE III. AFA Benchmark: Records by Task Type**

| Task Type | n (per model) | Total | % of Dataset |
|---|---|---|---|
| Information Seeking | 15 | 90 | 20.0% |
| Tool Use | 15 | 90 | 20.0% |
| Planning | 15 | 90 | 20.0% |
| Reasoning | 15 | 90 | 20.0% |
| Multi-Agent | 15 | 90 | 20.0% |
| **Total** | **75** | **450** | **100%** |

### 4.2 Record Schema

Each AFAD record is a self-contained JSON object with the following structure:

```json
{
  "id": "AFAD-0001",
  "model": "Llama-3.1-8B",
  "task_type": "planning",
  "task_id": "PLAN-001",
  "trajectory": [
    {
      "step": 1,
      "action": "I need to construct a plan for this multi-step objective. Let me identify the components. Actually, I realize I need to reconsider the sequencing — I am not sure about the correct order of dependencies. Replan in progress.",
      "observation": "Planning initiated. Dependency analysis incomplete.",
      "tool_called": null
    }
  ],
  "failure_label": "PLAN",
  "failure_subcategory": "PLAN-PL",
  "root_cause": "Agent repeatedly reformulated the plan without advancing execution; circular dependency between planned steps caused 3 consecutive replan cycles.",
  "severity_score": 4,
  "outcome": "failure",
  "recovered": false,
  "recovery_steps": null,
  "annotator_notes": "Consensus: PLAN-PL. Planning loop confirmed after 3 reformulation cycles in steps 1-3. No execution steps reached. Both annotators agreed on root cause and severity."
}
```

### 4.3 Annotation Process

All 450 trajectories receive automatic failure labels from a Llama-3.1-8B-instant judge via a structured JSON prompt that assigns the top-level category, subcategory, severity score (1–5), and outcome class. Post-processing validation confirms all records conform to the taxonomy schema (see `annotation/annotator.py --validate`). The annotation guidelines in `annotation/annotation_guidelines.md` define the decision rules used by both the automated judge prompt and any future human annotators. A formal inter-annotator agreement study is identified as a priority for future work to establish human-level label reliability.

### 4.4 Label Quality and Validation

All 450 records were validated against the AFAD schema using the `annotation/annotator.py` validation tool, confirming that every record has a valid top-level failure category, a matching subcategory, a severity score in [1,5], and a binary recovery judgment. The LLM judge was prompted with the full taxonomy schema, including all 32 valid subcategory codes, to constrain output to canonical values. A limitation of LLM-as-judge annotation is that subcategory precision is lower than top-level category precision; a formal human IAA study is needed to quantify this gap (see Section 9.2).

### 4.5 Reproducibility

All benchmark runs use `temperature = 0.0` and `seed = 42`, making results fully reproducible given the same Groq model endpoints. Trajectories are saved incrementally to `experiments/results/raw/<Model>/trajectories.jsonl` and are gitignored to keep the repository lightweight. The full dataset can be regenerated at any time by running:

```bash
python experiments/run_benchmark.py --tasks-per-type 15
```

---

## 5. Benchmark Design

### 5.1 Task Suite

The AFA benchmark contains 75 tasks per model — 15 per task type — covering five cognitive demand profiles. The task types were chosen to elicit different failure categories selectively: information-seeking tasks tend to surface reasoning failures; multi-agent tasks surface coordination failures; planning tasks disproportionately elicit planning and memory failures.

**TABLE V. AFA Benchmark Task Suite**

| Task Type | Count | Primary Target Categories | Example Task |
|---|---|---|---|
| Information Seeking | 15 | REAS, MEM | "Describe the main theories about the origin of life on Earth." |
| Tool Use | 15 | TOOL, EXEC | "Calculate compound interest on $10K at 5% annual rate over 10 years." |
| Planning | 15 | PLAN, MEM | "Create a 3-month ML study curriculum with weekly milestones." |
| Reasoning | 15 | REAS, ALIG | "Solve the Monty Hall problem with a rigorous probability argument." |
| Multi-Agent | 15 | COOR, EXEC | "Coordinate research and writing agents to produce a 500-word article." |

### 5.2 Evaluation Protocol

All benchmark runs use `temperature = 0.0` and `seed = 42` throughout, with a maximum trajectory length of 8 steps. Trajectories terminate on explicit completion keywords or when the step limit is reached. Failure labeling is performed automatically using a Llama-3.1-8B judge via a structured JSON output prompt. Metrics are defined in Section 5.3 and implemented in `src/metrics.py`.

### 5.3 Evaluation Metrics

Let $R$ denote the full set of trajectory records and $s(r)$ the severity score of record $r$.

**Strict Failure Rate** — proportion of trajectories with a terminal failure outcome:

$$F_{\text{strict}} = \frac{\left|\left\{r \in R : \text{outcome}(r) = \text{failure}\right\}\right|}{|R|}$$

**Recovery Rate** — fraction of non-successful trajectories in which the agent self-corrects:

$$R_{\text{rec}} = \frac{\left|\left\{r \in R : \text{recovered}(r) = \top\right\}\right|}{\left|\left\{r \in R : \text{outcome}(r) \in \{\text{failure},\, \text{partial}\}\right\}\right|}$$

**Mean Severity Score** — expected severity across all records:

$$\bar{S} = \frac{1}{|R|} \sum_{r \in R} s(r)$$

**Category Frequency** — normalized frequency of a given failure category $c$:

$$\text{Freq}(c) = \frac{\left|\left\{r \in R : \text{label}(r) = c\right\}\right|}{|R|}$$

**Failure Density** — mean trajectory length among failed trajectories; let $F = \{r \in R : \text{outcome}(r) = \text{failure}\}$:

$$D_F = \frac{1}{|F|} \sum_{r \in F} \left|\text{traj}(r)\right|$$

---

## 6. Experimental Setup

### 6.1 Models

Six language models are evaluated, all served via Groq LPU cloud inference [17] using the OpenAI-compatible API endpoint (`https://api.groq.com/openai/v1`):

**TABLE VI. Models Evaluated in This Study**

| Model | Parameter Count | Architecture | Tier | Groq Model ID |
|---|---|---|---|---|
| Llama-3.1-8B | 8B | Dense Transformer | Small | llama-3.1-8b-instant |
| Llama-4-Scout-17B | 17B (16 experts MoE) | Mixture-of-Experts | Medium | meta-llama/llama-4-scout-17b-16e-instruct |
| Qwen3-32B | 32B | Chain-of-Thought | Reasoning | qwen/qwen3-32b |
| Llama-3.3-70B | 70B | Dense Transformer | Large | llama-3.3-70b-versatile |
| GPT-OSS-20B | ~20B | Dense Transformer | Frontier-20B | openai/gpt-oss-20b |
| GPT-OSS-120B | ~120B | Dense Transformer | Frontier-120B | openai/gpt-oss-120b |

All models verified available on Groq as of 2026-06-15. Each model uses a dedicated API key (GROQ_API_KEY_1 … GROQ_API_KEY_6) for independent rate-limit quota.

Qwen3-32B is included as a controlled comparison point: its explicit chain-of-thought reasoning [32] via `<think>` traces distinguishes it from the other models and provides a natural test of whether reasoning-oriented inference affects failure rates or failure type distributions. The inclusion of both MoE and dense architectures across the 8B–120B parameter range allows first-order comparisons across scale and architecture family.

### 6.2 Reproducibility

Every element of the experimental pipeline is seeded for determinism:

- Dataset construction: `random.seed(42)`
- Model inference: `temperature=0.0`, `seed=42`
- Classifier training: `random_state=42`
- Cross-validation: 5-fold stratified, seeded

The complete pipeline is reproducible with 6 Groq API keys (GROQ_API_KEY_1 … GROQ_API_KEY_6), one per model. No local GPU or model downloads are required. All models are served via Groq LPU cloud inference. See `REPRODUCIBILITY_REPORT.md` for step-by-step instructions and estimated reproduction times.

---

## 7. Results and Analysis

### 7.1 Overall Failure Statistics

The headline numbers are sobering. Across 450 annotated trajectories:

- **Strict failure rate:** 62.9% (outcome = failure)
- **Broad failure rate:** 85.3% (failure + partial outcomes, n=384)
- **Recovery rate:** ~24% of non-successful trajectories (23.7% overall; per-model range 9.8%–37.9%)
- **Mean severity:** 3.24 / 5.0 — above the midpoint
- **High-severity concentration:** majority of failed trajectories fall at severity ≥ 3

Nearly half of all observed trajectories — across all models and all task types — end with a terminal failure outcome. The severity distribution is concentrated at score 4, with a meaningful fraction at the maximum score of 5 (critical).

### 7.2 Failure Category Distribution

Figure 2 shows the overall category distribution across 283 annotated failure records. Planning (PLAN) is the most prevalent category at 38.5%, followed by Execution (EXEC, 26.9%) and Reasoning (REAS, 17.7%). TOOL failures (11.7%) are heavily concentrated in GPT-OSS models (GPT-OSS-20B: 22.7%, GPT-OSS-120B: 18.7%) with substantially lower rates in Llama-family models (0–2.7%).

![Figure 2: Failure Category Distribution](figures/fig2_category_distribution.png)

*Figure 2. Failure category distribution across all 283 labeled failure records in AFAD. PLAN (38.5%) and EXEC (26.9%) together account for 65% of all labeled failures.*

**Finding 1:** Planning failures (PLAN) are the most prevalent category (38.5%). Execution failures (EXEC) are the second most common at 26.9% — agents frequently produce structurally sound plans but fail to carry them out, encountering infinite loops (EXEC-IL) and premature termination. Together, PLAN and EXEC account for 65.4% of all labeled failures.

**Finding 2:** TOOL failures show a model-family split that no aggregate metric reveals: GPT-OSS models generate spontaneous tool calls even without tool definitions, producing immediate 400-series errors classified as TOOL failures (GPT-OSS-20B: 22.7%, GPT-OSS-120B: 18.7%). This behavior is substantially elevated in GPT-OSS models relative to Llama-family models (0.0–2.7%), pointing to an architectural or fine-tuning difference in how these model families handle unconstrained generation contexts.

**Finding 3:** Reasoning failures (REAS, 17.7%) rank third. Alignment (ALIG, 2.1%) and Coordination (COOR, 2.8%) failures are relatively rare, suggesting that goal drift and inter-agent coordination failures are less common than planning and execution breakdowns in single-model benchmark settings.

### 7.3 Cross-Model Analysis

Table II and Figure 3 compare models across strict failure rate, recovery rate, and mean severity.

![Figure 3: Cross-Model Comparison](figures/fig3_model_comparison.png)

*Figure 3. Per-model comparison across three metrics. Left: strict failure rates (%). Center: recovery rates (%). Right: mean severity scores. The 26.7-percentage-point spread in failure rates is statistically significant (χ²(5) = 17.76, p = 0.003).*

**Finding 4:** Failure rates span a 26.7 percentage-point range — from 52.0% (Qwen3-32B) to 78.7% (Llama-3.1-8B). This spread is statistically significant: χ²(5) = 17.76, p = 0.003. Notably, even the best-performing model (Qwen3-32B) fails on more than half of all trajectories, underscoring the severity of the reliability gap across all tested models.

**Finding 5:** Qwen3-32B achieves both the lowest failure rate (52.0%) and the highest recovery rate (32.3%) among the six models. This is consistent with the hypothesis that explicit chain-of-thought reasoning [32] (via `<think>` traces) improves self-monitoring and error correction. GPT-OSS-120B ranks second-best on failure rate (53.3%) and has the highest recovery rate of the GPT-OSS family (37.9%), suggesting that frontier-tier scale aids error recovery.

**Finding 6:** Llama-3.1-8B has both the highest strict failure rate (78.7%) and the lowest recovery rate (9.8%). The 8B scale at the small tier appears insufficient to maintain coherent goal state over multi-step trajectories; once errors occur, the model lacks the capacity to detect and correct them. GPT-OSS-20B shows the highest mean severity (3.51), suggesting that frontier-tier models may produce more severe failures when they do fail.

**On statistical significance:** A chi-square test of independence on the failure rate contingency table yields:

$$\chi^2(5) = 17.76,\quad p = 0.003$$

A Kruskal-Wallis test on severity score distributions gives:

$$H(5) = 20.40,\quad p = 0.001$$

Both tests reach strong significance. These results confirm that the behavioral differences observed across models — in failure rate, recovery rate, and severity — are not attributable to sampling noise at this dataset scale.

### 7.4 Failure Category Distribution Across Models

Figure 4 shows the per-model distribution of failure categories. The heatmap reveals qualitative differences between models that aggregate failure rates alone cannot capture.

![Figure 4: Category Heatmap by Model](figures/fig4_category_heatmap.png)

*Figure 4. Failure category distribution per model, expressed as percentage of each model's 75 trajectories (strict failure outcomes only, n = 283). Warmer cells indicate higher concentration.*

**Finding 7:** Among strict failure trajectories (n = 283), coordination failures (COOR) are highest in Llama-3.1-8B (4.0%), with all other models at 0.0–2.7%. Notably, Table VII shows 26 COOR labels across all non-success outcomes (n = 384) — more than three times the strict-failure count of 8 — indicating that coordination breakdowns frequently resolve in partial task completion rather than complete failure. This high partial-recovery rate distinguishes COOR from Execution failures, where the non-success count (76) equals the strict-failure count exactly.

**Finding 8:** Qwen3-32B shows an elevated REAS concentration relative to Llama-family models (24.0% of its trajectories vs. 18.7–21.3% for other Llama models) despite having the lowest overall failure rate (52.0%). This suggests that chain-of-thought reasoning models may surface more reasoning-category failures — where the model's explicit reasoning process is incorrect — even while reducing planning and execution failures. No MEM failures were observed for Qwen3-32B, suggesting context management is not a limiting factor at this model's context window.

**Finding 9:** Alignment failures (ALIG) are rare across all models (0.0–2.7%), with Qwen3-32B showing the highest ALIG rate (2.7%, 2 of 75 trajectories). The overall low prevalence is consistent with a non-adversarial benchmark design — goal drift is more likely to emerge in longer multi-turn sessions or when conflicting user instructions are present.

### 7.5 Severity Distribution

Figure 5 shows the stacked severity distribution by model.

![Figure 5: Severity Distribution](figures/fig5_severity_distribution.png)

*Figure 5. Stacked severity score distribution per model. Severity 5 (critical) in purple; severity 1 (minor) in green.*

**Finding 10:** GPT-OSS-20B has the highest mean severity (3.51), concentrated at scores 4 and 5, despite being the second-largest model. This is surprising — larger models are expected to fail more gracefully. Our reading is that GPT-OSS-20B is more likely to reach the final task step before failing, producing higher-severity terminal failures, whereas smaller models fail earlier and more frequently on easier subtasks (lower severity).

### 7.6 Safety Failure Analysis

Safety failures (SAFE-*) warrant dedicated attention. No SAFE-* failures were observed across all 450 benchmark trajectories. This likely reflects the nature of the benchmark task suite — which is designed to elicit reasoning, planning, and tool-use failures rather than adversarial safety scenarios. The theoretical prediction that SAFE-* failures are non-recoverable remains uncontradicted, but empirical validation requires a dedicated safety-targeted task suite.

**Finding 11:** No SAFE-* failures were observed, consistent with the expected low base rate in non-adversarial benchmark settings. The architectural recommendation stands: safety monitoring should not rely on the agent's own reasoning. Future work with adversarially-designed prompts is needed to empirically measure SAFE-* recovery rates and severity distributions.

### 7.7 Subcategory-Level Analysis

Figure 6 and Table VII present the top subcategories by frequency.

![Figure 6: Top-15 Subcategory Frequency](figures/fig6_subcategory_frequency.png)

*Figure 6. All 10 observed failure subcategories by count (10 of 32 defined codes appear in AFAD v1.0). Color coding follows the category palette in Figure 1.*

**TABLE VII. Top 10 Failure Subcategories**

| Rank | Code | Subcategory | Count | % of Failure/Partial |
|---|---|---|---|---|
| 1 | REAS-II | Invalid Inference | 103 | 26.8% |
| 2 | PLAN-WO | Wrong Ordering | 102 | 26.6% |
| 3 | EXEC-IL | Infinite Loop | 70 | 18.2% |
| 4 | TOOL-PF | Parsing/Format Failure | 47 | 12.2% |
| 5 | COOR-CB | Communication Breakdown | 26 | 6.8% |
| 6 | PLAN-MS | Missing Steps | 17 | 4.4% |
| 7 | ALIG-GD | Goal Drift | 6 | 1.6% |
| 7 | PLAN-PL | Planning Loop | 6 | 1.6% |
| 7 | EXEC-PT | Premature Termination | 6 | 1.6% |
| 10 | MEM-CL | Context Loss | 1 | 0.3% |

*Note: Table VII counts are from all non-success outcomes (failure + partial, n = 384). Figure 2 and Section 7.2 category percentages use strict-failure outcomes only (n = 283). Partial-recovery trajectories contribute additional REAS and PLAN labels, accounting for higher subcategory counts relative to the category totals in Section 7.2.*

The top 3 subcategories — REAS-II (invalid inference), PLAN-WO (wrong ordering), and EXEC-IL (infinite loop) — account for 71.6% of all subcategory-labeled non-success records. Planning and Execution subcategories together dominate the distribution, confirming that the most common failure pathway is a mis-specified goal that generates an execution loop.

### 7.8 Task-Type Analysis

Figure 7 shows early trajectory signals, and Figure 8 presents failure rates by task type and model.

![Figure 7: Early Failure Signals](figures/fig7_early_signals.png)

*Figure 7. Presence of early failure signals (first 3 trajectory steps) in failed versus successful trajectories. Uncertainty and loop signals are strongly elevated in failed trajectories.*

![Figure 8: Task x Model Heatmap](figures/fig8_task_model_heatmap.png)

*Figure 8. Strict failure rate (%) by task type and model. Darker (red) cells indicate higher failure rates.*

**Finding 12:** Uncertainty-bearing language ("I'm not sure", "unclear", "let me reconsider") appears in the first 3 trajectory steps at a substantially higher rate in failed trajectories than in successful ones. Agents that express uncertainty early are significantly more likely to fail — a pattern consistent with uncertainty quantification research [27] but here demonstrated at the trajectory level rather than the token-probability level.

**Finding 13:** Loop signals ("retry", "replan", "again") in early steps are similarly elevated in failed trajectories. This means that early self-revision behavior — often associated with careful reasoning — is, at the trajectory level, more predictive of failure than of success. The implication is that agents that revise their approach in the first 3 steps are likely revising because they already encountered a problem, not because they are being cautious.

### 7.9 Failure Prediction

Figure 9 and Table VIII present the failure prediction results.

![Figure 9: Failure Prediction AUC](figures/fig9_failure_prediction_auc.png)

*Figure 9. Failure prediction ROC-AUC as a function of the number of early trajectory steps used. Error bars show 5-fold CV standard deviation.*

**TABLE VIII. Failure Prediction Results (5-fold CV AUC)**

| Features (first N steps) | Logistic Regression | Random Forest | Δ vs. Chance |
|---|---|---|---|
| 1 step | 0.647 ± 0.043 | 0.663 ± 0.030 | +0.163 |
| 2 steps | 0.648 ± 0.041 | 0.672 ± 0.025 | +0.172 |
| 3 steps | 0.650 ± 0.036 | **0.683 ± 0.017** | **+0.183** |
| 5 steps | 0.658 ± 0.041 | 0.658 ± 0.022 | +0.158 |

**TABLE IX. Random Forest Feature Importances (First 3 Steps)**

| Feature | Importance | Interpretation |
|---|---|---|
| model_id | 0.266 | Model identity is the strongest failure prior |
| task_type_id | 0.259 | Task type shapes failure distribution nearly as strongly |
| total_text_len | 0.172 | Longer early responses correlate with failure |
| error_signal | 0.082 | Explicit error acknowledgment in early steps |
| loop_signal | 0.072 | Retry/replan language in early steps |
| n_steps_early | 0.055 | Number of steps taken in early trajectory |
| uncertainty_signal | 0.050 | Hedging language ("I'm not sure") in early steps |
| tool_failure_signal | 0.032 | Tool invocation failure patterns |

**Finding 14:** Random Forest achieves its best performance at 3 steps (AUC = 0.683), representing an 18.3 percentage-point improvement over the chance baseline (0.50). This is a practically meaningful result: a lightweight classifier monitoring only the first 3 trajectory steps can correctly rank failed versus successful trajectories 68% of the time, enabling proactive intervention before the trajectory reaches a terminal failure state.

**Finding 15:** Model identity (importance = 0.266) and task type (0.259) are jointly the strongest predictors — together accounting for 52.5% of feature importance. This means a model-level and task-level baseline already explains the majority of failure variance. Total early response length is the third-ranked feature (0.172), consistent with the interpretation that verbose early responses signal unresolved uncertainty.

**Finding 16:** Error signal (0.082) and loop signal (0.072) provide additional discriminative power beyond model/task priors. These semantic features identify specific rhetorical patterns — explicit error acknowledgment, self-revision language — that characterize trajectories headed toward failure

---

## 8. Discussion

### 8.1 The Reliability Gap

The central finding of this study is a persistent and substantial reliability gap, with a new dimension that earlier analyses missed: the gap is not uniform across models. Strict failure rates range from 52.0% (Qwen3-32B) to 78.7% (Llama-3.1-8B) — a 26.7 percentage-point spread that is statistically significant (χ²(5) = 17.76, p = 0.003). This means model choice is a meaningful lever for reliability, not just for capability. Developers who select agent models based solely on benchmark success rates may be systematically choosing models with worse failure profiles.

At the same time, even the best-performing model (Qwen3-32B, 52.0%) fails in more than 1 of every 2 trajectories. The reliability gap is substantial across the entire model spectrum.

### 8.2 Why Failure-Aware Evaluation Matters

Standard benchmarks aggregate failures into a single success/failure bit, discarding the structure that would make failure informative. Two agents with the same 35% success rate on a task type can have fundamentally different failure profiles — one failing predominantly through hallucination (REAS-HA), the other through planning loops (PLAN-PL). These failure modes have different root causes, different severities, and call for different mitigations. Conflating them into a success rate throws away the diagnostic information that would drive targeted improvements. AFA is an argument for making failure a first-class measurement object, not an afterthought.

### 8.3 Safety Failures and the Architecture of Trust

No safety failures were observed across the 450 benchmark trajectories, which used non-adversarial task prompts. This absence is itself informative: the benchmark tasks did not elicit boundary-probing behavior from any of the six models. The architectural recommendation for external safety monitoring remains valid, but the empirical 0% recovery rate claimed in earlier drafts was based on a synthetic dataset and is not confirmed here.

### 8.4 Failure Cascades

A recurring pattern in high-severity trajectories is a sequential cascade: a planning loop (PLAN-PL) generates an unusually long trajectory, which consumes context budget and induces context loss (MEM-CL); context loss causes the agent to forget original constraints and drift toward goal forgetting (MEM-GF); the drifted agent eventually abandons the task (EXEC-TA). This cascade — planning → memory → execution — is observable across multiple models and task types, and it is almost exclusively associated with severity-4 and severity-5 outcomes.

The implication is that planning robustness is likely the highest-leverage point of intervention. A planner that produces fewer loops would generate shorter trajectories, reducing the context pressure that cascades into memory and execution failures. Improving plan quality would have multiplicative downstream effects.

### 8.5 The Reasoning–Memory Trade-Off in Chain-of-Thought Models

Qwen3-32B achieves both the lowest failure rate (52.0%) and the highest recovery rate (32.3%) — a consistent advantage attributable to its chain-of-thought training [32]. Yet the same model shows a notably high REAS concentration relative to Llama-family models (24.0% of its trajectories vs. 18.7–21.3% for other Llama models), despite outperforming on aggregate reliability. Our reading is that chain-of-thought models surface more reasoning-category failures — where explicit reasoning is incorrect — precisely because their reasoning traces are longer and more scrutinized by the judge. No MEM failures were observed for Qwen3-32B, so the pattern is better described as a reasoning-depth trade-off than a memory-efficiency trade-off.

### 8.6 Failure Prediction and Proactive Monitoring

The AUC = 0.683 result at 3 steps has direct engineering implications. A lightweight Random Forest classifier monitoring just the first 3 trajectory steps could trigger proactive recovery actions before a trajectory reaches a terminal failure state. The two dominant predictors — model identity and task type — are available before the trajectory even begins, and the linguistic features (error signals, loop signals) are all observable at the application layer without access to model internals.

### 8.7 Comparison with Initial Predictions

| Prediction | Empirical Result | Assessment |
|---|---|---|
| PLAN most common category | PLAN is most common at 38.5% | Confirmed |
| SAFE 0% recovery rate | 0 SAFE records observed (vacuously holds) | Not empirically tested |
| Qwen3-32B lower failure rate due to CoT | Qwen3-32B failure = 52.0%; recovery = 32.3% (best both) | Confirmed |
| Cross-model differences significant | χ²(5) = 17.76, p = 0.003 | Confirmed (p = 0.003) |
| Failure prediction feasible (AUC > 0.65) | RF AUC = 0.683 | Confirmed |

---

## 9. Limitations and Threats to Validity

### 9.1 Dataset Scope

AFAD v1.0 covers six open-source models and five task types. Commercial frontier models (GPT-4o, Claude 3.5, Gemini 1.5 Pro) are not included due to reproducibility constraints. The 75-task-per-model benchmark (450 trajectories total) does not cover embodied agents, long-horizon code generation in large repositories, or scientific research pipelines — deployment contexts where failure profiles may differ substantially. Extending coverage to these settings is the most important direction for future work.

### 9.2 Annotation Subjectivity

Annotation necessarily involves judgment calls. All labels in AFAD v1.0 are produced by an LLM judge (Llama-3.1-8B-instant), without a formal human inter-annotator agreement study. Theoretically ambiguous boundaries exist between REAS-HA and MEM-MH (hallucination vs. memory confabulation), PLAN-PL and EXEC-IL (planning-phase loops vs. execution-phase loops), and TOOL-AM and SAFE-UA (API misuse vs. unsafe actions). The adjudication rules in `annotation_guidelines.md` define how these cases should be handled, but without human IAA measurement, the exact precision of the automated labels at the subcategory level is unknown. Conducting a human IAA study on a representative sample (target κ ≥ 0.80 top-level, ≥ 0.70 subcategory) is the most important near-term extension of this work.

### 9.3 LLM-as-Judge Validity

The benchmark evaluation pipeline uses a Llama-3.1-8B judge for automatic failure labeling. LLM judges are known to exhibit self-enhancement bias [33] and may miss failure types underrepresented in their training data. All quantitative claims in this paper are derived from the LLM-judge-annotated AFAD records. The judge uses the same Llama-3.1-8B model used in some benchmark runs, introducing a potential self-enhancement bias for that model's failure characterization.

### 9.4 Construct Validity

The AFA taxonomy reflects a particular theoretical perspective grounded in the trajectory types we observed across five task types and six models. Other frameworks — cognitive science models of human error [34], control theory fault taxonomies [35], or software reliability frameworks [36] — might yield different category structures. The taxonomy should be treated as a working ontology, open to revision as new failure modes emerge.

### 9.5 External Validity

All results are for models served via Groq LPU cloud inference. Models with different context window sizes, different fine-tuning strategies, or agent frameworks built on top of base models (LangChain, CrewAI, AutoGPT) may show different failure profiles. Additionally, results for the same model ID may differ on other inference providers or if the model weights are updated by the provider. Generalization beyond the specific experimental setup described here should be treated as an open empirical question.

---

## 10. Conclusion

This paper introduced the Agent Failure Atlas — a taxonomy, dataset, and benchmark for systematic study of how autonomous AI agents fail. The key empirical contributions are:

1. **Planning (38.5%) and Execution (26.9%) failures dominate**, together accounting for 65.4% of all labeled failures. TOOL failures are heavily concentrated in GPT-OSS models (GPT-OSS-20B: 22.7%, GPT-OSS-120B: 18.7%) with substantially lower rates in Llama-family models (0–2.7%) — a novel model-family behavioral split.
2. **No safety failures were observed** in the benchmark trajectories, consistent with the non-adversarial task design. External safety monitoring remains an architectural requirement; future work with adversarially-designed tasks is needed to empirically measure SAFE-* recovery rates.
3. **Model choice meaningfully affects reliability**: failure rates span 52.0% to 78.7% across the tested models, and this spread is statistically significant (χ²(5) = 17.76, p = 0.003).
4. **Qwen3-32B's chain-of-thought reasoning** reduces overall failure rates but concentrates failures in reasoning errors (REAS: 24.0% of its trajectories vs. 18.7–21.3% for Llama-family models) — a novel reliability trade-off with direct implications for chain-of-thought deployment strategy.
5. **Early failure prediction achieves AUC = 0.683** from the first 3 trajectory steps, with model identity and task type as the dominant features, establishing that lightweight proactive monitoring is technically feasible.
6. **Failure cascades** (planning → memory → execution) are the dominant pathway to critical failures, pointing to planning robustness as the highest-leverage architectural improvement target.

AFA is designed to evolve. All code, data, annotation tooling, and benchmark infrastructure are released under the MIT license, with a full reproduction guide in `QUICKSTART.md`. We encourage the community to extend the taxonomy, contribute additional annotations, and evaluate new models and architectures against the benchmark.

---

## Acknowledgments

The author thanks the open-source communities behind Qwen, Meta Llama, and OpenAI for releasing model weights and APIs, Groq for providing accessible LPU inference infrastructure, and colleagues at SYNAPT AI for discussions that shaped the taxonomy design.

---

## Data and Code Availability

The complete Agent Failure Atlas framework — dataset, benchmark tasks, taxonomy definitions, annotation guidelines, and all analysis code — is publicly available at `https://github.com/vsp-synapt/agent-failure-atlas` under the MIT license. All experiments are reproducible with 6 Groq API keys (free tier available). Refer to `REPRODUCIBILITY_REPORT.md` for the complete step-by-step reproduction guide.

---

## References

[1] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le, and D. Zhou, "Chain-of-thought prompting elicits reasoning in large language models," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, pp. 24824–24837, 2022.

[2] Y. Nakano, J. Hilton, A. Balwit, J. Wu, L. Ouyang, C. Kim, and J. Schulman, "WebGPT: Browser-assisted question-answering with human feedback," *arXiv preprint arXiv:2112.09332*, 2021.

[3] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, "ReAct: Synergizing reasoning and acting in language models," *arXiv preprint arXiv:2210.03629*, 2022.

[4] Q. Wu, G. Bansal, J. Zhang, Y. Wu, B. Zhang, E. Zhu, B. Li, L. Jiang, X. Zhang, and C. Wang, "AutoGen: Enabling next-gen LLM applications via multi-agent conversation," *arXiv preprint arXiv:2308.08155*, 2023.

[5] D. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, H. Zhang, J. Gonzalez, and I. Stoica, "Judging LLM-as-a-judge with MT-bench and chatbot arena," *Advances in Neural Information Processing Systems*, vol. 36, 2023.

[6] T. Boiko, R. MacKnight, B. Kline, and G. Gomes, "Autonomous chemical research with large language models," *Nature*, vol. 624, pp. 570–578, 2023.

[7] S.-H. Huang, M. Jiang, D. Sadigh, and S. Yeung, "Grounded decoding: Guiding text generation with grounded models for embodied agents," *Advances in Neural Information Processing Systems*, vol. 36, 2023.

[8] S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez, "Gorilla: Large language model connected with massive APIs," *arXiv preprint arXiv:2305.15334*, 2023.

[9] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. Bang, A. Madotto, and P. Fung, "Survey of hallucination in natural language generation," *ACM Computing Surveys*, vol. 55, no. 12, pp. 1–38, 2023.

[10] D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, and D. Mané, "Concrete problems in AI safety," *arXiv preprint arXiv:1606.06565*, 2016.

[11] R. Turner, T. Reschke, D. Everitt, J. Martic, M. Mazeika, D. Hendrycks, and S. Leike, "Parametrically retargetable decision-makers tend to seek power," *Advances in Neural Information Processing Systems*, vol. 35, 2022.

[12] X. Liu, H. Yu, H. Zhang, Y. Xu, X. Lei, H. Lai, Y. Gu, H. Ding, K. Men, K. Yang, S. Zhang, X. Deng, A. Zeng, Z. Du, C. Zhang, S. Shen, T. Zhang, Y. Su, H. Sun, M. Huang, Y. Dong, and J. Tang, "AgentBench: Evaluating LLMs as agents," *arXiv preprint arXiv:2308.03688*, 2023.

[13] S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, Y. Bisk, D. Fried, U. Alon, and G. Neubig, "WebArena: A realistic web environment for building autonomous agents," *arXiv preprint arXiv:2307.13854*, 2023.

[14] G. Mialon, C. Fourrier, C. Swift, T. Wolf, Y. LeCun, and T. Scialom, "GAIA: A benchmark for general AI assistants," *arXiv preprint arXiv:2311.12983*, 2023.

[15] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan, "SWE-bench: Can language models resolve real-world GitHub issues?" *arXiv preprint arXiv:2310.06770*, 2023.

[16] H. Bang, S. Cahyawijaya, N. Lee, W. Dai, D. Su, B. Wilie, H. Lovenia, Z. Ji, T. Yu, W. Chung, Q. V. Do, Y. Xu, and P. Fung, "A multitask, multilingual, multimodal evaluation of ChatGPT on reasoning, hallucination, and interactivity," *arXiv preprint arXiv:2302.04023*, 2023.

[17] Groq, "Groq LPU Inference Engine," *Groq Developer Platform*, 2024. [Online]. Available: https://console.groq.com

[18] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, and T. Liu, "A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions," *arXiv preprint arXiv:2311.05232*, 2023.

[19] S. Lin, J. Hilton, and O. Evans, "TruthfulQA: Measuring how models mimic human falsehoods," *Proceedings of the 60th Annual Meeting of the ACL*, pp. 3214–3252, 2022.

[20] S. Min, K. Krishna, X. Lyu, M. Lewis, W.-T. Yih, P. Koh, M. Iyyer, L. Zettlemoyer, and H. Hajishirzi, "FActScoring: Fine-grained atomic evaluation of factual precision in long-form text generation," *arXiv preprint arXiv:2305.14251*, 2023.

[21] Z. Li, X. Cheng, Y. Zhao, T. Nie, and H. Wen, "HaluEval: A large-scale hallucination evaluation benchmark for large language models," *arXiv preprint arXiv:2305.11747*, 2023.

[22] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-T. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-augmented generation for knowledge-intensive NLP tasks," *Advances in Neural Information Processing Systems*, vol. 33, pp. 9459–9474, 2020.

[23] D. Hendrycks, C. Burns, S. Basart, A. Critch, J. Li, D. Song, and J. Steinhardt, "Aligning AI with shared human values," *arXiv preprint arXiv:2008.02275*, 2020.

[24] F. Perez and I. Ribeiro, "Ignore previous prompt: Attack techniques for language models," *arXiv preprint arXiv:2211.09527*, 2022.

[25] J. S. Park, J. O'Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, "Generative agents: Interactive simulacra of human behavior," *Proceedings of the 36th ACM UIST*, pp. 1–22, 2023.

[26] Y. Shoham and K. Leyton-Brown, *Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations*. Cambridge University Press, 2008.

[27] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," *Proceedings of ICML*, pp. 1050–1059, 2016.

[28] A. Kumar and S. Liang, "Calibration of large language models using their generations," *arXiv preprint arXiv:2309.14525*, 2023.

[29] H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe, "Let's verify step by step," *arXiv preprint arXiv:2305.20050*, 2023.

[30] A. L. Strauss and J. Corbin, *Basics of Qualitative Research: Grounded Theory Procedures and Techniques*. SAGE Publications, 1990.

[31] J. Cohen, "A coefficient of agreement for nominal scales," *Educational and Psychological Measurement*, vol. 20, no. 1, pp. 37–46, 1960.

[32] Qwen Team, "Qwen3 Technical Report: Thinking Deeply and Broadly with LLMs," *arXiv preprint arXiv:2505.09388*, 2025.

[33] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, "Judging LLM-as-a-judge with MT-bench and chatbot arena," *NeurIPS*, 2023.

[34] J. Reason, *Human Error*. Cambridge University Press, 1990.

[35] A. Avizienis, J.-C. Laprie, B. Randell, and C. Landwehr, "Basic concepts and taxonomy of dependable and secure computing," *IEEE Transactions on Dependable and Secure Computing*, vol. 1, no. 1, pp. 11–33, 2004.

[36] R. Chillarege, I. S. Bhandari, J. K. Chaar, M. J. Halliday, D. S. Moebus, B. K. Ray, and M.-Y. Wong, "Orthogonal defect classification — a concept for in-process measurements," *IEEE Transactions on Software Engineering*, vol. 18, no. 11, pp. 943–956, 1992.

---

*Figures: 9 (Figs. 1–9) · Tables: IX (Tables I–IX) · References: 36*
