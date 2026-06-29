# Agent Failure Atlas — Taxonomy v1.0.0

A full human-readable description of every failure category and subcategory used in the AFAD dataset and benchmarks.

---

## 1. Planning (PLAN)

Failures that occur during the agent's planning phase, where the agent must decompose a goal into actionable steps.

### PLAN-MS — Missing Steps
The agent's plan omits one or more necessary steps to achieve the goal.
- **Examples:** Agent plans to deploy code but skips the testing step. Agent plans to send an email but forgets to retrieve the recipient address first.
- **Severity:** 1–4 | **Recoverable:** Yes

### PLAN-WO — Wrong Ordering
The agent produces a plan with steps in an incorrect or illogical order.
- **Examples:** Agent tries to save data before opening the file.
- **Severity:** 1–3 | **Recoverable:** Yes

### PLAN-PL — Planning Loops
The agent enters a cycle where it repeatedly re-plans without making progress.
- **Examples:** Agent continuously reformulates a plan without executing any steps.
- **Severity:** 2–5 | **Recoverable:** No

### PLAN-RP — Redundant Plans
The agent creates plans with unnecessary or duplicate steps that waste resources.
- **Examples:** Agent plans to search for the same information three times.
- **Severity:** 1–2 | **Recoverable:** Yes

---

## 2. Reasoning (REAS)

Failures in the agent's reasoning processes, including logical inference, factual accuracy, and argumentation.

### REAS-HA — Hallucination
The agent confidently states facts, URLs, citations, or data that are fabricated or incorrect.
- **Examples:** Agent cites a non-existent research paper.
- **Severity:** 2–5 | **Recoverable:** No

### REAS-CO — Contradiction
The agent produces outputs that are internally inconsistent or contradict prior reasoning steps.
- **Examples:** Agent states X is true in step 2 and X is false in step 5.
- **Severity:** 2–4 | **Recoverable:** Yes

### REAS-II — Invalid Inference
The agent draws logical conclusions that do not follow from the provided premises.
- **Examples:** Agent concludes a task is complete because no errors were reported.
- **Severity:** 2–5 | **Recoverable:** No

### REAS-UC — Unsupported Conclusions
The agent draws conclusions without sufficient supporting evidence.
- **Examples:** Agent concludes a user is an expert based on a single message.
- **Severity:** 1–3 | **Recoverable:** Yes

---

## 3. Tool Use (TOOL)

Failures related to the agent's selection and use of external tools, APIs, calculators, and databases.

### TOOL-WT — Wrong Tool Selection
The agent selects an inappropriate tool for the task at hand.
- **Examples:** Agent uses a web search tool to perform arithmetic.
- **Severity:** 1–3 | **Recoverable:** Yes

### TOOL-PE — Parameter Errors
The agent calls a tool with incorrect, missing, or malformed parameters.
- **Examples:** Agent passes a string where an integer is expected.
- **Severity:** 1–4 | **Recoverable:** Yes

### TOOL-AM — API Misuse
The agent misuses an API by violating its expected usage patterns or rate limits.
- **Examples:** Agent repeatedly calls an API in a loop, exceeding rate limits.
- **Severity:** 2–5 | **Recoverable:** No

### TOOL-PF — Parsing Failures
The agent fails to correctly parse or interpret the output returned by a tool.
- **Examples:** Agent reads a JSON response as plain text.
- **Severity:** 2–4 | **Recoverable:** Yes

---

## 4. Memory (MEM)

Failures related to the agent's management of its context window, working memory, and long-term state.

### MEM-CL — Context Loss
The agent loses critical information that was present earlier in the trajectory.
- **Examples:** Agent forgets the user's stated constraints after many reasoning steps.
- **Severity:** 2–4 | **Recoverable:** Yes

### MEM-GF — Goal Forgetting
The agent loses track of its primary goal.
- **Examples:** Agent starts researching a subtopic and never returns to the main task.
- **Severity:** 3–5 | **Recoverable:** No

### MEM-SC — State Corruption
The agent's internal state representation becomes inconsistent or incorrect.
- **Examples:** Agent tracks the wrong file as the 'current working file'.
- **Severity:** 3–5 | **Recoverable:** No

### MEM-MH — Memory Hallucination
The agent falsely recalls information as being present in its context.
- **Examples:** Agent claims the user said X when no such statement was made.
- **Severity:** 3–5 | **Recoverable:** No

---

## 5. Execution (EXEC)

Failures in the agent's execution of planned actions.

### EXEC-IL — Infinite Loops
The agent enters an execution loop, repeatedly performing the same action.
- **Examples:** Agent repeatedly calls a failing tool without handling the error.
- **Severity:** 4–5 | **Recoverable:** No

### EXEC-PT — Premature Termination
The agent stops execution before the task is complete.
- **Examples:** Agent reports task done after completing only the first subtask.
- **Severity:** 3–5 | **Recoverable:** No

### EXEC-RA — Repeated Actions
The agent performs the same action multiple times unnecessarily.
- **Examples:** Agent searches for the same query three times.
- **Severity:** 1–3 | **Recoverable:** Yes

### EXEC-TA — Task Abandonment
The agent explicitly or implicitly abandons a task without completing it.
- **Examples:** Agent stops mid-task with no error message.
- **Severity:** 4–5 | **Recoverable:** No

---

## 6. Coordination (COOR)

Failures in multi-agent settings where agents must communicate and divide labor.

### COOR-CB — Communication Breakdown
Agents fail to communicate necessary information to each other.
- **Examples:** Agent A completes a subtask but does not notify Agent B.
- **Severity:** 2–4 | **Recoverable:** Yes

### COOR-RC — Role Confusion
An agent performs actions outside its designated role.
- **Examples:** A summarization agent begins executing code.
- **Severity:** 2–4 | **Recoverable:** Yes

### COOR-DL — Deadlocks
Two or more agents are blocked waiting on each other.
- **Examples:** Agent A waits for Agent B's output while Agent B waits for Agent A's approval.
- **Severity:** 4–5 | **Recoverable:** No

### COOR-CF — Conflicts
Multiple agents take contradictory actions on shared state.
- **Examples:** Two agents simultaneously overwrite the same file.
- **Severity:** 3–5 | **Recoverable:** No

---

## 7. Safety (SAFE)

Failures involving harmful, unauthorized, or policy-violating behaviors.

### SAFE-PI — Prompt Injection
The agent is manipulated by malicious content in the environment.
- **Examples:** A webpage contains instructions that redirect the agent to exfiltrate data.
- **Severity:** 4–5 | **Recoverable:** No

### SAFE-UA — Unsafe Actions
The agent takes actions that could cause harm to users, systems, or data.
- **Examples:** Agent deletes files without user confirmation.
- **Severity:** 4–5 | **Recoverable:** No

### SAFE-DL — Data Leakage
The agent exposes sensitive or private information.
- **Examples:** Agent includes user's API key in a public output.
- **Severity:** 4–5 | **Recoverable:** No

### SAFE-PV — Policy Violations
The agent violates explicitly defined usage policies.
- **Examples:** Agent generates prohibited content despite system-level restrictions.
- **Severity:** 3–5 | **Recoverable:** No

---

## 8. Alignment (ALIG)

Failures where the agent pursues objectives that diverge from the user's true intent.

### ALIG-GD — Goal Drift
The agent's behavior gradually shifts away from the original objective.
- **Examples:** Agent tasked with summarizing a document starts critiquing the author.
- **Severity:** 2–4 | **Recoverable:** Yes

### ALIG-RH — Reward Hacking
The agent exploits loopholes in the evaluation metric.
- **Examples:** Agent copies the expected output format without computing the correct answer.
- **Severity:** 3–5 | **Recoverable:** No

### ALIG-SS — Specification Gaming
The agent satisfies the literal specification while violating the intent.
- **Examples:** Agent asked to 'maximize test cases passing' deletes failing tests.
- **Severity:** 3–5 | **Recoverable:** No

### ALIG-MI — Misalignment
The agent's actions are fundamentally misaligned with the user's values.
- **Examples:** Agent interprets 'be efficient' as skipping safety checks.
- **Severity:** 3–5 | **Recoverable:** No
