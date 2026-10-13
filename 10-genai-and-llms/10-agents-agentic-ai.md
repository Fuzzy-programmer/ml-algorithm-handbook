# Chapter 6.10 — Agents and Agentic AI

[← Section index](README.md) · Prev [6.9](09-tool-calling.md) · Next [6.11 Evaluation, safety, LLMOps](11-evaluation-safety-llmops.md) · Priority 🔴 · Code: [`agent_loop_mock.py`](../code/agent_loop_mock.py)

> Fast-moving area. The patterns (workflow vs agent, ReAct loop, memory, guardrails) are durable; frameworks, model capabilities and platform products change monthly. Product-specific statements are dated and should be re-checked.

## 1. Definitions
- **Workflow:** LLM calls and tools orchestrated by **predefined code paths** (you decide the control flow).
- **Agent:** the LLM **dynamically directs its own process** — deciding which tools to use, in what order, and when it is done — in a loop with environmental feedback.
- **Agentic AI:** systems with goal-directed autonomy over multi-step tasks, often with memory, planning, tool use, and sometimes multiple cooperating agents.
Autonomy is a spectrum; most production value today comes from workflows and constrained agents.

## 2. Lineage
Symbolic agents/planners (STRIPS, BDI) and RL agents (Ch 6.3 specialized) → LLM prompting patterns: **ReAct** (Yao et al. 2022: Thought → Action → Observation), **Reflexion** (Shinn et al. 2023: verbal self-feedback), plan-and-execute, tree search → tool-use fine-tuning and APIs (6.9) → open frameworks (AutoGPT 2023 hype, LangChain/LangGraph, LlamaIndex, AutoGen, CrewAI, provider agent SDKs) → coding agents and computer-use agents → standard tool protocols (MCP) and managed agent platforms. Anthropic's "Building effective agents" (Dec 2024) popularized the workflow-vs-agent distinction and the guidance to use the simplest design that works.

## 3. Patterns (from simplest to most autonomous)
| Pattern | Control flow | Use when |
|---|---|---|
| Single LLM call (+retrieval/tools) | One step | Most tasks |
| **Prompt chaining** | Fixed sequence with gates/validation between steps | Task decomposes cleanly (extract → check → write) |
| **Routing** | Classifier sends input to specialized handler | Distinct categories, different prompts/models |
| **Parallelization** | Sectioning (independent subtasks) or voting (same task, many samples) | Speed or confidence |
| **Orchestrator–workers** | Central LLM decomposes dynamically, workers execute | Unpredictable subtasks (e.g. multi-file code changes) |
| **Evaluator–optimizer** | One generates, another critiques, loop | Clear criteria, iteration improves results |
| **Autonomous agent** | Open-ended loop with tools until done/budget | Open-ended problems, can't predict steps, trusted environment, can verify results |
| **Multi-agent** | Several agents with roles/handoffs/shared state | Parallel exploration, context isolation, large jobs — adds cost and failure modes |

## 4. The agent loop
```
goal + context ─▶ [LLM: plan / choose action] ─▶ tool call ─▶ observation ─┐
          ▲                                                                 │
          └──────────────── update state (messages, memory, scratchpad) ◀───┘
          stop when: goal met · budget/step cap · needs human · error threshold
```
The executable skeleton (message protocol, parallel results, validation, step cap) is in `agent_loop_mock.py` (Ch 6.9). Components:
- **Model + instructions** (system prompt: goal, constraints, tool guidance, when to stop/ask).
- **Tools** (read vs write separation; sandboxes; Ch 6.9).
- **Memory:** short-term = conversation/scratchpad (context window; summarize or compact when long); long-term = files, databases, vector stores, notes the agent reads/writes; retrieval of past experience.
- **Planning/reflection:** explicit plans, TODO lists, self-critique, verification steps (run tests, validate outputs).
- **Guardrails & oversight:** permissions, approval gates, budgets, timeouts, logging, kill switch.
- **Context management:** keep it small (summaries, compaction, tool-result trimming, sub-agents with isolated contexts).

## 5. Math/engineering intuition: compounding error and cost
If each step succeeds independently with probability `p`, an `n`-step task succeeds with `pⁿ`. **Worked example:** p=0.95, n=10 → 0.599; n=20 → 0.358; p=0.99, n=20 → 0.818. Hence: shorten chains, verify intermediate results, add retries/checkpoints, make steps recoverable. Cost/latency scale with steps × context length; a loop that re-sends long history each turn is expensive — use caching and compaction. Token budget example: 15 steps × average 20k input tokens = 300k input tokens per task before caching.

## 6. When to use / not
Use agents when: the path can't be hardcoded, outcomes are verifiable (tests, checks, ground-truth lookups), errors are recoverable, value justifies cost/latency, and you have sandboxing. Good fits: coding with test suites, research with source checking, data analysis in notebooks, IT/ops runbooks with approvals. Avoid when: a deterministic workflow suffices, errors are irreversible or costly, outputs can't be verified, latency is critical, or you can't monitor behavior. **Start simple: single call → workflow → agent**, and add autonomy only when evals prove the benefit.

## 7. Failure modes and mitigations
| Failure | Mitigation |
|---|---|
| Loops / no progress | Step and token caps; detect repeated actions; escalate to human |
| Wrong tool / bad args | Clear tools, validation, error feedback |
| Compounding errors | Checkpoints, verification, smaller steps |
| Prompt injection from tool results/web | Data-vs-instruction separation, least privilege, approval for risky actions |
| Excessive agency / unsafe actions | Permission model, dry-run, allow-lists, audit logs |
| Context overflow / forgotten goal | Compaction, notes/plan files, re-stating goal |
| Cost blowups | Budgets, caching, cheaper sub-agents for reading-heavy work |
| Hallucinated completion ("done" without doing) | Programmatic completion checks, tests, independent verifier |

## 8. Platform options (verify current docs)
Build with: the model API + your loop (max control); an SDK **tool-runner** helper (loop automated, your tools, you host); a batteries-included agent SDK (built-in file/shell tools, you host); or a **managed agent service** where the provider runs the loop and hosts a sandbox (persisted agent definitions, sessions, event streams). Pick by who should own the loop and the sandbox. Multi-agent coordination, scheduled/autonomous runs, outcome-graded sessions and memory stores are emerging platform features; details vary and change rapidly.

## 9. Evaluation
Evaluate **tasks end-to-end** with verifiable outcomes (unit tests passed, correct final answer, state of database) not just the final text; benchmarks: SWE-bench (software issues), WebArena/BrowserGym, τ-bench (tool-agent-user interaction), GAIA, OSWorld — useful but narrow and contaminated; build in-house suites from real tasks. Track: success rate, steps, cost, latency, human interventions, safety incidents, variance over repeated runs (agents are stochastic: run each task several times and report pass@1 and pass^k). Review traces, not only outcomes. Use sandboxed replays of tools for determinism.

## 10. Case study (template)
Data-analysis agent: tools `run_sql` (read-only), `python_sandbox`, `plot`; workflow-first design (plan → query → validate row counts → analyze → draft). Add an evaluator step that re-computes headline numbers independently. Success criteria: numeric answers match analyst-verified ground truth on 50 questions, zero writes, median cost < target. Roll out with human review, then relax.

## Interview Q&A
- *Agent vs workflow?* Model-directed control flow vs code-directed.
- *When not to use an agent?* When a fixed pipeline works or outcomes can't be verified/recovered.
- *What is ReAct?* Interleaving reasoning traces with tool actions and observations.
- *Why do agents fail on long tasks?* Compounding per-step errors, context bloat, drift from the goal.
- *How secure an agent?* Least privilege, sandboxing, approval gates, input/output validation, logging, red-teaming.
- *Single vs multi-agent?* Multi-agent helps with parallelism and context isolation, at the cost of coordination complexity and tokens.

## Exercises
Turn the mock harness into a ReAct loop with a scratchpad and a stopping check; measure success probability vs steps with injected random tool failures; design an approval policy for a finance agent; write an eval of 20 tasks with programmatic checkers.

## Further reading
Yao et al. (2022) ReAct; Shinn et al. (2023) Reflexion; Anthropic (2024) "Building effective agents"; Wang et al. (2023) "A survey on LLM-based autonomous agents"; Jimenez et al. (2023) SWE-bench; Yao et al. (2024) τ-bench; Model Context Protocol docs; OWASP LLM Top 10 (excessive agency); provider agent documentation (check current versions).
