# Chapter 6.3 — Prompt Engineering

[← Section index](README.md) · Prev [6.2](02-language-models-llms.md) · Next [6.4 RAG](04-rag.md) · Priority 🔴

> **Version-sensitivity:** prompting best practices shift with each model generation. Prompts tuned for older models are often too prescriptive for newer ones; re-test when you change models. Everything below is a method for *measuring* prompts, not a guarantee.

## 1. Identity & problem
Prompt engineering is the design of model inputs (instructions, context, examples, output format) to elicit reliable behavior **without changing weights**. It is the cheapest and fastest adaptation lever, and the first step before RAG (6.4) or fine-tuning (6.5).

## 2. Evolution
GPT-3 few-shot prompting (Brown et al. 2020) → chain-of-thought (Wei et al. 2022) and "let's think step by step" (Kojima et al. 2022) → self-consistency (Wang et al. 2022) → ReAct reasoning+acting (Yao et al. 2022) → tree/graph-of-thought variants → instruction-tuned models that follow plain instructions → structured outputs / schema-constrained decoding and tool calling (6.9) → reasoning models with built-in thinking and effort controls (less need for hand-written "think step by step"). Automated prompt optimization (APE, DSPy-style compilers, evolutionary search) uses eval sets to search prompts.

## 3. Anatomy of a good prompt
| Element | Purpose | Tip |
|---|---|---|
| Role/context | Who/what the model is working for, why | State the purpose and audience, not just the task |
| Task | Exactly what to produce | Say what *to do*, with success criteria; avoid contradictory rules |
| Inputs | The data to process | Delimit clearly (XML-style tags, fenced blocks); put long documents before the question |
| Examples | Show format and edge cases | 2–5 diverse, correct examples (few-shot); beware overfitting to their surface form |
| Output spec | Format/schema/length | JSON schema or structured-output feature beats "please return JSON" |
| Reasoning guidance | When multi-step logic is needed | Allow thinking space; for models with built-in reasoning, use the effort/thinking setting rather than long scripts |
| Fallbacks | What to do if info is missing | "If the answer isn't in the context, say so" reduces fabrication |

## 4. Techniques
- **Zero-shot / few-shot:** instructions alone vs instructions + examples. Few-shot helps format and borderline decisions.
- **Chain-of-thought (CoT):** elicit intermediate reasoning; improves multi-step tasks on many models, but adds tokens and the visible rationale is not guaranteed to be the true cause of the answer.
- **Self-consistency:** sample several reasoning paths, majority-vote. Worked example: 5 samples answer `[42, 42, 40, 42, 41]` → vote picks 42. Cost ×5.
- **Decomposition / prompt chaining:** split into steps with checks between them (extract → validate → generate).
- **Structured output:** request a schema and validate; retry on failure.
- **Role/persona and style guides:** shape tone; do not rely on them for correctness.
- **Retrieval/context stuffing:** supply sources (6.4); ask for quotes/citations.
- **Prefill/format control:** some newer APIs no longer support assistant prefill; use structured outputs or instructions instead.
- **Prompt caching:** keep the stable prefix (system prompt, tools, examples) first and dynamic content last so repeated requests reuse cached computation (provider-specific; verify cache-hit metrics).

## 5. Security: prompt injection
Untrusted text (web pages, emails, tool results, user documents) can contain instructions that hijack the model ("ignore previous instructions…"). Mitigations: treat retrieved/tool content as **data not instructions**; give least-privilege tools; allow-list tool names and validate arguments in code (not by prompt); require human confirmation for destructive actions; separate trusted (system) and untrusted channels; monitor and red-team (6.11). Prompts alone are not a security boundary. A runnable demonstration of the *code-level* defense (allow-list + schema validation blocking a hijacked call) is in `code/agent_loop_mock.py` (Ch 6.9).

## 6. A systematic workflow
1. Write the **eval set first**: 30–200 representative inputs with expected outputs or a rubric; include hard cases and adversarial ones; keep a held-out test slice.
2. Establish a baseline prompt and score it (exact match / field F1 / rubric / LLM-judge validated against humans).
3. Change **one thing at a time**; log prompt version, model, parameters, scores, cost, latency.
4. Inspect failures by category; fix categories (not single examples).
5. Re-run on the held-out slice before shipping; re-run whenever the model or prompt changes.
Template harness (illustrative; the scoring loop is plain Python — run it with your own `call_model`):
```python
def evaluate(prompt_template, cases, call_model, score):
    results = []
    for c in cases:
        out = call_model(prompt_template.format(**c["inputs"]))
        results.append(score(out, c["expected"]))
    return sum(results) / len(results)
# Compare prompt_v1 vs prompt_v2 on the SAME cases; report the difference with a confidence interval (bootstrap), not a single run.
```
LLM output is stochastic: repeat runs or fix temperature/seed where available, and report variance.

## 7. When prompting is not enough
Signals: scores plateau despite clear prompts; need proprietary knowledge (→ RAG); need a consistent style/format at huge volume or lower latency/cost (→ fine-tune or distill, 6.5); need actions (→ tools, 6.9); need verification (→ programmatic checks).

## 8. Strengths / weaknesses
+ Zero training cost, instant iteration, no data pipeline. − Brittle to wording and model changes, non-deterministic, long prompts cost tokens/latency, hard to guarantee behavior, injection risk.

## 9. Common mistakes
Vague success criteria · piling on contradictory rules · examples that leak the test set · evaluating on 5 cherry-picked cases · changing prompt and model simultaneously · demanding JSON in prose and parsing with regex · trusting model self-reports of confidence · neglecting cost of long prompts · copying prompts tuned for another model.

## 10. Case study (template)
Support-ticket triage into 12 categories: baseline logistic regression on TF-IDF (Ch 2.1); zero-shot prompt with category definitions; few-shot with 2 examples/category; schema-constrained output; compare macro-F1, cost/ticket, latency; analyze confusions; decide between prompt, small fine-tuned encoder, or hybrid.

## Interview Q&A
- *Prompting vs fine-tuning?* Prompting changes inputs only (fast, cheap, brittle); fine-tuning changes weights (needs data, gives consistency/efficiency).
- *Why can CoT help?* Gives the model room to compute intermediate steps in tokens; benefits vary by model and task.
- *How do you reduce hallucination?* Provide sources, require citations/quotes, allow "unknown," verify with tools/checks, constrain outputs.
- *How do you test a prompt change?* Same eval set, A/B with variance and held-out slice.
- *How defend against prompt injection?* Architectural controls (least privilege, validation, human approval) — not just wording.

## Exercises
Build a 50-case eval for a task you care about and compare three prompts; add 10 adversarial cases; measure the effect of example order; write an injection test for a document summarizer.

## Further reading
Brown et al. (2020); Wei et al. (2022) CoT; Wang et al. (2022) self-consistency; Yao et al. (2022) ReAct; Greshake et al. (2023) on indirect prompt injection; OWASP Top 10 for LLM Applications; provider prompting guides (check current versions): Anthropic, OpenAI; Khattab et al. (2023) DSPy.
