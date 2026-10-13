# Chapter 6.11 — Evaluation, Safety and LLMOps

[← Section index](README.md) · Prev [6.10](10-agents-agentic-ai.md) · Back to [Roadmap](../README.md) · Priority 🔴

> Evaluation is the bottleneck of every chapter in this section: you cannot improve prompts, retrieval, fine-tunes or agents faster than you can measure them.

## 1. Why LLM evaluation is hard
Open-ended outputs (many valid answers), stochastic generation, benchmark contamination and saturation, sensitivity to prompt wording, no single metric for "good," and high cost of human labeling. Offline scores often mispredict production behavior.

## 2. Evaluation layers
| Layer | Question | Methods |
|---|---|---|
| Unit/component | Does each piece work? | Retrieval Recall@K, tool-argument accuracy, parser success, schema validity |
| Task | Is the output correct for the job? | Exact match, field-level F1, unit tests, executable checks, rubric scoring |
| System/e2e | Does the whole pipeline succeed? | Task success on realistic scenarios, human review, agent traces |
| Safety | Can it be misused or hurt users? | Red-teaming, jailbreak/injection suites, policy tests, bias audits |
| Operational | Is it fast/cheap/reliable? | Latency percentiles, cost/task, error rate, availability |
| Online | Do users benefit? | A/B tests, retention, escalation rate, thumbs, complaint review |

## 3. Building a task eval (workflow)
1. Collect **real inputs** (logs, tickets, documents) and cover rare/hard cases; 50–300 to start; keep a held-out slice and never tune on it.
2. Define the **success criterion** (programmatic if possible; otherwise a written rubric with examples).
3. Label expected outputs/rubric scores; measure inter-annotator agreement.
4. Automate scoring; log model, prompt version, parameters, cost, latency.
5. Report **uncertainty**: bootstrap CI. **Worked example:** 80 items, 64 correct → 80% accuracy; normal-approx standard error `√(0.8·0.2/80)=0.0447`, 95% CI ≈ 80% ± 8.8 points → [71%, 89%]. A new prompt scoring 84% on the same 80 items is **not** distinguishable from 80% — use paired comparisons (McNemar/bootstrap on per-item differences) and more items.
6. Re-run on every model, prompt or retrieval change (regression suite); add every production failure as a new test.

## 4. LLM-as-judge
Use a strong model to grade outputs against a rubric or compare two answers. Known biases (Zheng et al. 2023, MT-Bench): **position bias** (prefers first/second), **verbosity bias**, **self-preference**, sensitivity to rubric wording. Practices: randomize order and average both orders, require reasoning then a score, use pairwise comparisons for subjective quality, give reference answers when available, **calibrate against human labels** (report agreement/Cohen's κ) on a sample, and never use the same model/prompt being tuned as the sole judge. Judges are for scale, not a substitute for ground truth.

## 5. Metric cheat-sheet
| Output type | Metrics |
|---|---|
| Classification/extraction | Accuracy, macro-F1, field exact-match, schema validity |
| Code | pass@k on unit tests (Chen et al. 2021: `pass@k = 1 − C(n−c,k)/C(n,k)`) |
| Summarization/translation | ROUGE/BLEU/chrF (weak), BERTScore, rubric/judge, human preference |
| RAG | Retrieval Recall@K/MRR, faithfulness/groundedness, answer correctness, citation precision |
| Agents | Task success, steps, cost, safety violations, pass^k consistency |
| Safety | Attack success rate, refusal rate on harmful, over-refusal rate on benign |
**Worked pass@k example:** n=10 samples, c=3 correct, k=1: `1 − C(7,1)/C(10,1) = 1 − 7/10 = 0.30`; k=5: `1 − C(7,5)/C(10,5) = 1 − 21/252 = 0.9167`.

## 6. Safety and security
- **Threats:** prompt injection (direct/indirect), jailbreaks, data exfiltration via tools, insecure output handling (XSS/SQL from model text), excessive agency, training-data leakage/PII, model denial-of-service (huge inputs), supply-chain risks (malicious models/tools), bias and harmful content, over-reliance. See OWASP Top 10 for LLM Applications and NIST AI RMF.
- **Defenses (layered):** least privilege and sandboxing; validate/escape model outputs before using them as code, SQL, HTML; allow-lists and schemas for tools; human approval for high-impact actions; separate trusted/untrusted content; input/output moderation and classifiers; rate limits; secrets never in prompts; audit logs; incident response plan; red-team before launch and continuously. Prompt wording alone is not a control.
- **Privacy/compliance:** data minimization, retention settings, vendor data-use terms, PII redaction, regional requirements; documentation (model cards/system cards, risk assessments).
- **Responsible-use checks:** subgroup evaluation, harm review, user disclosure, escalation paths.

## 7. LLMOps (running LLM systems)
| Concern | Practice |
|---|---|
| Versioning | Version prompts, tool schemas, retrieval indexes, models and eval sets together; pin model versions; staged rollouts |
| Observability | Trace every request (prompt, retrieved chunks, tool calls, outputs, tokens, latency, errors); sample for review; dashboards and alerts |
| Cost control | Prompt caching (stable prefix first), shorter context, right-size models (route simple requests to smaller models), batch APIs for offline jobs, output caps, budgets per user/task |
| Reliability | Timeouts, retries with backoff, fallbacks across models/providers, rate-limit handling, graceful degradation, circuit breakers |
| Quality over time | Drift monitoring (inputs, retrieval hit rates, user feedback), periodic re-evaluation, handling model deprecations and changing behavior |
| Governance | Access control, approvals for prompt/tool changes, audit trails |
**Cost arithmetic (illustrative; prices change — check current pricing):** with an example list price of $4 per million input tokens and $20 per million output tokens (a current frontier-tier rate as cached 2026-10-06), a request with 2,000 input and 500 output tokens costs `2,000·4/1e6 + 500·20/1e6 = $0.008 + $0.010 = $0.018`; 1M such requests ≈ $18,000. Cached input tokens are billed at a fraction of the normal input price on providers that support prompt caching, so stable prefixes (system prompt, tools, reference documents) placed first can cut input cost substantially — verify the cache-hit counters in usage metadata rather than assuming.

## 8. Release checklist
- [ ] Eval suite with CI gate (no regression on critical slices)
- [ ] Safety/red-team results reviewed; known limitations documented
- [ ] Monitoring, tracing, alerts, cost budgets in place
- [ ] Human fallback / escalation path; kill switch
- [ ] Data handling reviewed (PII, retention, vendor terms)
- [ ] Rollout plan: shadow → canary → gradual; rollback tested
- [ ] Post-launch review cadence and ownership

## 9. Common mistakes
Shipping on vibes from 5 examples · tuning on the test set · using a judge model without human calibration · ignoring variance from sampling · measuring only accuracy and not cost/latency/safety · benchmark chasing instead of task evals · treating prompts as unversioned config · no tracing so failures can't be reproduced · assuming model upgrades are drop-in.

## Interview Q&A
- *How would you evaluate a RAG chatbot?* Gold questions with evidence passages; retrieval metrics + faithfulness + correctness; human-calibrated judge; online feedback.
- *Pitfalls of LLM-as-judge?* Position/verbosity/self-preference bias; calibrate with humans.
- *Why CIs on eval scores?* Small eval sets give wide intervals; use paired tests to compare variants.
- *How reduce cost?* Caching, routing to smaller models, shorter prompts, batching, output limits, distillation (Ch 6.5).
- *How handle prompt injection?* Architectural controls: least privilege, validation, approvals, data/instruction separation, monitoring.

## Exercises
Compute CI and a paired bootstrap for two prompts on the same 100 items; write a rubric and measure human–judge agreement on 30 outputs; build an injection test suite of 20 attacks for a RAG app; implement a cost dashboard from usage logs.

## Further reading
Zheng et al. (2023) "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena"; Chen et al. (2021) Codex/pass@k; Liang et al. (2022) HELM; OWASP Top 10 for LLM Applications; NIST AI Risk Management Framework; Perez et al. (2022) "Red teaming language models with language models"; Chip Huyen, *AI Engineering*; Hamel Husain's writing on LLM evals.
