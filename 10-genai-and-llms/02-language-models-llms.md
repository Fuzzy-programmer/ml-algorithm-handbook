# Chapter 6.2 — Language Models → Large Language Models

[← Section index](README.md) · Prev [6.1](01-generative-vs-discriminative.md) · Next [6.3 Prompt engineering](03-prompt-engineering.md) · Priority 🔴 · Code: [`genai_demos.py`](../code/genai_demos.py), [`deep_learning_numpy.py`](../code/deep_learning_numpy.py)

> **Version-sensitivity:** architectures and training recipes here are durable concepts; model names, prices, context sizes and API parameters change quickly and are dated where quoted.

## 1. Identity & problem
A language model assigns probabilities to token sequences: `p(w₁…w_n)=Πp(w_t|w_{<t})`. An LLM is a very large such model (Transformer decoder) pretrained on trillions of tokens, then adapted to follow instructions. Used for chat, code, extraction, reasoning, tool use, retrieval-grounded answers.

## 2. History (what each step fixed)
| Era | Model | Advance | Limitation |
|---|---|---|---|
| 1948– | n-gram LMs (Shannon; Jelinek) | Counts + smoothing | Sparse, short context |
| 2003 | Neural LM (Bengio et al.) | Embeddings generalize | Small context, slow |
| 2013 | word2vec / GloVe | Cheap static embeddings | One vector per word, no context |
| 2014–16 | RNN/LSTM LMs, seq2seq + attention | Context, translation | Sequential, limited range |
| 2017 | **Transformer** (Ch 5.4) | Parallel, long-range | Quadratic attention |
| 2018 | ELMo, **GPT**, **BERT** | Pretrain → fine-tune paradigm | Task-specific fine-tuning |
| 2019–20 | GPT-2, **GPT-3** (few-shot in-context learning), T5, scaling laws (Kaplan 2020) | Scale unlocks prompting | Costly, unreliable |
| 2022 | **Chinchilla** (Hoffmann et al.): compute-optimal ≈ more tokens per parameter (~20 tokens/param rule of thumb); **InstructGPT** (SFT + RLHF) | Better data/compute balance, instruction following | Alignment tax, hallucination |
| 2023– | Open models (LLaMA family etc.), long context, MoE, reasoning-trained models with test-time compute, tool use, multimodality | Capability/efficiency | Evaluation, safety, cost |

## 3. Intuition
Predict the next token, trillions of times. To predict well the model must learn grammar, facts, code patterns and some reasoning heuristics. Instruction tuning and preference training then steer it into an assistant.

## 4. Math
- **Tokenization:** text → subword tokens (BPE: Gage 1994; Sennrich et al. 2016; WordPiece; SentencePiece/Unigram). Vocabulary 30k–250k.
**Worked BPE (executed).** Corpus: `low`×5, `lower`×2, `newest`×6, `widest`×3. Start from characters (+end marker). Merge the most frequent adjacent pair each step: `es`(9) → `est`(9) → `est</w>`(9) → `lo`(7) → `low`(7) → `ne`(6) → `new`(6) → `newest</w>`(6). Result: frequent words become single tokens; rare words (`widest` → `w i d est</w>`) split into pieces.
- **Loss:** cross-entropy `−(1/N)Σ log p_θ(x_t|x_{<t})`; **perplexity** `=exp(loss)`. Uniform over a 50k vocabulary gives perplexity 50,000; good LMs are far lower.
- **Sampling** from logits `z`: `p_i = softmax(z_i/T)`. Executed example, logits `[4,3,2,1,0]`: T=0.5 → `[0.865,0.117,0.016,0.002,0.000]` (entropy 0.66 bits); T=1 → `[0.636,0.234,0.086,0.032,0.012]` (1.44 bits); T=2 → `[0.429,0.260,0.158,0.096,0.058]` (2.01 bits). **Top-k** keeps the k most likely; **top-p (nucleus)** keeps the smallest set with cumulative mass ≥ p (p=0.9 keeps tokens {0,1,2}, mass 0.957). Greedy = T→0. Some current API models fix or reject sampling parameters — check the docs.
- **KV cache:** generation reuses past keys/values. Memory per token `=2·layers·kv_heads·head_dim·bytes`. Executed example (fp16, 32 layers, head_dim 128, 4,096 context, batch 1): 32 KV heads → **2.15 GB**; grouped-query attention with 8 KV heads → **0.54 GB**; batch 16 with GQA → **8.6 GB** — why serving long contexts is memory-bound.
- **Parameters:** a Transformer layer ≈ `12d²`; total ≈ `12Ld²` + embeddings (e.g. d=4096, L=32 ⇒ ≈ 6.4B non-embedding parameters).
- **Scaling laws:** loss falls as a power law in parameters, data and compute; compute ≈ `6·N·D` FLOPs for N parameters and D tokens (a standard approximation).

## 5. Training pipeline
1. **Pretraining** (next-token, web/code/books; deduplication, filtering) →
2. **Supervised fine-tuning / instruction tuning** on demonstrations →
3. **Preference optimization** (RLHF/DPO, Ch 6.6) and safety training →
4. Optionally **reasoning training** with reinforcement learning on verifiable rewards, and tool-use training.
Inference: autoregressive decoding with KV cache, batching, quantization, speculative decoding (Ch 6.5).

## 6. Capabilities and failure modes
In-context learning (few-shot), chain-of-thought, code, translation, tool use. Failures: **hallucination** (fluent falsehoods), outdated knowledge, sensitivity to prompt wording, arithmetic/logic slips, context-position effects ("lost in the middle"), prompt injection, memorization/privacy, bias. Mitigations: retrieval grounding (6.4), tools (6.9), verification, evals (6.11), constrained outputs.

## 7. When to use / not
Use: language-heavy tasks with fuzzy rules, drafting, extraction from messy text, code assistance, semantic search (embeddings), orchestrating tools. Avoid: tasks with strict numeric/logical guarantees when a deterministic program works; high-volume simple classification where a small supervised model is cheaper and calibrated; when errors are costly and unverifiable; private data without appropriate controls.

## 8. Hyperparameters (inference and usage)
Model choice (capability/latency/cost), temperature/top-p (where supported), max output tokens, stop sequences, system prompt, context budget, reasoning/effort setting (where supported), streaming, caching, timeouts/retries.

## 9. Implementation
Executed: BPE and sampling in `genai_demos.py`; attention internals in `deep_learning_numpy.py`. Calling a hosted model (Anthropic Python SDK; **not executed here — no API key in the authoring environment**; model IDs from the SDK reference cached 2026-10-06 and may change):
```python
import anthropic
client = anthropic.Anthropic()               # reads ANTHROPIC_API_KEY or a configured profile
resp = client.messages.create(model="claude-opus-5-5", max_tokens=1024,
                              messages=[{"role": "user", "content": "Summarize BPE in two sentences."}])
print(resp.content[0].text, resp.usage.input_tokens, resp.usage.output_tokens)
```
Open models: Hugging Face `transformers` / `vllm` for local serving.

## 10. Evaluation
Intrinsic: perplexity. Extrinsic: benchmark suites (MMLU-style knowledge, code benchmarks, reasoning sets) — contaminated and saturating, so **build task-specific evals** (Ch 6.11). Measure latency (time-to-first-token, tokens/s), cost per task, safety, robustness to paraphrase.

## 11. Case study (template)
Contract-clause extraction: baseline regex + rules; LLM with a JSON schema; measure field-level F1 on 200 labeled contracts, error taxonomy (omissions, hallucinated values), cost per document; add retrieval/verification where needed.

## 12. Comparison
Encoder-only (BERT-style): classification/embeddings, cheap. Decoder-only (GPT-style): general generation. Encoder–decoder (T5): seq2seq tasks. Small fine-tuned models often beat prompted giants on narrow, high-volume tasks on cost and latency.

## Interview Q&A
- *Why subword tokens?* Open vocabulary with manageable size; rare words decompose.
- *What does temperature do?* Rescales logits before softmax: lower → sharper/more deterministic; higher → flatter/more diverse.
- *Why is the KV cache important?* Avoids recomputing past keys/values; but it dominates memory for long contexts.
- *What is perplexity?* Exponentiated average negative log-likelihood; effective branching factor.
- *Why hallucinations?* The objective rewards plausible continuations, not truth; the model lacks grounding unless given sources/tools.

## Exercises
Train BPE on your own text and inspect the merges; compute KV-cache memory for a model you use; implement top-p sampling and verify the nucleus mass; estimate pretraining FLOPs with `6ND`.

## Further reading
Vaswani et al. (2017); Radford et al. (2018, 2019); Brown et al. (2020); Kaplan et al. (2020); Hoffmann et al. (2022); Ouyang et al. (2022); Sennrich et al. (2016); Jurafsky & Martin, *Speech and Language Processing* (LLM chapters); Karpathy's "Let's build GPT" and "Let's build the GPT tokenizer".
